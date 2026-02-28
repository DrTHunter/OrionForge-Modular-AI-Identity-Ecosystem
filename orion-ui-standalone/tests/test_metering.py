"""Torture tests for src/observability/metering.py — token accounting & cost metering.

Run from project root:
    python -m tests.test_metering

No LLM connection required — exercises dataclasses, pricing, cost computation,
log persistence, aggregation, estimation helpers, and edge cases.
"""

import json
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.observability.metering import (
    TokenUsage,
    CostBreakdown,
    Metering,
    load_pricing,
    get_price,
    compute_cost,
    meter_response,
    meter_from_raw_usage,
    zero_metering,
    estimate_tokens_from_text,
    estimate_tokens_from_messages,
    log_cost_event,
    read_cost_log,
    aggregate_costs,
    reset_pricing_cache,
    set_cost_log_path,
)
from src.llm_client.base import LLMResponse

PASS = 0
FAIL = 0


def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label}  {detail}")


# ─────────────────────────────────────────────
# 1. TokenUsage dataclass
# ─────────────────────────────────────────────
def test_token_usage():
    print("\n=== TokenUsage ===")
    t1 = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    t2 = TokenUsage(prompt_tokens=200, completion_tokens=80, total_tokens=280)

    # Addition
    t3 = t1 + t2
    check("add prompt tokens", t3.prompt_tokens == 300, f"got {t3.prompt_tokens}")
    check("add completion tokens", t3.completion_tokens == 130, f"got {t3.completion_tokens}")
    check("add total tokens", t3.total_tokens == 430, f"got {t3.total_tokens}")

    # is_estimated propagation
    t_est = TokenUsage(prompt_tokens=10, is_estimated=True)
    t_exact = TokenUsage(prompt_tokens=20, is_estimated=False)
    check("estimated OR propagates", (t_est + t_exact).is_estimated is True)
    check("both exact stays exact", (t_exact + t_exact).is_estimated is False)

    # Round-trip
    d = t1.to_dict()
    check("to_dict has prompt_tokens", d["prompt_tokens"] == 100)
    check("to_dict has completion_tokens", d["completion_tokens"] == 50)
    check("to_dict has is_estimated", d["is_estimated"] is False)

    t_back = TokenUsage.from_dict(d)
    check("from_dict round-trip", t_back.prompt_tokens == 100 and t_back.total_tokens == 150)

    # from_dict with missing keys → defaults to 0
    t_empty = TokenUsage.from_dict({})
    check("from_dict empty defaults", t_empty.prompt_tokens == 0 and t_empty.is_estimated is False)

    # Zero usage addition identity
    z = TokenUsage()
    check("zero + something = something", (z + t1).total_tokens == 150)


# ─────────────────────────────────────────────
# 2. CostBreakdown dataclass
# ─────────────────────────────────────────────
def test_cost_breakdown():
    print("\n=== CostBreakdown ===")
    c1 = CostBreakdown(input_cost=0.001, output_cost=0.002, total_cost=0.003)
    c2 = CostBreakdown(input_cost=0.004, output_cost=0.005, total_cost=0.009)

    c3 = c1 + c2
    check("add input_cost", abs(c3.input_cost - 0.005) < 1e-9)
    check("add output_cost", abs(c3.output_cost - 0.007) < 1e-9)
    check("add total_cost", abs(c3.total_cost - 0.012) < 1e-9)
    check("currency preserved", c3.currency == "USD")

    # Round-trip
    d = c1.to_dict()
    check("to_dict rounds", isinstance(d["total_cost"], float))
    c_back = CostBreakdown.from_dict(d)
    check("from_dict round-trip", abs(c_back.input_cost - 0.001) < 1e-6)

    # from_dict empty → zeros
    c_empty = CostBreakdown.from_dict({})
    check("from_dict empty defaults", c_empty.total_cost == 0.0)

    # Cached + training
    c_full = CostBreakdown(
        input_cost=0.01, cached_input_cost=0.005,
        output_cost=0.02, training_cost=0.001, total_cost=0.036,
    )
    d_full = c_full.to_dict()
    check("cached_input in dict", d_full["cached_input_cost"] == 0.005)
    check("training_cost in dict", d_full["training_cost"] == 0.001)


# ─────────────────────────────────────────────
# 3. Metering dataclass
# ─────────────────────────────────────────────
def test_metering():
    print("\n=== Metering ===")
    m1 = Metering(
        usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        cost=CostBreakdown(input_cost=0.01, output_cost=0.02, total_cost=0.03),
        model="gpt-4", provider="openai",
    )
    m2 = Metering(
        usage=TokenUsage(prompt_tokens=200, completion_tokens=80, total_tokens=280),
        cost=CostBreakdown(input_cost=0.03, output_cost=0.05, total_cost=0.08),
        model="gpt-4", provider="openai",
    )

    m3 = m1 + m2
    check("add usage total", m3.usage.total_tokens == 430)
    check("add cost total", abs(m3.cost.total_cost - 0.11) < 1e-9)
    check("model preserved", m3.model == "gpt-4")

    # zero_metering
    z = zero_metering()
    check("zero metering", z.usage.total_tokens == 0 and z.cost.total_cost == 0.0)
    check("zero + m1 = m1", (z + m1).usage.total_tokens == 150)

    # to_dict / from_dict round-trip
    d = m1.to_dict()
    check("to_dict has usage", "usage" in d)
    check("to_dict has cost", "cost" in d)
    check("to_dict model", d["model"] == "gpt-4")

    m_back = Metering.from_dict(d)
    check("from_dict round-trip", m_back.usage.prompt_tokens == 100)

    # Model/provider propagation — first non-empty wins
    m_empty = Metering(model="", provider="")
    m_filled = Metering(model="gpt-3", provider="openai")
    combined = m_empty + m_filled
    check("model propagated from non-empty", combined.model == "gpt-3")


# ─────────────────────────────────────────────
# 4. Pricing registry
# ─────────────────────────────────────────────
def test_pricing():
    print("\n=== Pricing Registry ===")
    tmp = tempfile.mkdtemp()
    pricing_path = os.path.join(tmp, "pricing.yaml")

    try:
        import yaml
        pricing = {
            "openai": {
                "gpt-4": {
                    "input_per_1m": 30.0,
                    "cached_input_per_1m": 15.0,
                    "output_per_1m": 60.0,
                },
                "gpt-3.5": {
                    "input_per_1m": 0.5,
                    "output_per_1m": 1.5,
                },
                "_default": {
                    "input_per_1m": 1.0,
                    "output_per_1m": 2.0,
                },
            },
            "anthropic": {
                "claude-sonnet-4-20250514": {
                    "input_per_1m": 3.0,
                    "output_per_1m": 15.0,
                },
            },
        }
        with open(pricing_path, "w") as f:
            yaml.dump(pricing, f)

        loaded = load_pricing(pricing_path)
        check("load_pricing returns dict", isinstance(loaded, dict))
        check("openai in pricing", "openai" in loaded)

        # Exact match
        ip, cip, op, tp = get_price("openai", "gpt-4", loaded)
        check("exact match input", ip == 30.0, f"got {ip}")
        check("exact match cached", cip == 15.0)
        check("exact match output", op == 60.0)

        # Prefix match — "gpt-3.5-turbo" should match "gpt-3.5"
        ip2, _, op2, _ = get_price("openai", "gpt-3.5-turbo", loaded)
        check("prefix match input", ip2 == 0.5, f"got {ip2}")
        check("prefix match output", op2 == 1.5)

        # Default fallback
        ipd, _, opd, _ = get_price("openai", "unknown-model", loaded)
        check("default fallback input", ipd == 1.0, f"got {ipd}")
        check("default fallback output", opd == 2.0)

        # Unknown provider → zeros
        ipu, _, opu, _ = get_price("unknown-provider", "model", loaded)
        check("unknown provider → 0", ipu == 0.0 and opu == 0.0)

        # Missing file → empty dict
        empty = load_pricing("/nonexistent/pricing.yaml")
        check("missing file → empty", empty == {})

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ─────────────────────────────────────────────
# 5. compute_cost
# ─────────────────────────────────────────────
def test_compute_cost():
    print("\n=== compute_cost ===")
    import yaml
    tmp = tempfile.mkdtemp()

    try:
        pricing = {
            "openai": {
                "gpt-4": {
                    "input_per_1m": 30.0,
                    "cached_input_per_1m": 15.0,
                    "output_per_1m": 60.0,
                },
            },
        }

        usage = TokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
        cost = compute_cost(usage, "openai", "gpt-4", pricing)

        expected_input = 1000 * 30.0 / 1_000_000  # 0.03
        expected_output = 500 * 60.0 / 1_000_000   # 0.03
        check("input_cost correct", abs(cost.input_cost - expected_input) < 1e-9,
              f"got {cost.input_cost}")
        check("output_cost correct", abs(cost.output_cost - expected_output) < 1e-9)
        check("total = input + output", abs(cost.total_cost - (expected_input + expected_output)) < 1e-9)
        check("cached_input_cost = 0 (no cached)", cost.cached_input_cost == 0.0)

        # With cached tokens
        cost2 = compute_cost(usage, "openai", "gpt-4", pricing, cached_tokens=200)
        expected_standard = (1000 - 200) * 30.0 / 1_000_000
        expected_cached = 200 * 15.0 / 1_000_000
        check("cached: standard input reduced", abs(cost2.input_cost - expected_standard) < 1e-9)
        check("cached: cached_input_cost", abs(cost2.cached_input_cost - expected_cached) < 1e-9)

        # Zero usage → zero cost
        z = TokenUsage()
        cost_z = compute_cost(z, "openai", "gpt-4", pricing)
        check("zero usage → zero cost", cost_z.total_cost == 0.0)

        # Cached tokens > prompt tokens → standard_input = 0
        usage_small = TokenUsage(prompt_tokens=100, completion_tokens=0, total_tokens=100)
        cost_over = compute_cost(usage_small, "openai", "gpt-4", pricing, cached_tokens=200)
        check("cached > prompt → input=0", cost_over.input_cost == 0.0)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ─────────────────────────────────────────────
# 6. Estimation helpers
# ─────────────────────────────────────────────
def test_estimation():
    print("\n=== Estimation Helpers ===")

    # estimate_tokens_from_text
    tokens = estimate_tokens_from_text("Hello world, this is a test string.")
    check("estimate text > 0", tokens > 0)
    check("estimate text roughly chars/4",
          abs(tokens - len("Hello world, this is a test string.") // 4) <= 1)

    check("estimate empty text → 0", estimate_tokens_from_text("") == 0)
    check("estimate single char → 1", estimate_tokens_from_text("x") == 1)

    # estimate_tokens_from_messages
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    msg_tokens = estimate_tokens_from_messages(msgs)
    check("estimate msgs > 0", msg_tokens > 0)

    empty_msgs = estimate_tokens_from_messages([])
    check("estimate empty msgs → 1", empty_msgs == 1)

    # Messages with None content
    msgs_none = [{"role": "user", "content": None}]
    check("None content → safe", estimate_tokens_from_messages(msgs_none) >= 1)


# ─────────────────────────────────────────────
# 7. meter_response (exact + estimated)
# ─────────────────────────────────────────────
def test_meter_response():
    print("\n=== meter_response ===")
    import yaml
    pricing = {
        "openai": {
            "gpt-4": {
                "input_per_1m": 30.0,
                "output_per_1m": 60.0,
            },
        },
    }

    # Exact usage
    resp = LLMResponse(
        content="Hello!",
        model="gpt-4",
        usage={"prompt_tokens": 500, "completion_tokens": 100, "total_tokens": 600},
    )
    m = meter_response(resp, provider="openai", pricing=pricing)
    check("metered model", m.model == "gpt-4")
    check("metered provider", m.provider == "openai")
    check("metered prompt_tokens", m.usage.prompt_tokens == 500)
    check("metered not estimated", m.usage.is_estimated is False)
    check("metered cost > 0", m.cost.total_cost > 0)

    # Estimated usage (no usage dict)
    resp_est = LLMResponse(content="Some output text here", model="gpt-4", usage=None)
    messages = [{"role": "user", "content": "A" * 100}]
    m_est = meter_response(resp_est, provider="openai", messages=messages, pricing=pricing)
    check("estimated is_estimated=True", m_est.usage.is_estimated is True)
    check("estimated prompt > 0", m_est.usage.prompt_tokens > 0)
    check("estimated completion > 0", m_est.usage.completion_tokens > 0)


# ─────────────────────────────────────────────
# 8. meter_from_raw_usage
# ─────────────────────────────────────────────
def test_meter_from_raw_usage():
    print("\n=== meter_from_raw_usage ===")
    pricing = {
        "openai": {
            "gpt-4": {"input_per_1m": 30.0, "output_per_1m": 60.0},
        },
    }
    raw = {"prompt_tokens": 300, "completion_tokens": 100, "total_tokens": 400}
    m = meter_from_raw_usage(raw, provider="openai", model="gpt-4", pricing=pricing)
    check("raw metering model", m.model == "gpt-4")
    check("raw metering cost > 0", m.cost.total_cost > 0)
    check("raw metering not estimated", m.usage.is_estimated is False)

    # With cached tokens (OpenAI format)
    raw_cached = {
        "prompt_tokens": 300, "completion_tokens": 100, "total_tokens": 400,
        "prompt_tokens_details": {"cached_tokens": 50},
    }
    m2 = meter_from_raw_usage(raw_cached, provider="openai", model="gpt-4", pricing=pricing)
    check("cached tokens recognized", m2.cost.total_cost <= m.cost.total_cost)

    # Empty usage
    m_empty = meter_from_raw_usage({}, provider="openai", model="gpt-4", pricing=pricing)
    check("empty raw → zero cost", m_empty.cost.total_cost == 0.0)


# ─────────────────────────────────────────────
# 9. Cost log persistence
# ─────────────────────────────────────────────
def test_cost_log():
    print("\n=== Cost Log ===")
    tmp = tempfile.mkdtemp()
    log_path = os.path.join(tmp, "cost_log.jsonl")
    set_cost_log_path(log_path)

    try:
        m1 = Metering(
            usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            cost=CostBreakdown(total_cost=0.01),
            model="gpt-4", provider="openai",
        )
        m2 = Metering(
            usage=TokenUsage(prompt_tokens=200, completion_tokens=80, total_tokens=280),
            cost=CostBreakdown(total_cost=0.02),
            model="claude-sonnet-4-20250514", provider="anthropic",
        )

        # Write events
        ev1 = log_cost_event(m1, agent="astraea", chat_id="chat_001")
        check("event has ts", "ts" in ev1)
        check("event has agent", ev1["agent"] == "astraea")
        check("event has chat_id", ev1["chat_id"] == "chat_001")

        ev2 = log_cost_event(m2, agent="callum", chat_id="chat_002")

        # Read all
        events = read_cost_log()
        check("read all events", len(events) == 2, f"got {len(events)}")

        # Filter by agent
        astraea_events = read_cost_log(agent="astraea")
        check("filter by agent", len(astraea_events) == 1)
        check("correct agent", astraea_events[0]["agent"] == "astraea")

        # Filter by since (future → nothing)
        future_events = read_cost_log(since="2099-01-01T00:00:00+00:00")
        check("since future → empty", len(future_events) == 0)

        # Limit
        limited = read_cost_log(limit=1)
        check("limit=1 works", len(limited) == 1)

        # JSONL format valid
        with open(log_path, "r") as f:
            lines = f.readlines()
        check("JSONL line count", len(lines) == 2)
        for i, line in enumerate(lines):
            try:
                json.loads(line)
                check(f"JSONL line {i} valid", True)
            except json.JSONDecodeError:
                check(f"JSONL line {i} valid", False, line[:50])

    finally:
        set_cost_log_path(None)
        shutil.rmtree(tmp, ignore_errors=True)


# ─────────────────────────────────────────────
# 10. aggregate_costs
# ─────────────────────────────────────────────
def test_aggregate_costs():
    print("\n=== aggregate_costs ===")
    events = [
        {
            "agent": "astraea", "model": "gpt-4", "provider": "openai",
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "cost": {"input_cost": 0.003, "output_cost": 0.003, "total_cost": 0.006,
                     "cached_input_cost": 0.0, "training_cost": 0.0},
        },
        {
            "agent": "callum", "model": "gpt-4", "provider": "openai",
            "usage": {"prompt_tokens": 200, "completion_tokens": 80},
            "cost": {"input_cost": 0.006, "output_cost": 0.0048, "total_cost": 0.0108,
                     "cached_input_cost": 0.0, "training_cost": 0.0},
        },
        {
            "agent": "astraea", "model": "claude-sonnet-4-20250514", "provider": "anthropic",
            "usage": {"prompt_tokens": 150, "completion_tokens": 60},
            "cost": {"input_cost": 0.0005, "output_cost": 0.0009, "total_cost": 0.0014,
                     "cached_input_cost": 0.0, "training_cost": 0.0},
        },
    ]

    agg = aggregate_costs(events)
    check("num_calls", agg["num_calls"] == 3)
    check("total_prompt_tokens", agg["total_prompt_tokens"] == 450)
    check("total_completion_tokens", agg["total_completion_tokens"] == 190)
    check("total_tokens", agg["total_tokens"] == 640)
    check("total_cost > 0", agg["total_cost"] > 0)
    check("by_model has gpt-4", "gpt-4" in agg["by_model"])
    check("by_agent has astraea", "astraea" in agg["by_agent"])
    check("by_agent has callum", "callum" in agg["by_agent"])

    # Empty events
    empty_agg = aggregate_costs([])
    check("empty → zero", empty_agg["total_cost"] == 0.0 and empty_agg["num_calls"] == 0)


# ─────────────────────────────────────────────
# 11. Reset pricing cache
# ─────────────────────────────────────────────
def test_reset_pricing():
    print("\n=== reset_pricing_cache ===")
    reset_pricing_cache()
    check("reset doesn't error", True)
    # Call twice to make sure it's idempotent
    reset_pricing_cache()
    check("double reset ok", True)


# ─────────────────────────────────────────────
if __name__ == "__main__":
    test_token_usage()
    test_cost_breakdown()
    test_metering()
    test_pricing()
    test_compute_cost()
    test_estimation()
    test_meter_response()
    test_meter_from_raw_usage()
    test_cost_log()
    test_aggregate_costs()
    test_reset_pricing()

    print(f"\n{'='*40}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL:
        sys.exit(1)
    else:
        print("All tests passed.")
