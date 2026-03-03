"""Token accounting and USD cost metering for LLM calls.

Central module for all metering logic.  Consumers import data classes
and helper functions; they never compute costs themselves.

Usage:
    from src.observability.metering import meter_response, zero_metering

    m = meter_response(response, provider="openai", messages=messages)
    session = zero_metering()
    session = session + m  # accumulate
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml

from src.llm_client.base import LLMResponse


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------

@dataclass
class TokenUsage:
    """Raw token counts for a single LLM call or an aggregation."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    is_estimated: bool = False

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            is_estimated=self.is_estimated or other.is_estimated,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "is_estimated": self.is_estimated,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TokenUsage":
        return cls(
            prompt_tokens=d.get("prompt_tokens", 0),
            completion_tokens=d.get("completion_tokens", 0),
            total_tokens=d.get("total_tokens", 0),
            is_estimated=d.get("is_estimated", False),
        )


@dataclass
class CostBreakdown:
    """USD cost breakdown for a single call or aggregation.

    Four cost dimensions:
      input_cost        — standard input tokens
      cached_input_cost — input tokens served from provider cache
      output_cost       — output / completion tokens
      training_cost     — fine-tuning / training tokens
    """

    input_cost: float = 0.0
    cached_input_cost: float = 0.0
    output_cost: float = 0.0
    training_cost: float = 0.0
    total_cost: float = 0.0
    currency: str = "USD"

    def __add__(self, other: "CostBreakdown") -> "CostBreakdown":
        return CostBreakdown(
            input_cost=self.input_cost + other.input_cost,
            cached_input_cost=self.cached_input_cost + other.cached_input_cost,
            output_cost=self.output_cost + other.output_cost,
            training_cost=self.training_cost + other.training_cost,
            total_cost=self.total_cost + other.total_cost,
            currency=self.currency,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_cost": round(self.input_cost, 8),
            "cached_input_cost": round(self.cached_input_cost, 8),
            "output_cost": round(self.output_cost, 8),
            "training_cost": round(self.training_cost, 8),
            "total_cost": round(self.total_cost, 8),
            "currency": self.currency,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CostBreakdown":
        return cls(
            input_cost=d.get("input_cost", 0.0),
            cached_input_cost=d.get("cached_input_cost", 0.0),
            output_cost=d.get("output_cost", 0.0),
            training_cost=d.get("training_cost", 0.0),
            total_cost=d.get("total_cost", 0.0),
            currency=d.get("currency", "USD"),
        )


@dataclass
class Metering:
    """Combined usage + cost for a single LLM call or aggregation."""

    usage: TokenUsage = field(default_factory=TokenUsage)
    cost: CostBreakdown = field(default_factory=CostBreakdown)
    model: str = ""
    provider: str = ""

    def __add__(self, other: "Metering") -> "Metering":
        return Metering(
            usage=self.usage + other.usage,
            cost=self.cost + other.cost,
            model=self.model or other.model,
            provider=self.provider or other.provider,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "usage": self.usage.to_dict(),
            "cost": self.cost.to_dict(),
            "model": self.model,
            "provider": self.provider,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Metering":
        return cls(
            usage=TokenUsage.from_dict(d.get("usage", {})),
            cost=CostBreakdown.from_dict(d.get("cost", {})),
            model=d.get("model", ""),
            provider=d.get("provider", ""),
        )


# ------------------------------------------------------------------
# Pricing registry
# ------------------------------------------------------------------

_PRICING_CACHE: Optional[Dict] = None


def _default_pricing_path() -> str:
    """Return the default pricing YAML path relative to project root."""
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    return os.path.join(project_root, "config", "pricing.yaml")


def load_pricing(path: Optional[str] = None) -> Dict:
    """Load the pricing registry YAML.  Returns empty dict if file missing."""
    path = path or _default_pricing_path()
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get_pricing() -> Dict:
    """Lazy-load and cache the pricing registry."""
    global _PRICING_CACHE
    if _PRICING_CACHE is None:
        _PRICING_CACHE = load_pricing()
    return _PRICING_CACHE


def reset_pricing_cache() -> None:
    """Clear the cached pricing (useful for tests)."""
    global _PRICING_CACHE
    _PRICING_CACHE = None


def get_price(
    provider: str, model: str, pricing: Optional[Dict] = None
) -> Tuple[float, float, float, float]:
    """Return (input_per_1m, cached_input_per_1m, output_per_1m, training_per_1m).

    Lookup order:
      1. pricing[provider][model]        (exact match)
      2. pricing[provider][prefix]       (model starts with a known key)
      3. pricing[provider]["_default"]   (provider fallback)
      4. (0.0, 0.0, 0.0, 0.0)           (unknown provider/model)
    """
    pricing = pricing or _get_pricing()
    provider_prices = pricing.get(provider, {})

    # 1. Exact match
    model_prices = provider_prices.get(model)

    # 2. Prefix match — e.g. "gpt-5.2-2025-12-11" matches "gpt-5.2"
    if not model_prices:
        for key, val in provider_prices.items():
            if key.startswith("_"):
                continue
            if isinstance(val, dict) and model.startswith(key):
                model_prices = val
                break

    # 3. Provider default
    if not model_prices:
        model_prices = provider_prices.get("_default")

    if not model_prices:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        float(model_prices.get("input_per_1m", 0.0)),
        float(model_prices.get("cached_input_per_1m", 0.0)),
        float(model_prices.get("output_per_1m", 0.0)),
        float(model_prices.get("training_per_1m", 0.0)),
    )


def compute_cost(
    usage: TokenUsage,
    provider: str,
    model: str,
    pricing: Optional[Dict] = None,
    cached_tokens: int = 0,
) -> CostBreakdown:
    """Compute USD cost from token usage and pricing registry."""
    input_per_1m, cached_per_1m, output_per_1m, training_per_1m = get_price(
        provider, model, pricing
    )
    # Cached tokens are subtracted from regular input
    standard_input = max(0, usage.prompt_tokens - cached_tokens)
    input_cost = standard_input * input_per_1m / 1_000_000
    cached_input_cost = cached_tokens * cached_per_1m / 1_000_000
    output_cost = usage.completion_tokens * output_per_1m / 1_000_000
    training_cost = 0.0  # Set externally for fine-tune jobs
    total = input_cost + cached_input_cost + output_cost + training_cost
    return CostBreakdown(
        input_cost=input_cost,
        cached_input_cost=cached_input_cost,
        output_cost=output_cost,
        training_cost=training_cost,
        total_cost=total,
    )


# ------------------------------------------------------------------
# Estimation helpers
# ------------------------------------------------------------------

def estimate_tokens_from_text(text: str) -> int:
    """Estimate token count from a string using chars/4 heuristic."""
    return max(len(text) // 4, 1) if text else 0


def estimate_tokens_from_messages(messages: List[Dict[str, Any]]) -> int:
    """Estimate prompt token count from a message list using chars/4."""
    total_chars = 0
    for msg in messages:
        content = msg.get("content", "") or ""
        total_chars += len(content)
    return max(total_chars // 4, 1)


# ------------------------------------------------------------------
# Response metering (the main boundary function)
# ------------------------------------------------------------------

def meter_response(
    response: LLMResponse,
    provider: str,
    messages: Optional[List[Dict[str, Any]]] = None,
    pricing: Optional[Dict] = None,
) -> Metering:
    """Create a Metering object from a single LLM response.

    If ``response.usage`` is populated (e.g. OpenAI), uses exact counts.
    If ``response.usage`` is None (e.g. Ollama), estimates via chars/4.
    """
    cached_tokens = 0
    if response.usage:
        usage = TokenUsage(
            prompt_tokens=response.usage.get("prompt_tokens", 0),
            completion_tokens=response.usage.get("completion_tokens", 0),
            total_tokens=response.usage.get("total_tokens", 0),
            is_estimated=False,
        )
        # Some providers (OpenAI, Anthropic) report cached tokens
        cached_tokens = (
            response.usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
            or response.usage.get("cache_read_input_tokens", 0)
            or 0
        )
    else:
        prompt_tokens = (
            estimate_tokens_from_messages(messages) if messages else 0
        )
        completion_tokens = estimate_tokens_from_text(response.content or "")
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            is_estimated=True,
        )

    model = response.model
    cost = compute_cost(usage, provider, model, pricing, cached_tokens=cached_tokens)
    return Metering(usage=usage, cost=cost, model=model, provider=provider)


def meter_from_raw_usage(
    usage_dict: Dict[str, Any],
    provider: str,
    model: str,
    pricing: Optional[Dict] = None,
) -> Metering:
    """Create a Metering from a raw API usage dict (no LLMResponse needed).

    This is the convenience function used by the chat endpoint which makes
    raw httpx calls instead of going through the LLMClient layer.
    """
    usage = TokenUsage(
        prompt_tokens=usage_dict.get("prompt_tokens", 0),
        completion_tokens=usage_dict.get("completion_tokens", 0),
        total_tokens=usage_dict.get("total_tokens", 0),
        is_estimated=False,
    )
    cached_tokens = (
        usage_dict.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        if isinstance(usage_dict.get("prompt_tokens_details"), dict) else 0
    ) or usage_dict.get("cache_read_input_tokens", 0) or 0
    cost = compute_cost(usage, provider, model, pricing, cached_tokens=cached_tokens)
    return Metering(usage=usage, cost=cost, model=model, provider=provider)


def zero_metering() -> Metering:
    """Return a zero-valued Metering for use as an accumulator seed."""
    return Metering()


# ------------------------------------------------------------------
# Cost log persistence — append-only JSONL
# ------------------------------------------------------------------

import json
from datetime import datetime, timezone as _tz

_COST_LOG_PATH: Optional[str] = None


def _default_cost_log_path() -> str:
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    return os.path.join(project_root, "data", "orion", "cost_log.jsonl")


def set_cost_log_path(path: str) -> None:
    global _COST_LOG_PATH
    _COST_LOG_PATH = path


def _get_cost_log_path() -> str:
    return _COST_LOG_PATH or _default_cost_log_path()


def log_cost_event(
    metering: Metering,
    agent: str = "",
    chat_id: str = "",
) -> Dict[str, Any]:
    """Persist a cost event to the append-only JSONL log.

    Returns the event dict that was written.
    """
    path = _get_cost_log_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    event = {
        "ts": datetime.now(_tz.utc).isoformat(),
        "agent": agent,
        "chat_id": chat_id,
        "model": metering.model,
        "provider": metering.provider,
        "usage": metering.usage.to_dict(),
        "cost": metering.cost.to_dict(),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
    return event


def read_cost_log(
    since: Optional[str] = None,
    agent: Optional[str] = None,
    limit: int = 1000,
) -> List[Dict[str, Any]]:
    """Read cost events from the JSONL log.

    *since* is an ISO timestamp; only events after it are returned.
    """
    path = _get_cost_log_path()
    if not os.path.isfile(path):
        return []
    events: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if since and ev.get("ts", "") < since:
                continue
            if agent and ev.get("agent", "") != agent:
                continue
            events.append(ev)
            if len(events) >= limit:
                break
    return events


def aggregate_costs(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate a list of cost events into summary stats."""
    total_input = 0.0
    total_cached = 0.0
    total_output = 0.0
    total_training = 0.0
    total_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    by_model: Dict[str, float] = {}
    by_agent: Dict[str, float] = {}

    for ev in events:
        cost = ev.get("cost", {})
        usage = ev.get("usage", {})
        tc = cost.get("total_cost", 0.0)
        total_input += cost.get("input_cost", 0.0)
        total_cached += cost.get("cached_input_cost", 0.0)
        total_output += cost.get("output_cost", 0.0)
        total_training += cost.get("training_cost", 0.0)
        total_cost += tc
        total_prompt_tokens += usage.get("prompt_tokens", 0)
        total_completion_tokens += usage.get("completion_tokens", 0)
        model = ev.get("model", "unknown")
        by_model[model] = by_model.get(model, 0.0) + tc
        agent = ev.get("agent", "unknown")
        by_agent[agent] = by_agent.get(agent, 0.0) + tc

    return {
        "total_cost": round(total_cost, 6),
        "input_cost": round(total_input, 6),
        "cached_input_cost": round(total_cached, 6),
        "output_cost": round(total_output, 6),
        "training_cost": round(total_training, 6),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "num_calls": len(events),
        "by_model": dict(sorted(by_model.items(), key=lambda x: -x[1])),
        "by_agent": dict(sorted(by_agent.items(), key=lambda x: -x[1])),
    }
