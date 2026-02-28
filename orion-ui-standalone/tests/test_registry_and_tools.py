"""Torture tests for the tool registry, cost tracker, and web search tool.

Run from project root:
    python -m tests.test_registry_and_tools

Exercises tool resolution, registry listing, cost tracker actions,
web search helpers, and the knowledge gate — all offline.
"""

import json
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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


# ═════════════════════════════════════════════
# TOOL REGISTRY TESTS
# ═════════════════════════════════════════════

def test_registry_list():
    print("\n=== Registry — list_registered_tools ===")
    from src.tools.registry import list_registered_tools
    tools = list_registered_tools()
    check("returns list", isinstance(tools, list))
    check("has echo", "echo" in tools)
    check("has memory", "memory" in tools)
    check("has directives", "directives" in tools)
    check("has cost_tracker", "cost_tracker" in tools)
    check("has continuation_update", "continuation_update" in tools)
    check("has web_search", "web_search" in tools)
    check("sorted", tools == sorted(tools))


def test_registry_resolve():
    print("\n=== Registry — _resolve_tool ===")
    from src.tools.registry import _resolve_tool

    # Echo (stateless)
    echo_def = _resolve_tool("echo")
    check("echo resolves", echo_def is not None)
    check("echo type=function", echo_def["type"] == "function")
    check("echo name", echo_def["function"]["name"] == "echo")

    # Directives (stateless)
    dir_def = _resolve_tool("directives")
    check("directives resolves", dir_def is not None)
    check("directives name", dir_def["function"]["name"] == "directives")

    # Cost tracker (stateless)
    ct_def = _resolve_tool("cost_tracker")
    check("cost_tracker resolves", ct_def is not None)

    # Continuation update (stateless)
    cu_def = _resolve_tool("continuation_update")
    check("continuation_update resolves", cu_def is not None)

    # Unknown tool
    unk = _resolve_tool("nonexistent_tool")
    check("unknown tool → None", unk is None)


def test_registry_execute():
    print("\n=== Registry — execute_tool ===")
    from src.tools.registry import execute_tool

    # Echo
    result = execute_tool("echo", {"message": "test123"})
    check("execute echo", result == "test123", f"got {result!r}")

    # Unknown tool → KeyError
    try:
        execute_tool("fake_tool", {})
        check("unknown raises KeyError", False, "no exception")
    except KeyError:
        check("unknown raises KeyError", True)


def test_registry_agent_defs():
    print("\n=== Registry — get_tool_defs_for_agent ===")
    from src.tools.registry import get_tool_defs_for_agent, _PROFILES_DIR
    from pathlib import Path
    import yaml

    # Create a temp profile with allowed_tools
    tmp_profiles = tempfile.mkdtemp()
    orig_profiles = str(_PROFILES_DIR)

    import src.tools.registry as reg
    reg._PROFILES_DIR = Path(tmp_profiles)

    try:
        profile = {"allowed_tools": ["echo", "directives"]}
        with open(os.path.join(tmp_profiles, "test_agent.yaml"), "w") as f:
            yaml.dump(profile, f)

        defs = get_tool_defs_for_agent("test_agent")
        check("returns list", isinstance(defs, list))
        check("correct count", len(defs) == 2, f"got {len(defs)}")
        names = {d["function"]["name"] for d in defs}
        check("has echo", "echo" in names)
        check("has directives", "directives" in names)

        # Profile with no allowed_tools
        with open(os.path.join(tmp_profiles, "empty_agent.yaml"), "w") as f:
            yaml.dump({}, f)
        empty_defs = get_tool_defs_for_agent("empty_agent")
        check("no tools → empty", len(empty_defs) == 0)

        # Non-existent agent → empty
        none_defs = get_tool_defs_for_agent("no_such_agent")
        check("missing profile → empty", len(none_defs) == 0)

        # Profile with unknown tool → skips it
        profile_bad = {"allowed_tools": ["echo", "nonexistent"]}
        with open(os.path.join(tmp_profiles, "bad_agent.yaml"), "w") as f:
            yaml.dump(profile_bad, f)
        bad_defs = get_tool_defs_for_agent("bad_agent")
        check("unknown tool skipped", len(bad_defs) == 1)

    finally:
        reg._PROFILES_DIR = Path(orig_profiles)
        shutil.rmtree(tmp_profiles, ignore_errors=True)


# ═════════════════════════════════════════════
# COST TRACKER TOOL TESTS
# ═════════════════════════════════════════════

def test_cost_tracker_definition():
    print("\n=== CostTrackerTool — definition ===")
    from src.tools.cost_tracker import CostTrackerTool

    defn = CostTrackerTool.definition()
    check("name is cost_tracker", defn["name"] == "cost_tracker")
    check("has parameters", "parameters" in defn)
    check("has action enum",
          "action" in defn["parameters"]["properties"])
    actions = defn["parameters"]["properties"]["action"]["enum"]
    check("6 actions", len(actions) == 6, f"got {len(actions)}")
    for a in ["get_pricing", "set_pricing", "list_models",
              "cost_summary", "cost_log", "session_cost"]:
        check(f"action '{a}' present", a in actions)


def test_cost_tracker_get_pricing():
    print("\n=== CostTrackerTool — get_pricing ===")
    import yaml
    from src.tools.cost_tracker import CostTrackerTool, _pricing_path
    import src.tools.cost_tracker as ct_mod

    tmp = tempfile.mkdtemp()
    orig_fn = ct_mod._pricing_path

    config_dir = os.path.join(tmp, "config")
    os.makedirs(config_dir)
    pricing_file = os.path.join(config_dir, "pricing.yaml")
    ct_mod._pricing_path = lambda: pricing_file

    try:
        pricing = {
            "openai": {
                "gpt-4": {"input_per_1m": 30.0, "output_per_1m": 60.0},
                "_default": {"input_per_1m": 1.0, "output_per_1m": 2.0},
            },
        }
        with open(pricing_file, "w") as f:
            yaml.dump(pricing, f)

        # All providers
        r = json.loads(CostTrackerTool.execute({"action": "get_pricing"}))
        check("all providers has 'providers'", "providers" in r)
        check("openai in providers", "openai" in r["providers"])

        # Single provider
        r2 = json.loads(CostTrackerTool.execute({
            "action": "get_pricing", "provider": "openai",
        }))
        check("single provider has 'models'", "models" in r2)

        # Provider + model (exact)
        r3 = json.loads(CostTrackerTool.execute({
            "action": "get_pricing", "provider": "openai", "model": "gpt-4",
        }))
        check("exact model pricing", r3["pricing"]["input_per_1m"] == 30.0)

        # Provider + model (unknown → _default)
        r4 = json.loads(CostTrackerTool.execute({
            "action": "get_pricing", "provider": "openai", "model": "xxx",
        }))
        check("default fallback", r4["pricing"].get("input_per_1m") == 1.0)

    finally:
        ct_mod._pricing_path = orig_fn
        shutil.rmtree(tmp, ignore_errors=True)


def test_cost_tracker_set_pricing():
    print("\n=== CostTrackerTool — set_pricing ===")
    import yaml
    import src.tools.cost_tracker as ct_mod

    tmp = tempfile.mkdtemp()
    config_dir = os.path.join(tmp, "config")
    os.makedirs(config_dir)
    pricing_file = os.path.join(config_dir, "pricing.yaml")
    orig_fn = ct_mod._pricing_path
    ct_mod._pricing_path = lambda: pricing_file

    try:
        # Set on empty file
        r = json.loads(CostTrackerTool.execute({
            "action": "set_pricing",
            "provider": "deepseek",
            "model": "deepseek-v2",
            "input_per_1m": 0.14,
            "output_per_1m": 0.28,
        }))
        check("set ok", r.get("ok") is True)
        check("set model", r["model"] == "deepseek-v2")

        # Verify persisted
        with open(pricing_file, "r") as f:
            saved = yaml.safe_load(f)
        check("persisted to file", saved["deepseek"]["deepseek-v2"]["input_per_1m"] == 0.14)

        # Missing provider/model
        r2 = json.loads(CostTrackerTool.execute({
            "action": "set_pricing",
        }))
        check("missing fields → error", "error" in r2)

    finally:
        ct_mod._pricing_path = orig_fn
        shutil.rmtree(tmp, ignore_errors=True)


def test_cost_tracker_unknown_action():
    print("\n=== CostTrackerTool — unknown action ===")
    from src.tools.cost_tracker import CostTrackerTool
    r = json.loads(CostTrackerTool.execute({"action": "explode"}))
    check("unknown action → error", "error" in r)


def test_cost_tracker_list_models():
    print("\n=== CostTrackerTool — list_models ===")
    import src.tools.cost_tracker as ct_mod

    tmp = tempfile.mkdtemp()
    config_dir = os.path.join(tmp, "config")
    os.makedirs(config_dir)
    conn_file = os.path.join(config_dir, "connections.json")
    orig_fn = ct_mod._connections_path
    ct_mod._connections_path = lambda: conn_file

    try:
        conn_data = {
            "connections": [
                {"name": "My OpenAI", "provider": "openai", "enabled": True,
                 "models": ["gpt-4", "gpt-3.5-turbo"]},
                {"name": "Disabled", "provider": "anthropic", "enabled": False,
                 "models": ["claude-sonnet-4-20250514"]},
            ],
        }
        with open(conn_file, "w") as f:
            json.dump(conn_data, f)

        r = json.loads(CostTrackerTool.execute({"action": "list_models"}))
        check("has connections key", "connections" in r)
        check("enabled connection listed", "My OpenAI" in r["connections"])
        check("disabled connection excluded", "Disabled" not in r["connections"])

    finally:
        ct_mod._connections_path = orig_fn
        shutil.rmtree(tmp, ignore_errors=True)


# ═════════════════════════════════════════════
# WEB SEARCH TOOL TESTS (offline/helpers only)
# ═════════════════════════════════════════════

def test_web_search_definition():
    print("\n=== WebSearchTool — definition ===")
    from src.tools.web_search import WebSearchTool
    defn = WebSearchTool.definition()
    check("name is web_search", defn["name"] == "web_search")
    check("has action enum", "action" in defn["parameters"]["properties"])
    actions = defn["parameters"]["properties"]["action"]["enum"]
    check("search in actions", "search" in actions)
    check("scrape in actions", "scrape" in actions)


def test_web_search_clean_text():
    print("\n=== WebSearchTool — _clean_text ===")
    from src.tools.web_search import _clean_text

    # Plain text passes through
    check("plain text", _clean_text("hello world") == "hello world")

    # HTML stripped
    result = _clean_text("<p>Hello</p> <b>World</b>")
    check("HTML stripped", "Hello" in result and "World" in result)
    check("no tags in result", "<" not in result)

    # Whitespace collapsed
    result2 = _clean_text("hello   \n\t  world")
    check("whitespace collapsed", result2 == "hello world")

    # Empty
    check("empty string → empty", _clean_text("") == "")


def test_web_search_truncate():
    print("\n=== WebSearchTool — _truncate ===")
    from src.tools.web_search import _truncate

    text = "one two three four five six seven eight nine ten"
    check("under limit passes through", _truncate(text, 100) == text)
    truncated = _truncate(text, 3)
    check("truncated to 3 words", truncated == "one two three...")
    check("single word", _truncate("hello", 1) == "hello")
    check("empty", _truncate("", 5) == "")


def test_web_search_remove_emojis():
    print("\n=== WebSearchTool — _remove_emojis ===")
    from src.tools.web_search import _remove_emojis
    check("text preserved", _remove_emojis("hello world") == "hello world")
    result = _remove_emojis("hello 🌍 world")
    check("emoji removed", "🌍" not in result)
    check("text intact", "hello" in result and "world" in result)


def test_web_search_knowledge_gate():
    print("\n=== WebSearchTool — knowledge gate ===")
    from src.tools.web_search import WebSearchTool

    tool = WebSearchTool()

    # Missing reason → blocked
    r1 = json.loads(tool.execute({"action": "search", "query": "test"}))
    check("no reason → gate", "gate" in r1 or "missing_justification" in r1.get("gate", ""),
          f"got {r1}")

    # Already knows → blocked
    r2 = json.loads(tool.execute({
        "action": "search", "query": "test",
        "knowledge_check": "I already know this from my training data",
        "reason": "just checking",
    }))
    check("already knows → blocked", r2.get("gate") == "blocked", f"got {r2}")

    # Various skip signals
    for signal in ["i already know", "general knowledge", "common knowledge",
                   "from my training", "no search needed"]:
        r = json.loads(tool.execute({
            "action": "search", "query": "test",
            "knowledge_check": f"This is {signal}.",
            "reason": "testing",
        }))
        check(f"signal '{signal[:20]}' blocked", r.get("gate") == "blocked")


def test_web_search_mode_presets():
    print("\n=== WebSearchTool — mode presets ===")
    from src.tools.web_search import _get_mode_preset

    fast = _get_mode_preset("fast")
    check("fast returns 3-tuple", len(fast) == 3)
    check("fast pages", fast[0] >= 1)

    normal = _get_mode_preset("normal")
    check("normal pages > fast", normal[0] >= fast[0])

    deep = _get_mode_preset("deep")
    check("deep pages > normal", deep[0] >= normal[0])
    check("deep word limit > normal", deep[2] >= normal[2])


def test_web_search_effective_config():
    print("\n=== WebSearchTool — get_effective_config ===")
    from src.tools.web_search import get_effective_config

    cfg = get_effective_config()
    check("has searxng_url", "searxng_url" in cfg)
    check("has ignored_sites", "ignored_sites" in cfg)
    check("has modes", "modes" in cfg)
    check("modes has fast", "fast" in cfg["modes"])
    check("modes has normal", "normal" in cfg["modes"])
    check("modes has deep", "deep" in cfg["modes"])
    for mode in cfg["modes"].values():
        check(f"mode has pages", "pages" in mode)
        check(f"mode has return_count", "return_count" in mode)
        check(f"mode has word_limit", "word_limit" in mode)


def test_web_search_no_query():
    print("\n=== WebSearchTool — edge: search with empty query ===")
    from src.tools.web_search import WebSearchTool
    tool = WebSearchTool()

    # Scrape with no URL
    r = json.loads(tool.execute({"action": "scrape", "url": ""}))
    check("scrape no url → error", "error" in r)

    r2 = json.loads(tool.execute({"action": "scrape"}))
    check("scrape missing url → error", "error" in r2)


# ═════════════════════════════════════════════
if __name__ == "__main__":
    from src.tools.cost_tracker import CostTrackerTool

    test_registry_list()
    test_registry_resolve()
    test_registry_execute()
    test_registry_agent_defs()

    test_cost_tracker_definition()
    test_cost_tracker_get_pricing()
    test_cost_tracker_set_pricing()
    test_cost_tracker_unknown_action()
    test_cost_tracker_list_models()

    test_web_search_definition()
    test_web_search_clean_text()
    test_web_search_truncate()
    test_web_search_remove_emojis()
    test_web_search_knowledge_gate()
    test_web_search_mode_presets()
    test_web_search_effective_config()
    test_web_search_no_query()

    print(f"\n{'='*40}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL:
        sys.exit(1)
    else:
        print("All tests passed.")
