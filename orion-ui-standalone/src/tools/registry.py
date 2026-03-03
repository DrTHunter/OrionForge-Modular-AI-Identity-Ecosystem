"""Tool registry — resolves OpenAI-format tool definitions for an agent.

Each agent profile YAML declares an ``allowed_tools`` list.  The registry
only loads definitions for those tools and **enforces authorization at
execution time** — if an agent tries to call a tool not in its profile,
the call is blocked and a clear error is returned to the LLM.

Usage:
    from src.tools.registry import get_tool_defs_for_agent, execute_tool

    tool_defs = get_tool_defs_for_agent("astraea")
    # tool_defs is a list of {"type": "function", "function": {...}} dicts

    result = execute_tool("echo", {"text": "hi"}, agent_name="astraea")
    # Raises PermissionError if "echo" is not in astraea's allowed_tools
"""

import importlib
import inspect
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PROFILES_DIR = _PROJECT_ROOT / "profiles"

# ── Tool catalogue ────────────────────────────────────────────────
# Map tool name ➜ (module_path, class_name, needs_instance)
# needs_instance = True means the class has __init__ / stateful execute(self, ...)
_TOOL_MAP: Dict[str, Tuple[str, str, bool]] = {
    "echo":                ("src.tools.echo",                "EchoTool",              False),
    "memory":              ("src.tools.memory_tool",         "MemoryTool",            True),
    "directives":          ("src.tools.directives_tool",     "DirectivesTool",        False),
    "cost_tracker":        ("src.tools.cost_tracker",        "CostTrackerTool",       False),
    "continuation_update": ("src.tools.continuation_update", "ContinuationUpdateTool", False),
    "web_search":          ("src.tools.web_search",          "WebSearchTool",         True),
    "email":               ("src.tools.email_tool",          "EmailTool",             True),
    "inbox":               ("src.tools.inbox",               "InboxTool",             False),
    "runtime_info":        ("src.tools.runtime_info",        "RuntimeInfoTool",       False),
    "agi_loop":            ("src.tools.agi_loop",            "AGILoopTool",           False),
    "model_router":        ("src.tools.model_router",        "ModelRouterTool",       False),
}

# Singleton cache for stateful tool instances
_instances: Dict[str, Any] = {}


# ── Profile helpers ───────────────────────────────────────────────

def _load_profile(agent: str) -> dict:
    path = _PROFILES_DIR / f"{agent}.yaml"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get_allowed_set(agent: str) -> Set[str]:
    """Return the set of tool names this agent is authorised to use."""
    profile = _load_profile(agent)
    return set(profile.get("allowed_tools", []))


def _resolve_tool(name: str) -> Dict[str, Any] | None:
    """Import a tool class and return its OpenAI function-calling definition.

    Returns ``{"type": "function", "function": <definition>}`` or None.
    """
    if name not in _TOOL_MAP:
        log.debug("[registry] Unknown tool %r — skipped", name)
        return None

    mod_path, cls_name, needs_instance = _TOOL_MAP[name]
    try:
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name)
        defn = cls.definition()

        # Warm singleton cache for stateful tools
        if needs_instance and name not in _instances:
            _instances[name] = cls()

        return {"type": "function", "function": defn}
    except Exception as exc:
        log.warning("[registry] Failed to load tool %r: %s", name, exc)
        return None


# ── Public API ────────────────────────────────────────────────────

def get_allowed_tools(agent: str) -> List[str]:
    """Return the list of tool names enabled for *agent* (from profile YAML).

    This is the authoritative source for what an agent can use.
    """
    return sorted(_get_allowed_set(agent))


def is_tool_allowed(agent: str, tool_name: str) -> bool:
    """Check whether *agent* is authorised to use *tool_name*."""
    return tool_name in _get_allowed_set(agent)


def get_tool_defs_for_agent(agent: str) -> List[Dict[str, Any]]:
    """Return OpenAI-format tool definitions for *agent*'s ``allowed_tools``.

    Only tools listed in the agent's profile YAML are resolved.
    Returns an empty list if the profile has no allowed_tools or none
    resolved successfully.
    """
    profile = _load_profile(agent)
    allowed = profile.get("allowed_tools", [])
    if not allowed:
        log.info("[registry] %s — no allowed_tools in profile, 0 tools loaded", agent)
        return []

    defs: List[Dict[str, Any]] = []
    for name in allowed:
        td = _resolve_tool(name)
        if td:
            defs.append(td)

    log.info("[registry] %s — %d/%d tools resolved: %s",
             agent, len(defs), len(allowed),
             [d["function"]["name"] for d in defs])
    return defs


def execute_tool(name: str, arguments: Dict[str, Any],
                 agent_name: str = "") -> str:
    """Run a tool by name with the given arguments.

    **Authorization**: When *agent_name* is provided the tool must appear
    in that agent's ``allowed_tools``.  If not, a ``PermissionError`` is
    raised and the error message is returned to the LLM so it knows the
    tool is unavailable.

    Raises:
        PermissionError – agent is not authorised to use this tool
        KeyError        – tool does not exist in the registry
        RuntimeError    – tool execution failed
    """
    # 1. Check the tool exists at all
    if name not in _TOOL_MAP:
        raise KeyError(f"Unknown tool: {name}")

    # 2. Per-agent authorization gate
    if agent_name:
        allowed = _get_allowed_set(agent_name)
        if allowed and name not in allowed:
            log.warning("[registry] BLOCKED — %s attempted tool %r which is not in its allowed_tools %s",
                        agent_name, name, sorted(allowed))
            raise PermissionError(
                f"Tool '{name}' is not enabled for agent '{agent_name}'. "
                f"Authorised tools: {sorted(allowed)}"
            )

    # 3. Import & execute
    mod_path, cls_name, needs_instance = _TOOL_MAP[name]
    mod = importlib.import_module(mod_path)
    cls = getattr(mod, cls_name)

    if needs_instance:
        if name not in _instances:
            _instances[name] = cls()
        sig = inspect.signature(_instances[name].execute)
        if "agent_name" in sig.parameters:
            result = _instances[name].execute(arguments, agent_name=agent_name)
        else:
            result = _instances[name].execute(arguments)
    else:
        result = cls.execute(arguments)

    return str(result)


def list_registered_tools() -> List[str]:
    """Return all tool names the registry knows about (regardless of agent)."""
    return sorted(_TOOL_MAP.keys())
