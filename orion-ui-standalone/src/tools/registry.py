"""Tool registry — resolves OpenAI-format tool definitions for an agent.

Reads the agent's ``allowed_tools`` from its profile YAML, imports each
tool class, and returns the function-calling definitions the LLM needs.

Usage:
    from src.tools.registry import get_tool_defs_for_agent
    tool_defs = get_tool_defs_for_agent("astraea")
    # tool_defs is a list of {"type": "function", "function": {...}} dicts
"""

import importlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PROFILES_DIR = _PROJECT_ROOT / "profiles"

# Map tool name ➜ (module_path, class_name, needs_instance)
# needs_instance = True means the class has __init__ / stateful execute(self, ...)
_TOOL_MAP: Dict[str, Tuple[str, str, bool]] = {
    "echo":                ("src.tools.echo",                "EchoTool",              False),
    "memory":              ("src.tools.memory_tool",         "MemoryTool",            True),
    "directives":          ("src.tools.directives_tool",     "DirectivesTool",        False),
    "cost_tracker":        ("src.tools.cost_tracker",        "CostTrackerTool",       False),
    "continuation_update": ("src.tools.continuation_update", "ContinuationUpdateTool", False),
}

# Singleton cache for stateful tool instances
_instances: Dict[str, Any] = {}


def _load_profile(agent: str) -> dict:
    path = _PROFILES_DIR / f"{agent}.yaml"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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

def get_tool_defs_for_agent(agent: str) -> List[Dict[str, Any]]:
    """Return OpenAI-format tool definitions for *agent*'s ``allowed_tools``.

    Returns an empty list if the profile has no allowed_tools or none
    resolved successfully.
    """
    profile = _load_profile(agent)
    allowed = profile.get("allowed_tools", [])
    if not allowed:
        return []

    defs: List[Dict[str, Any]] = []
    for name in allowed:
        td = _resolve_tool(name)
        if td:
            defs.append(td)

    if defs:
        log.info("[registry] %s — %d tools resolved: %s",
                 agent, len(defs),
                 [d["function"]["name"] for d in defs])
    return defs


def list_registered_tools() -> List[str]:
    """Return all tool names the registry knows about."""
    return sorted(_TOOL_MAP.keys())
