"""model_router — Task-aware model tier selection.

Resolves which model/tier to use based on task type using the
model_router.json configuration.  Supports:
  - Task classification → tier mapping
  - Tier escalation on failure
  - Connection resolution per tier
  - Agent-queryable tool interface

The router is used both:
  1. Internally by the chat pipeline for automatic model selection
  2. As a callable tool for agents to query routing decisions
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"
MODEL_ROUTER_FILE = _CONFIG_DIR / "model_router.json"


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default or {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default or {}


# ── Defaults (mirrors app.py) ────────────────────────────────────
_DEFAULTS = {
    "enabled": True,
    "task_tier_map": {
        "coding": "cheap_cloud",
        "summarization": "local_cheap",
        "planning": "local_strong",
        "high_stakes": "cheap_cloud",
        "final_polish": "expensive_cloud",
        "memory_ops": "local_cheap",
        "reflection": "local_cheap",
        "tool_use": "cheap_cloud",
        "agi_tick": "cheap_cloud",
        "general": "__auto__",
    },
}


# ── Config I/O ────────────────────────────────────────────────────

def load_router_config() -> dict:
    """Load the merged router configuration."""
    saved = _read_json(MODEL_ROUTER_FILE, {})
    merged = {**saved}
    if "tiers" not in merged:
        merged["tiers"] = []
    if "task_tier_map" not in merged:
        merged["task_tier_map"] = _DEFAULTS["task_tier_map"]
    if "enabled" not in merged:
        merged["enabled"] = True
    return merged


# ── Task classification ───────────────────────────────────────────

# Priority-ordered keyword sets → task type
_CLASSIFICATION_RULES: List[Tuple[List[str], str]] = [
    (["write code", "implement", "function", "class ", "debug",
      "fix bug", "refactor", "script", "program", "syntax",
      "compile", "api endpoint", "unit test", "regex"],              "coding"),
    (["summarize", "summary", "tldr", "condense", "recap",
      "brief", "shorten"],                                           "summarization"),
    (["plan", "strategy", "roadmap", "design", "architect",
      "outline", "milestone", "breakdown"],                          "planning"),
    (["review", "critical", "security", "audit", "verify",
      "dangerous", "production", "deploy"],                          "high_stakes"),
    (["polish", "final", "publish", "release", "proofread",
      "camera-ready"],                                               "final_polish"),
    (["remember", "memory", "recall", "forget", "vault",
      "memorize"],                                                   "memory_ops"),
    (["reflect", "think about", "consider", "introspect",
      "journal", "self-assess"],                                     "reflection"),
    (["search", "look up", "find info", "browse", "web search",
      "scrape"],                                                     "tool_use"),
]


def classify_task(stimulus: str) -> str:
    """Simple keyword-based task classification.

    Returns a task type key from the task_tier_map.
    """
    text = stimulus.lower()

    for keywords, task_type in _CLASSIFICATION_RULES:
        if any(kw in text for kw in keywords):
            return task_type

    return "general"


# ── Tier resolution ───────────────────────────────────────────────

def _is_direct_model(value: str) -> bool:
    """Check if a task_tier_map value is a direct model override (model:<conn_id>:<model>)."""
    return isinstance(value, str) and value.startswith("model:")


def _parse_direct_model(value: str) -> Tuple[str, str]:
    """Parse ``model:<connection_id>:<model_name>`` → (connection_id, model_name)."""
    parts = value.split(":", 2)
    if len(parts) == 3:
        return parts[1], parts[2]
    return "", ""


def resolve_tier(task_type: str, config: Optional[dict] = None) -> Optional[dict]:
    """Given a task type, return the matching tier config dict.

    Returns None if no matching tier is found, router is disabled,
    or the task is mapped to a direct model override.
    """
    cfg = config or load_router_config()

    if not cfg.get("enabled", True):
        return None

    task_map = cfg.get("task_tier_map", {})
    tier_label = task_map.get(task_type, "__auto__")

    if tier_label == "__auto__":
        return None  # let default model resolution handle it

    # Direct model overrides bypass tier resolution
    if _is_direct_model(tier_label):
        return None

    tiers = cfg.get("tiers", [])
    for tier in tiers:
        if tier.get("label") == tier_label and tier.get("enabled", True):
            return tier

    return None


def resolve_model_for_task(
    stimulus: str,
    config: Optional[dict] = None,
) -> Tuple[Optional[str], Optional[str], str]:
    """Classify a stimulus and resolve the model to use.

    Returns ``(model, provider, task_type)``.
    model/provider are ``None`` if the router doesn't override
    (falls back to default resolution).

    If the task is mapped to a direct model (``model:<conn_id>:<name>``),
    the model name is returned and provider is set to
    ``__direct__:<connection_id>`` so callers can resolve the connection.
    """
    cfg = config or load_router_config()
    task_type = classify_task(stimulus)

    if not cfg.get("enabled", True):
        return None, None, task_type

    # Check for direct model override in task_tier_map
    task_map = cfg.get("task_tier_map", {})
    mapping_value = task_map.get(task_type, "__auto__")

    if _is_direct_model(mapping_value):
        conn_id, model_name = _parse_direct_model(mapping_value)
        log.info("[router] Task '%s' → direct model '%s' (conn=%s)",
                 task_type, model_name, conn_id)
        return model_name, f"__direct__:{conn_id}", task_type

    # Standard tier resolution
    tier = resolve_tier(task_type, cfg)

    if tier:
        model = tier.get("primary_model")
        provider = tier.get("provider")
        log.info("[router] Task '%s' → tier '%s' → model '%s' (%s)",
                 task_type, tier.get("label"), model, provider)
        return model, provider, task_type

    return None, None, task_type


def get_next_tier(
    current_tier_id: str,
    config: Optional[dict] = None,
) -> Optional[dict]:
    """Get the next escalation tier after the current one.

    Tiers are ordered t0 → t3.  Returns None if already at highest.
    """
    cfg = config or load_router_config()
    tiers = cfg.get("tiers", [])
    tier_ids = [t["id"] for t in tiers if t.get("enabled", True)]

    try:
        idx = tier_ids.index(current_tier_id)
        if idx + 1 < len(tier_ids):
            next_id = tier_ids[idx + 1]
            return next((t for t in tiers if t["id"] == next_id), None)
    except ValueError:
        pass

    return None


def get_tier_for_connection(
    tier_label: str,
    config: Optional[dict] = None,
) -> Optional[dict]:
    """Return the tier dict with the given label (for connection resolution)."""
    cfg = config or load_router_config()
    tiers = cfg.get("tiers", [])
    for t in tiers:
        if t.get("label") == tier_label and t.get("enabled", True):
            return t
    return None


# ── Tool interface ────────────────────────────────────────────────

class ModelRouterTool:
    """Tool interface for agents to query the model router."""

    @staticmethod
    def definition() -> dict:
        return {
            "name": "model_router",
            "description": (
                "Query the model routing system. Actions: "
                "'resolve' — classify a task and see which model/tier would handle it; "
                "'list_tiers' — show all configured tiers and their models; "
                "'get_map' — show the current task-to-tier mapping; "
                "'classify' — classify text into a task type without resolving a model."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["resolve", "list_tiers", "get_map", "classify"],
                        "description": "The action to perform.",
                    },
                    "text": {
                        "type": "string",
                        "description": (
                            "The stimulus text to classify "
                            "(for 'resolve' and 'classify' actions)."
                        ),
                    },
                },
                "required": ["action"],
            },
        }

    @staticmethod
    def execute(arguments: dict) -> str:
        action = arguments.get("action", "list_tiers")
        cfg = load_router_config()

        if action == "resolve":
            text = arguments.get("text", "")
            if not text:
                return json.dumps({"error": "Provide 'text' to classify and resolve"})
            model, provider, task_type = resolve_model_for_task(text, cfg)
            tier = resolve_tier(task_type, cfg)
            return json.dumps({
                "task_type": task_type,
                "routed_model": model,
                "routed_provider": provider,
                "tier": tier.get("label") if tier else None,
                "tier_id": tier.get("id") if tier else None,
                "fallback": model is None,
            }, indent=2)

        elif action == "list_tiers":
            tiers = cfg.get("tiers", [])
            summary = []
            for t in tiers:
                summary.append({
                    "id": t["id"],
                    "label": t.get("label", ""),
                    "enabled": t.get("enabled", True),
                    "provider": t.get("provider", ""),
                    "model": t.get("primary_model", ""),
                    "temperature": t.get("temperature", 0.7),
                    "max_iterations": t.get("max_iterations", 10),
                    "cost": t.get("cost_per_call", ""),
                })
            return json.dumps({"tiers": summary, "enabled": cfg.get("enabled", True)}, indent=2)

        elif action == "get_map":
            return json.dumps({
                "task_tier_map": cfg.get("task_tier_map", {}),
                "enabled": cfg.get("enabled", True),
            }, indent=2)

        elif action == "classify":
            text = arguments.get("text", "")
            if not text:
                return json.dumps({"error": "Provide 'text' to classify"})
            task_type = classify_task(text)
            tier = resolve_tier(task_type, cfg)
            return json.dumps({
                "task_type": task_type,
                "would_route_to": tier.get("label") if tier else "__auto__",
                "model": tier.get("primary_model") if tier else None,
            }, indent=2)

        return json.dumps({"error": f"Unknown action: {action}"})
