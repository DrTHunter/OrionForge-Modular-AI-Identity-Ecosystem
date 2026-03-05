"""model_router — Task-aware model tier selection (tool wrapper).

Thin wrapper around ``src.routing.model_router`` that provides:
  1. Backward-compatible module-level functions for existing imports
  2. A callable ``ModelRouterTool`` for agent queries
  3. A ``BudgetSummaryTool`` for agents to check spending

All core logic lives in ``src/routing/model_router.py``.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ── Re-exports from the routing subsystem ─────────────────────────
# Existing code imports these names from ``src.tools.model_router``;
# re-exporting keeps everything backward-compatible.

from src.routing.model_router import (   # noqa: F401
    classify_task,
    load_router_config,
    save_router_config,
    resolve_model_for_task,
    resolve_tier,
    ModelRouter,
    RoutingDecision,
    Tier,
    TierConfig,
    TaskType,
    CONFIG_DEFAULTS,
    MODEL_ROUTER_FILE,
    _TASK_KEYWORDS,
    DEFAULT_TASK_TIER_MAP,
)
from src.routing.budget_tracker import BudgetTracker  # noqa: F401

# Backward-compatible aliases used by older tests
_DEFAULTS = {"enabled": True, "task_tier_map": dict(DEFAULT_TASK_TIER_MAP)}
_CLASSIFICATION_RULES = [(kws, tt) for tt, kws in _TASK_KEYWORDS.items()]


# ── Legacy helpers (kept for callers that use the dict-level API) ─

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


# ── Agent tool: ModelRouter ───────────────────────────────────────

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
                "'classify' — classify text into a task type without resolving a model; "
                "'budget' — show current budget/spending status."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["resolve", "list_tiers", "get_map", "classify", "budget"],
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
            router = ModelRouter.from_config(cfg)
            decision = router.route(text)
            return json.dumps({
                "task_type": decision.task_type,
                "routed_model": decision.model,
                "routed_provider": decision.provider,
                "tier": decision.tier_name or None,
                "tier_id": decision.tier_id or None,
                "connection_id": decision.connection_id or None,
                "reason": decision.reason,
                "fallback": decision.fallback,
                "is_direct_model": decision.is_direct_model,
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
            tier_dict = resolve_tier(task_type, cfg)
            return json.dumps({
                "task_type": task_type,
                "would_route_to": tier_dict.get("label") if tier_dict else "__auto__",
                "model": tier_dict.get("primary_model") if tier_dict else None,
            }, indent=2)

        elif action == "budget":
            try:
                tracker = BudgetTracker()
                return json.dumps(tracker.get_summary(), indent=2)
            except Exception as exc:
                return json.dumps({"error": f"Budget tracker unavailable: {exc}"})

        return json.dumps({"error": f"Unknown action: {action}"})
