"""Routing subsystem — tiered model routing, budget tracking, context management."""

from src.routing.model_router import (  # noqa: F401 — public API
    ModelRouter,
    RoutingDecision,
    Tier,
    TierConfig,
    TaskType,
    classify_task,
    resolve_model_for_task,
    resolve_tier,
    load_router_config,
    save_router_config,
    CONFIG_DEFAULTS,
)
from src.routing.budget_tracker import BudgetTracker, BudgetState  # noqa: F401

__all__ = [
    "ModelRouter",
    "RoutingDecision",
    "Tier",
    "TierConfig",
    "TaskType",
    "classify_task",
    "resolve_model_for_task",
    "resolve_tier",
    "load_router_config",
    "save_router_config",
    "CONFIG_DEFAULTS",
    "BudgetTracker",
    "BudgetState",
]
