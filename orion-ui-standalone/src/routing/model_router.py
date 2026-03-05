"""ModelRouter — tiered routing with escalation for agent LLM calls.

Tier ladder (cheapest → most expensive):
  T0  local_cheap     — Ollama small models (qwen2.5:7b, phi-3, etc.)   ~$0/call
  T1  local_strong    — Ollama large models (llama3:70b, mixtral, etc.) ~$0/call
  T2  cheap_cloud     — DeepSeek-V3/Chat, GPT-5-mini, GPT-5-nano       ~$0.001/call
  T3  expensive_cloud — GPT-5.2, Claude Opus 4.6, GPT-5.2-pro          ~$0.01–0.10/call
  T4  code_light      — Fast coding model (DeepSeek, Codestral, etc.)   ~$0.001/call
  T5  code_heavy      — Power coding model (Claude, GPT-4o, etc.)       ~$0.01–0.10/call

Escalation chains:
  General : local_cheap → local_strong → cheap_cloud → expensive_cloud
  Coding  : code_light  → code_heavy   → expensive_cloud

Routing algorithm:
  1. Classify task type from stimulus (coding_light, coding_heavy, etc.)
  2. Start at the tier mapped for that task type
  3. If stuck (same error 3+ times), escalate via chain
  4. Skip disabled tiers automatically
  5. Enforce budget gates — refuse to escalate if budget exceeded
  6. Support direct model overrides (model:<conn_id>:<name>)
"""

import json
import logging
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"
MODEL_ROUTER_FILE = _CONFIG_DIR / "model_router.json"


# ------------------------------------------------------------------
# Tier definitions
# ------------------------------------------------------------------

class Tier(IntEnum):
    LOCAL_CHEAP = 0
    LOCAL_STRONG = 1
    CHEAP_CLOUD = 2
    EXPENSIVE_CLOUD = 3
    CODE_LIGHT = 4
    CODE_HEAVY = 5


TIER_NAMES = {
    Tier.LOCAL_CHEAP: "local_cheap",
    Tier.LOCAL_STRONG: "local_strong",
    Tier.CHEAP_CLOUD: "cheap_cloud",
    Tier.EXPENSIVE_CLOUD: "expensive_cloud",
    Tier.CODE_LIGHT: "code_light",
    Tier.CODE_HEAVY: "code_heavy",
}

TIER_IDS = {
    Tier.LOCAL_CHEAP: "t0",
    Tier.LOCAL_STRONG: "t1",
    Tier.CHEAP_CLOUD: "t2",
    Tier.EXPENSIVE_CLOUD: "t3",
    Tier.CODE_LIGHT: "t4",
    Tier.CODE_HEAVY: "t5",
}

_LABEL_TO_TIER = {v: k for k, v in TIER_NAMES.items()}
_ID_TO_TIER = {v: k for k, v in TIER_IDS.items()}

# Escalation paths: which tier comes next when stuck-looping
_ESCALATION_CHAIN = {
    Tier.LOCAL_CHEAP: Tier.LOCAL_STRONG,
    Tier.LOCAL_STRONG: Tier.CHEAP_CLOUD,
    Tier.CHEAP_CLOUD: Tier.EXPENSIVE_CLOUD,
    Tier.EXPENSIVE_CLOUD: None,          # top of general chain
    Tier.CODE_LIGHT: Tier.CODE_HEAVY,
    Tier.CODE_HEAVY: Tier.EXPENSIVE_CLOUD,
}

# Tiers that cost money (triggers budget gates)
_PAID_TIERS = frozenset({Tier.CHEAP_CLOUD, Tier.EXPENSIVE_CLOUD,
                         Tier.CODE_LIGHT, Tier.CODE_HEAVY})

# Premium tiers (soft-limit caps to a cheaper alternative)
_PREMIUM_TIERS = frozenset({Tier.EXPENSIVE_CLOUD, Tier.CODE_HEAVY})


# ------------------------------------------------------------------
# Task classification
# ------------------------------------------------------------------

class TaskType:
    CODING = "coding"                # backward-compat alias
    CODING_LIGHT = "coding_light"    # simple fixes, renames, formatting
    CODING_HEAVY = "coding_heavy"    # implement, debug, refactor, architect
    SUMMARIZATION = "summarization"
    PLANNING = "planning"
    HIGH_STAKES = "high_stakes"
    FINAL_POLISH = "final_polish"
    GENERAL = "general"
    MEMORY_OPS = "memory_ops"
    REFLECTION = "reflection"
    TOOL_USE = "tool_use"
    AGI_TICK = "agi_tick"

    _ALL = {
        "coding", "coding_light", "coding_heavy",
        "summarization", "planning", "high_stakes",
        "final_polish", "general", "memory_ops", "reflection",
        "tool_use", "agi_tick",
    }


# Default task → tier mapping (used when no config file exists)
DEFAULT_TASK_TIER_MAP = {
    TaskType.CODING_LIGHT: "code_light",
    TaskType.CODING_HEAVY: "code_heavy",
    TaskType.SUMMARIZATION: "local_cheap",
    TaskType.PLANNING: "local_strong",
    TaskType.HIGH_STAKES: "cheap_cloud",
    TaskType.FINAL_POLISH: "expensive_cloud",
    TaskType.MEMORY_OPS: "local_cheap",
    TaskType.REFLECTION: "local_cheap",
    TaskType.TOOL_USE: "cheap_cloud",
    TaskType.AGI_TICK: "cheap_cloud",
    TaskType.GENERAL: "__auto__",
}


# Priority-ordered keyword sets for task classification (scored, not first-match).
# Heavy coding is listed first so it wins ties (dict insertion order).
_TASK_KEYWORDS: Dict[str, List[str]] = {
    TaskType.CODING_HEAVY: [
        "write code", "implement", "function", "class ", "debug",
        "fix bug", "refactor", "script", "program", "syntax",
        "compile", "api endpoint", "unit test", "regex",
        "code", "error", "method",
        "algorithm", "data structure", "migration",
        "build", "integration", "complex",
        "design pattern", "framework",
        "database", "schema", "rewrite", "api",
    ],
    TaskType.CODING_LIGHT: [
        "typo", "rename", "add comment", "reformat", "lint",
        "type hint", "import", "simple", "tweak",
        "quick fix", "one-liner", "small change",
        "docstring", "cleanup", "tidy", "minor",
        "snippet", "whitespace", "indent", "spelling",
        "variable", "add log", "print statement",
    ],
    TaskType.SUMMARIZATION: [
        "summarize", "summary", "tldr", "condense", "recap",
        "brief", "shorten", "digest", "overview", "compress",
    ],
    TaskType.PLANNING: [
        "plan", "strategy", "roadmap", "design", "outline",
        "milestone", "breakdown", "proposal", "architect",
        "organize", "prioritize", "suggest",
    ],
    TaskType.HIGH_STAKES: [
        "review", "critical", "security", "audit", "verify",
        "dangerous", "production", "deploy", "migration",
        "breaking change", "data loss", "irreversible",
    ],
    TaskType.FINAL_POLISH: [
        "polish", "final", "publish", "release", "proofread",
        "camera-ready", "quality check", "perfection",
        "final review", "ship it", "last pass",
    ],
    TaskType.MEMORY_OPS: [
        "remember", "memory", "recall", "forget", "vault",
        "memorize", "store memory",
    ],
    TaskType.REFLECTION: [
        "reflect", "think about", "consider", "introspect",
        "journal", "self-assess", "ponder", "narrative",
    ],
    TaskType.TOOL_USE: [
        "search", "look up", "find info", "browse", "web search",
        "scrape",
    ],
}


def classify_task(
    text: str,
    recent_errors: Optional[List[str]] = None,
    explicit_type: Optional[str] = None,
) -> str:
    """Classify task type from text content and context.

    Uses scored keyword matching (not first-match) for better accuracy.
    Boosts coding score when recent errors are present.

    Parameters
    ----------
    text : str
        The user message or stimulus text.
    recent_errors : list of str, optional
        Recent error messages (shifts classification toward coding).
    explicit_type : str, optional
        If set, overrides auto-classification.

    Returns
    -------
    str
        One of the TaskType constants.
    """
    if explicit_type and explicit_type in TaskType._ALL:
        return explicit_type

    lower = text.lower()
    scores: Dict[str, int] = {}

    for task_type, keywords in _TASK_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in lower)
        if score > 0:
            scores[task_type] = score

    # Boost heavy coding if there are recent errors (errors = complex problem)
    if recent_errors and len(recent_errors) > 0:
        scores[TaskType.CODING_HEAVY] = scores.get(TaskType.CODING_HEAVY, 0) + len(recent_errors)

    if not scores:
        return TaskType.GENERAL

    return max(scores, key=scores.get)  # type: ignore[arg-type]


# ------------------------------------------------------------------
# Tier configuration (per-tier limits)
# ------------------------------------------------------------------

@dataclass
class TierConfig:
    """Configuration for a single tier."""
    tier: Tier
    tier_id: str                        # "t0", "t1", etc.
    label: str                          # "local_cheap", etc.
    provider: str                       # "ollama", "openai", "deepseek"
    model: str                          # primary model name
    connection_id: str = ""             # OrionForge connection ID
    base_url: str = ""
    temperature: float = 0.7
    max_output_tokens: int = 4096
    max_iterations: int = 10
    max_retries_before_escalate: int = 3
    alt_models: list = field(default_factory=list)
    cost_per_call: str = ""
    default_for: str = ""
    enabled: bool = True

    def to_json_dict(self) -> dict:
        """Convert back to the JSON format used by model_router.json and the UI."""
        return {
            "id": self.tier_id,
            "label": self.label,
            "enabled": self.enabled,
            "connection_id": self.connection_id,
            "provider": self.provider,
            "primary_model": self.model,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "max_iterations": self.max_iterations,
            "retries_before_escalate": self.max_retries_before_escalate,
            "alt_models": self.alt_models,
            "cost_per_call": self.cost_per_call,
            "default_for": self.default_for,
        }


@dataclass
class RoutingDecision:
    """Result of a routing decision."""
    tier: Optional[Tier]
    tier_name: str
    tier_id: str
    task_type: str
    model: Optional[str]
    provider: Optional[str]
    connection_id: str = ""
    reason: str = ""
    budget_remaining: float = -1.0
    escalated_from: Optional[Tier] = None
    is_direct_model: bool = False
    direct_conn_id: str = ""
    fallback: bool = False              # True when router defers to default

    def to_dict(self) -> dict:
        """Serialize for logging/API responses."""
        return {
            "tier": self.tier_name,
            "tier_id": self.tier_id,
            "task_type": self.task_type,
            "model": self.model,
            "provider": self.provider,
            "connection_id": self.connection_id,
            "reason": self.reason,
            "budget_remaining": self.budget_remaining,
            "escalated_from": TIER_NAMES.get(self.escalated_from) if self.escalated_from is not None else None,
            "is_direct_model": self.is_direct_model,
            "direct_conn_id": self.direct_conn_id,
            "fallback": self.fallback,
        }


# ------------------------------------------------------------------
# Config I/O (JSON ↔ typed structures)
# ------------------------------------------------------------------

_DEFAULT_TIERS = [
    {
        "id": "t0", "label": "local_cheap", "enabled": True,
        "connection_id": "", "provider": "ollama",
        "primary_model": "qwen2.5:7b",
        "temperature": 0.6, "max_output_tokens": 2048,
        "max_iterations": 8, "retries_before_escalate": 3,
        "alt_models": [], "cost_per_call": "~$0.00", "default_for": "",
    },
    {
        "id": "t1", "label": "local_strong", "enabled": True,
        "connection_id": "", "provider": "ollama",
        "primary_model": "llama3:70b",
        "temperature": 0.5, "max_output_tokens": 4096,
        "max_iterations": 10, "retries_before_escalate": 3,
        "alt_models": [], "cost_per_call": "~$0.00", "default_for": "",
    },
    {
        "id": "t2", "label": "cheap_cloud", "enabled": True,
        "connection_id": "", "provider": "deepseek",
        "primary_model": "deepseek-chat",
        "temperature": 0.4, "max_output_tokens": 8192,
        "max_iterations": 12, "retries_before_escalate": 2,
        "alt_models": [], "cost_per_call": "~$0.001", "default_for": "",
    },
    {
        "id": "t3", "label": "expensive_cloud", "enabled": True,
        "connection_id": "", "provider": "openai",
        "primary_model": "gpt-4o",
        "temperature": 0.3, "max_output_tokens": 16384,
        "max_iterations": 15, "retries_before_escalate": 2,
        "alt_models": [], "cost_per_call": "~$0.01\u20130.10", "default_for": "",
    },    {
        "id": "t4", "label": "code_light", "enabled": True,
        "connection_id": "", "provider": "deepseek",
        "primary_model": "deepseek-chat",
        "temperature": 0.3, "max_output_tokens": 4096,
        "max_iterations": 8, "retries_before_escalate": 3,
        "alt_models": [], "cost_per_call": "~$0.001",
        "default_for": "Light coding — typos, renames, formatting",
    },
    {
        "id": "t5", "label": "code_heavy", "enabled": True,
        "connection_id": "", "provider": "openai",
        "primary_model": "gpt-4o",
        "temperature": 0.2, "max_output_tokens": 16384,
        "max_iterations": 15, "retries_before_escalate": 2,
        "alt_models": [], "cost_per_call": "~$0.01–0.10",
        "default_for": "Heavy coding — implement, debug, refactor",
    },]

CONFIG_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "tiers": list(_DEFAULT_TIERS),
    "task_tier_map": dict(DEFAULT_TASK_TIER_MAP),
}


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default or {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default or {}


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_router_config() -> dict:
    """Load the merged router configuration as a raw dict (for UI/API)."""
    saved = _read_json(MODEL_ROUTER_FILE, {})
    merged = {**CONFIG_DEFAULTS, **saved}
    if not merged.get("tiers"):
        merged["tiers"] = list(_DEFAULT_TIERS)
    if not merged.get("task_tier_map"):
        merged["task_tier_map"] = dict(DEFAULT_TASK_TIER_MAP)
    if "enabled" not in merged:
        merged["enabled"] = True
    return merged


def save_router_config(data: dict) -> None:
    """Save router configuration to disk."""
    _write_json(MODEL_ROUTER_FILE, data)


def _parse_tier_config(t: dict) -> TierConfig:
    """Parse a single tier JSON dict into a TierConfig."""
    tier_id = t.get("id", "t0")
    label = t.get("label", "")
    tier_enum = _LABEL_TO_TIER.get(label) or _ID_TO_TIER.get(tier_id, Tier.LOCAL_CHEAP)
    return TierConfig(
        tier=tier_enum,
        tier_id=tier_id,
        label=label,
        provider=t.get("provider", "ollama"),
        model=t.get("primary_model", ""),
        connection_id=t.get("connection_id", ""),
        temperature=t.get("temperature", 0.7),
        max_output_tokens=t.get("max_output_tokens", 4096),
        max_iterations=t.get("max_iterations", 10),
        max_retries_before_escalate=t.get("retries_before_escalate", 3),
        alt_models=t.get("alt_models", []),
        cost_per_call=t.get("cost_per_call", ""),
        default_for=t.get("default_for", ""),
        enabled=t.get("enabled", True),
    )


# ------------------------------------------------------------------
# Direct model override helpers
# ------------------------------------------------------------------

def _is_direct_model(value: str) -> bool:
    """Check if a task_tier_map value is ``model:<conn_id>:<model>``."""
    return isinstance(value, str) and value.startswith("model:")


def _parse_direct_model(value: str) -> Tuple[str, str]:
    """Parse ``model:<connection_id>:<model_name>`` → (connection_id, model_name)."""
    parts = value.split(":", 2)
    if len(parts) == 3:
        return parts[1], parts[2]
    return "", ""


# ------------------------------------------------------------------
# ModelRouter
# ------------------------------------------------------------------

class ModelRouter:
    """Routes LLM calls through a tiered model ladder.

    - Task classification → tier mapping
    - Stuck-loop detection and automatic escalation
    - Budget-gate enforcement (refuses expensive tiers when budget blown)
    - Disabled-tier skipping
    - Direct model overrides (model:<conn_id>:<model>)

    Parameters
    ----------
    tier_configs : dict[Tier, TierConfig]
        Configuration for each tier.
    task_tier_map : dict[str, str]
        Maps task type → tier label (or ``__auto__`` or ``model:...``).
    budget_tracker : optional BudgetTracker
        If provided, routing respects budget limits.
    enabled : bool
        Master switch for the router.
    """

    def __init__(
        self,
        tier_configs: Dict[Tier, TierConfig],
        task_tier_map: Dict[str, str],
        budget_tracker: "Any" = None,
        enabled: bool = True,
    ):
        self.tier_configs = tier_configs
        self.task_tier_map = task_tier_map
        self.budget_tracker = budget_tracker
        self.enabled = enabled
        self._stuck_counter: Dict[str, int] = {}   # error_hash → count
        self._recent_errors: List[str] = []

    # ── Core routing ──────────────────────────────────────────────

    def route(
        self,
        stimulus: str,
        recent_errors: Optional[List[str]] = None,
        explicit_task_type: Optional[str] = None,
        force_tier: Optional[Tier] = None,
        fallback_task: str = "general",
    ) -> RoutingDecision:
        """Decide which tier/model to use for a given stimulus.

        Parameters
        ----------
        stimulus : str
            The user message or AGI tick text to classify.
        recent_errors : list of str, optional
            Recent error strings (triggers stuck-loop escalation).
        explicit_task_type : str, optional
            Override auto-classification.
        force_tier : Tier, optional
            Force a specific tier (bypasses all logic).
        fallback_task : str
            Task type to use when nothing matches (default: "general").

        Returns
        -------
        RoutingDecision
        """
        if not self.enabled:
            return RoutingDecision(
                tier=None, tier_name="", tier_id="",
                task_type=classify_task(stimulus, recent_errors, explicit_task_type),
                model=None, provider=None,
                reason="Router disabled", fallback=True,
            )

        if recent_errors:
            self._recent_errors = recent_errors[-10:]

        # 1. Classify the task
        task_type = classify_task(stimulus, recent_errors, explicit_task_type)

        # 2. Look up the tier mapping for this task type
        mapping_value = self.task_tier_map.get(task_type)
        # Backward compat: coding_light/coding_heavy fall back to "coding" entry
        if mapping_value is None and task_type in (TaskType.CODING_LIGHT, TaskType.CODING_HEAVY):
            mapping_value = self.task_tier_map.get(TaskType.CODING)
        if mapping_value is None:
            mapping_value = "__auto__"

        # 3. Force tier bypasses all mapping / __auto__ logic
        if force_tier is not None:
            target_tier = force_tier
            reason = f"Forced to tier {TIER_NAMES[force_tier]}"
        else:
            # 4. Handle direct model overrides (model:<conn_id>:<model>)
            if _is_direct_model(mapping_value):
                conn_id, model_name = _parse_direct_model(mapping_value)
                log.info("[router] Task '%s' → direct model '%s' (conn=%s)",
                         task_type, model_name, conn_id)
                return RoutingDecision(
                    tier=None, tier_name="direct", tier_id="",
                    task_type=task_type, model=model_name, provider=None,
                    connection_id="", reason=f"Direct model override for '{task_type}'",
                    is_direct_model=True, direct_conn_id=conn_id,
                )

            # 5. Handle __auto__ (defer to caller's default model)
            if mapping_value == "__auto__":
                return RoutingDecision(
                    tier=None, tier_name="__auto__", tier_id="",
                    task_type=task_type, model=None, provider=None,
                    reason=f"Task '{task_type}' mapped to __auto__", fallback=True,
                )

            # 6. Resolve the target tier from the label
            target_tier = _LABEL_TO_TIER.get(mapping_value)
            if target_tier is None:
                return RoutingDecision(
                    tier=None, tier_name=mapping_value, tier_id="",
                    task_type=task_type, model=None, provider=None,
                    reason=f"Unknown tier label '{mapping_value}'", fallback=True,
                )

            reason = f"Task '{task_type}' → tier '{TIER_NAMES[target_tier]}'"

        # 7. Stuck-loop escalation (uses _ESCALATION_CHAIN)
        original_target = target_tier
        if recent_errors:
            error_key = recent_errors[-1][:100]
            self._stuck_counter[error_key] = self._stuck_counter.get(error_key, 0) + 1
            count = self._stuck_counter[error_key]
            cfg = self.tier_configs.get(target_tier)
            threshold = cfg.max_retries_before_escalate if cfg else 3
            next_tier = _ESCALATION_CHAIN.get(target_tier)
            if count >= threshold and next_tier is not None:
                old_tier = target_tier
                target_tier = next_tier
                reason = (
                    f"Stuck loop ({count} retries), "
                    f"escalating {TIER_NAMES[old_tier]} → {TIER_NAMES[target_tier]}"
                )
                log.warning("[router] %s", reason)
        else:
            self._stuck_counter.clear()

        # 8. Skip disabled tiers — follow escalation chain
        checked = set()
        while True:
            cfg = self.tier_configs.get(target_tier)
            if cfg and cfg.enabled:
                break
            checked.add(target_tier)
            next_tier = _ESCALATION_CHAIN.get(target_tier)
            if next_tier is None or next_tier in checked:
                break
            target_tier = next_tier

        # 9. Budget gate
        budget_remaining = -1.0
        if self.budget_tracker:
            try:
                budget_remaining = self.budget_tracker.remaining()
                if budget_remaining <= 0 and target_tier in _PAID_TIERS:
                    # Force down to local
                    target_tier = Tier.LOCAL_STRONG
                    cfg_ls = self.tier_configs.get(Tier.LOCAL_STRONG)
                    if not (cfg_ls and cfg_ls.enabled):
                        target_tier = Tier.LOCAL_CHEAP
                    reason = "Budget exhausted — forced to local tier"
                    log.warning("[router] %s", reason)
                elif self.budget_tracker.is_soft_limit_hit() and target_tier in _PREMIUM_TIERS:
                    if target_tier == Tier.CODE_HEAVY:
                        target_tier = Tier.CODE_LIGHT
                        reason = "Soft budget limit hit — capped at code_light"
                    else:
                        target_tier = Tier.CHEAP_CLOUD
                        reason = "Soft budget limit hit — capped at cheap cloud"
                    log.info("[router] %s", reason)
            except Exception as exc:
                log.warning("[router] Budget check failed: %s", exc)

        # 10. Resolve final tier config
        cfg = self.tier_configs.get(target_tier)
        if cfg is None or not cfg.enabled:
            # Fallback to cheapest available enabled tier
            cfg = None
            for t in Tier:
                if t in self.tier_configs and self.tier_configs[t].enabled:
                    cfg = self.tier_configs[t]
                    target_tier = t
                    reason = f"Fallback to {TIER_NAMES[t]} (requested tier unavailable)"
                    break
            if cfg is None:
                return RoutingDecision(
                    tier=None, tier_name="", tier_id="",
                    task_type=task_type, model=None, provider=None,
                    reason="No tiers configured or enabled", fallback=True,
                )

        log.info("[router] %s → model '%s' (%s)", reason, cfg.model, cfg.provider)

        return RoutingDecision(
            tier=target_tier,
            tier_name=TIER_NAMES.get(target_tier, ""),
            tier_id=TIER_IDS.get(target_tier, ""),
            task_type=task_type,
            model=cfg.model,
            provider=cfg.provider,
            connection_id=cfg.connection_id,
            reason=reason,
            budget_remaining=budget_remaining,
            escalated_from=original_target if original_target != target_tier else None,
        )

    # ── Convenience lookups ───────────────────────────────────────

    def resolve_tier(self, task_type: str) -> Optional[TierConfig]:
        """Given a task type, return the matching TierConfig (or None)."""
        if not self.enabled:
            return None
        mapping_value = self.task_tier_map.get(task_type)
        if mapping_value is None and task_type in (TaskType.CODING_LIGHT, TaskType.CODING_HEAVY):
            mapping_value = self.task_tier_map.get(TaskType.CODING)
        if mapping_value is None:
            mapping_value = "__auto__"
        if mapping_value == "__auto__" or _is_direct_model(mapping_value):
            return None
        tier_enum = _LABEL_TO_TIER.get(mapping_value)
        if tier_enum is None:
            return None
        cfg = self.tier_configs.get(tier_enum)
        if cfg and cfg.enabled:
            return cfg
        return None

    def get_next_tier(self, current_tier: Tier) -> Optional[TierConfig]:
        """Get the next escalation tier via _ESCALATION_CHAIN (or None)."""
        next_t = _ESCALATION_CHAIN.get(current_tier)
        if next_t is not None:
            cfg = self.tier_configs.get(next_t)
            if cfg and cfg.enabled:
                return cfg
        return None

    def clear_stuck_counter(self) -> None:
        """Reset the stuck-loop error tracking."""
        self._stuck_counter.clear()
        self._recent_errors.clear()

    # ── Factory methods ───────────────────────────────────────────

    @classmethod
    def from_config(cls, config: Optional[dict] = None, budget_tracker: "Any" = None) -> "ModelRouter":
        """Create a ModelRouter from the JSON config dict (model_router.json format).

        If *config* is None, loads from disk.
        """
        cfg = config or load_router_config()

        tier_configs: Dict[Tier, TierConfig] = {}
        for t_dict in cfg.get("tiers", []):
            tc = _parse_tier_config(t_dict)
            tier_configs[tc.tier] = tc

        task_tier_map = cfg.get("task_tier_map", dict(DEFAULT_TASK_TIER_MAP))

        return cls(
            tier_configs=tier_configs,
            task_tier_map=task_tier_map,
            budget_tracker=budget_tracker,
            enabled=cfg.get("enabled", True),
        )

    def to_config_dict(self) -> dict:
        """Export current state back to JSON config dict (for save/API)."""
        return {
            "enabled": self.enabled,
            "tiers": [tc.to_json_dict() for tc in sorted(
                self.tier_configs.values(), key=lambda c: c.tier,
            )],
            "task_tier_map": dict(self.task_tier_map),
        }


# ------------------------------------------------------------------
# Backward-compatible module-level functions
# ------------------------------------------------------------------
# These preserve the API that web/app.py and src/tools/model_router.py
# already import, so existing code keeps working without changes.

def resolve_model_for_task(
    stimulus: str,
    config: Optional[dict] = None,
    recent_errors: Optional[List[str]] = None,
) -> Tuple[Optional[str], Optional[str], str]:
    """Classify a stimulus and resolve the model to use.

    Returns ``(model, provider, task_type)``.

    - model/provider are None if the router doesn't override.
    - For direct-model overrides, provider is ``__direct__:<conn_id>``.

    This is a convenience wrapper that creates a throwaway ModelRouter.
    For persistent use (stuck-loop tracking, budget), use ModelRouter directly.
    """
    router = ModelRouter.from_config(config)
    decision = router.route(stimulus, recent_errors=recent_errors)

    if decision.is_direct_model:
        return decision.model, f"__direct__:{decision.direct_conn_id}", decision.task_type

    if decision.fallback:
        return None, None, decision.task_type

    return decision.model, decision.provider, decision.task_type


def resolve_tier(task_type: str, config: Optional[dict] = None) -> Optional[dict]:
    """Given a task type, return the matching tier config dict.

    Returns None if no match (for __auto__, direct model, or disabled tier).
    """
    router = ModelRouter.from_config(config)
    tc = router.resolve_tier(task_type)
    if tc:
        return tc.to_json_dict()
    return None
