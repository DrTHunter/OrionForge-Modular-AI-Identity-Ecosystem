"""Cost Tracker tool — manage pricing, query cost logs, get summaries.

Agents can use this tool to check spend, query pricing for models,
and review cost breakdowns.  The pricing registry itself is managed
by the Pricing UI page, but this tool exposes it for agent automation.
"""

import json
import os
from typing import Any, Dict

import yaml


# ── Lazy helpers ─────────────────────────────────────────

def _pricing_path() -> str:
    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "config", "pricing.yaml")
    )


def _load_pricing() -> Dict:
    path = _pricing_path()
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_pricing(data: Dict) -> None:
    path = _pricing_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _connections_path() -> str:
    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "config", "connections.json")
    )


def _load_connections() -> Dict:
    path = _connections_path()
    if not os.path.isfile(path):
        return {"connections": [], "agent_connections": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class CostTrackerTool:
    """Token cost management and tracking tool.

    Actions:
      get_pricing       — get pricing for a specific model or all models
      set_pricing       — update pricing for a model
      list_models       — list all available models from connected LLM APIs
      cost_summary      — get aggregated cost stats (today, this week, all-time)
      cost_log          — get recent cost log entries
      session_cost      — get cost for the current chat session
    """

    @staticmethod
    def definition() -> dict:
        return {
            "name": "cost_tracker",
            "description": (
                "Manage LLM token pricing and track costs. "
                "Can query pricing per model, update rates, list available models, "
                "and retrieve cost summaries or detailed logs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "get_pricing",
                            "set_pricing",
                            "list_models",
                            "cost_summary",
                            "cost_log",
                            "session_cost",
                        ],
                        "description": "The action to perform.",
                    },
                    "provider": {
                        "type": "string",
                        "description": "LLM provider name (openai, anthropic, deepseek, ollama).",
                    },
                    "model": {
                        "type": "string",
                        "description": "Model name (e.g. gpt-4o, claude-sonnet-4-20250514).",
                    },
                    "input_per_1m": {
                        "type": "number",
                        "description": "Input cost in USD per 1M tokens.",
                    },
                    "cached_input_per_1m": {
                        "type": "number",
                        "description": "Cached input cost in USD per 1M tokens.",
                    },
                    "output_per_1m": {
                        "type": "number",
                        "description": "Output cost in USD per 1M tokens.",
                    },
                    "training_per_1m": {
                        "type": "number",
                        "description": "Training cost in USD per 1M tokens.",
                    },
                    "agent": {
                        "type": "string",
                        "description": "Filter cost data by agent name.",
                    },
                    "since": {
                        "type": "string",
                        "description": "ISO timestamp — return events after this time.",
                    },
                    "chat_id": {
                        "type": "string",
                        "description": "Chat session ID for session_cost action.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of log entries to return (default 50).",
                    },
                },
                "required": ["action"],
            },
        }

    @staticmethod
    def execute(arguments: dict) -> str:
        action = arguments.get("action", "")
        try:
            if action == "get_pricing":
                return CostTrackerTool._get_pricing(arguments)
            elif action == "set_pricing":
                return CostTrackerTool._set_pricing(arguments)
            elif action == "list_models":
                return CostTrackerTool._list_models(arguments)
            elif action == "cost_summary":
                return CostTrackerTool._cost_summary(arguments)
            elif action == "cost_log":
                return CostTrackerTool._cost_log(arguments)
            elif action == "session_cost":
                return CostTrackerTool._session_cost(arguments)
            else:
                return json.dumps({"error": f"Unknown action: {action}"})
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    # ── Actions ──────────────────────────────────────────

    @staticmethod
    def _get_pricing(args: dict) -> str:
        pricing = _load_pricing()
        provider = args.get("provider")
        model = args.get("model")

        if provider and model:
            provider_prices = pricing.get(provider, {})
            # exact → prefix → _default
            mp = provider_prices.get(model)
            if not mp:
                for key, val in provider_prices.items():
                    if key.startswith("_"):
                        continue
                    if isinstance(val, dict) and model.startswith(key):
                        mp = val
                        break
            if not mp:
                mp = provider_prices.get("_default", {})
            return json.dumps({"provider": provider, "model": model, "pricing": mp})
        elif provider:
            return json.dumps({"provider": provider, "models": pricing.get(provider, {})})
        else:
            # flatten: list all providers and their models
            summary = {}
            for prov, models in pricing.items():
                summary[prov] = list(models.keys())
            return json.dumps({"providers": summary})

    @staticmethod
    def _set_pricing(args: dict) -> str:
        provider = args.get("provider")
        model = args.get("model")
        if not provider or not model:
            return json.dumps({"error": "provider and model are required"})

        pricing = _load_pricing()
        if provider not in pricing:
            pricing[provider] = {}

        entry = pricing[provider].get(model, {})
        for key in ("input_per_1m", "cached_input_per_1m", "output_per_1m", "training_per_1m"):
            if key in args:
                entry[key] = float(args[key])
        pricing[provider][model] = entry
        _save_pricing(pricing)

        # Reset the metering cache so new prices take effect
        try:
            from src.observability.metering import reset_pricing_cache
            reset_pricing_cache()
        except Exception:
            pass

        return json.dumps({"ok": True, "provider": provider, "model": model, "pricing": entry})

    @staticmethod
    def _list_models(args: dict) -> str:
        store = _load_connections()
        result = {}
        for conn in store.get("connections", []):
            if not conn.get("enabled"):
                continue
            name = conn.get("name", conn.get("provider", "unknown"))
            provider = conn.get("provider", "unknown")
            models = conn.get("models", [])
            result[name] = {"provider": provider, "models": models}
        return json.dumps({"connections": result})

    @staticmethod
    def _cost_summary(args: dict) -> str:
        from src.observability.metering import read_cost_log, aggregate_costs
        from datetime import datetime, timezone, timedelta

        now = datetime.now(timezone.utc)
        agent = args.get("agent")

        # All-time
        all_events = read_cost_log(agent=agent, limit=100000)
        all_time = aggregate_costs(all_events)

        # Today
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        today_events = read_cost_log(since=today_start, agent=agent, limit=100000)
        today = aggregate_costs(today_events)

        # Last 7 days
        week_start = (now - timedelta(days=7)).isoformat()
        week_events = read_cost_log(since=week_start, agent=agent, limit=100000)
        this_week = aggregate_costs(week_events)

        # Last 30 days
        month_start = (now - timedelta(days=30)).isoformat()
        month_events = read_cost_log(since=month_start, agent=agent, limit=100000)
        this_month = aggregate_costs(month_events)

        return json.dumps({
            "today": today,
            "this_week": this_week,
            "this_month": this_month,
            "all_time": all_time,
        })

    @staticmethod
    def _cost_log(args: dict) -> str:
        from src.observability.metering import read_cost_log
        events = read_cost_log(
            since=args.get("since"),
            agent=args.get("agent"),
            limit=args.get("limit", 50),
        )
        return json.dumps({"events": events, "count": len(events)})

    @staticmethod
    def _session_cost(args: dict) -> str:
        from src.observability.metering import read_cost_log, aggregate_costs
        chat_id = args.get("chat_id")
        if not chat_id:
            return json.dumps({"error": "chat_id is required"})
        all_events = read_cost_log(limit=100000)
        session_events = [e for e in all_events if e.get("chat_id") == chat_id]
        summary = aggregate_costs(session_events)
        return json.dumps({"chat_id": chat_id, **summary})
