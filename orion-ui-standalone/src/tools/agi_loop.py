"""agi_loop — Autonomous tick-based execution loop.

The AGI loop is a background service that drives an agent autonomously.
Each tick:
  1. Builds a stimulus (user-provided or auto-generated)
  2. Sends it through the same chat pipeline used by interactive chat
  3. The model router selects the appropriate tier/model
  4. Budget and error tracking enforce safety caps
  5. Results are logged for inspection

Tool interface:
  Agents can query loop status or request pause/resume via the
  ``agi_loop`` tool.  The actual runner is started/stopped via the
  web API layer (``/api/agi-loop/start``, ``/api/agi-loop/stop``).
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"
_DATA_DIR = _PROJECT_ROOT / "data" / "orion"


# ═══════════════════════════════════════════════════════════════════
#  LOOP STATE — singleton tracking all runtime data
# ═══════════════════════════════════════════════════════════════════

class AGILoopState:
    """Singleton managing the running state of the AGI loop."""

    _instance: Optional["AGILoopState"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.reset()

    def reset(self):
        self.running: bool = False
        self.paused: bool = False
        self.task: Optional[asyncio.Task] = None
        self.current_tick: int = 0
        self.total_ticks: int = 0
        self.current_loop: int = 0
        self.total_cost: float = 0.0
        self.session_cost: float = 0.0
        self.error_streak: int = 0
        self.tick_history: List[Dict[str, Any]] = []
        self.last_error: Optional[str] = None
        self.started_at: Optional[str] = None
        self.stopped_at: Optional[str] = None
        self.stop_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "running": self.running,
            "paused": self.paused,
            "current_tick": self.current_tick,
            "total_ticks": self.total_ticks,
            "current_loop": self.current_loop,
            "total_cost": round(self.total_cost, 6),
            "session_cost": round(self.session_cost, 6),
            "error_streak": self.error_streak,
            "last_error": self.last_error,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "stop_reason": self.stop_reason,
            "recent_ticks": self.tick_history[-20:],
        }

    def log_tick(self, tick_data: dict):
        self.tick_history.append(tick_data)
        # Cap in-memory history at 200
        if len(self.tick_history) > 200:
            self.tick_history = self.tick_history[-200:]
        self._persist_tick(tick_data)

    def _persist_tick(self, tick_data: dict):
        """Append the latest tick to a JSONL file on disk."""
        path = _DATA_DIR / "agi_loop_history.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(tick_data, default=str) + "\n")
        except Exception as exc:
            log.warning("[agi_loop] Failed to persist tick: %s", exc)


# Global singleton
_state = AGILoopState()


def get_loop_state() -> AGILoopState:
    return _state


# ═══════════════════════════════════════════════════════════════════
#  TOOL DEFINITION — callable by agents
# ═══════════════════════════════════════════════════════════════════

class AGILoopTool:
    """Tool interface for agents to query/control the AGI loop."""

    @staticmethod
    def definition() -> dict:
        return {
            "name": "agi_loop",
            "description": (
                "Query or control the autonomous AGI execution loop. "
                "Actions: "
                "'status' — get current loop state, tick count, cost, errors; "
                "'tick_history' — get recent tick results with optional limit; "
                "'request_pause' — request the loop to pause after current tick; "
                "'request_resume' — request the loop to resume from pause."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "status",
                            "tick_history",
                            "request_pause",
                            "request_resume",
                        ],
                        "description": "The action to perform.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max tick history entries to return (default 10).",
                    },
                },
                "required": ["action"],
            },
        }

    @staticmethod
    def execute(arguments: dict) -> str:
        action = arguments.get("action", "status")
        state = get_loop_state()

        if action == "status":
            return json.dumps(state.to_dict(), indent=2)

        elif action == "tick_history":
            limit = arguments.get("limit", 10)
            history = state.tick_history[-limit:]
            return json.dumps({
                "ticks": history,
                "total_recorded": len(state.tick_history),
            }, indent=2)

        elif action == "request_pause":
            if not state.running:
                return json.dumps({"ok": False, "reason": "Loop is not running"})
            state.paused = True
            return json.dumps({
                "ok": True,
                "message": "Pause requested — loop will pause after current tick",
            })

        elif action == "request_resume":
            if not state.paused:
                return json.dumps({"ok": False, "reason": "Loop is not paused"})
            state.paused = False
            return json.dumps({"ok": True, "message": "Loop resumed"})

        return json.dumps({"error": f"Unknown action: {action}"})
