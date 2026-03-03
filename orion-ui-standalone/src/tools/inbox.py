"""inbox tool — unified agent-to-operator messaging and task queue.

Backed by a single JSONL file: data/shared/inbox.jsonl
Each line: {"id", "from", "type", "priority", "subject", "body",
            "needs_approval", "status", "created_at", "done_at"}

A derived markdown view is kept at data/shared/inbox.md for easy
operator reading.  The JSONL is the canonical truth store.

Combines direct messaging (messages, warnings, ideas, permission requests)
with task management (add tasks, fetch next, acknowledge).  The AGI loop
uses this tool whenever it needs to communicate with the operator.
"""

import json
import os
import re
import tempfile
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from src.data_paths import inbox_path, shared_dir

_SAFE_NAME = re.compile(r"^[a-zA-Z0-9_-]+$")


def _jsonl_path() -> str:
    """Compute JSONL path lazily so DATA_ROOT overrides take effect."""
    return inbox_path()


def _md_path() -> str:
    """Compute markdown path lazily so DATA_ROOT overrides take effect."""
    return os.path.join(os.path.dirname(_jsonl_path()), "inbox.md")

_VALID_TYPES = {"message", "task", "tool_request", "warning", "idea"}
_VALID_PRIORITIES = {"low", "normal", "high", "urgent"}
_VALID_ACTIONS = {"send", "add_task", "next_task", "ack"}

_MAX_SUBJECT = 120
_MAX_BODY = 2000


def _ensure_dir() -> None:
    shared_dir()  # auto-creates


def _read_all() -> List[Dict[str, Any]]:
    """Read every line from the JSONL file."""
    path = _jsonl_path()
    if not os.path.exists(path):
        return []
    entries: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _write_all(entries: List[Dict[str, Any]]) -> None:
    """Atomic rewrite of the entire JSONL file."""
    _ensure_dir()
    fd, tmp = tempfile.mkstemp(dir=shared_dir(), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        os.replace(tmp, _jsonl_path())
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _append_one(entry: Dict[str, Any]) -> None:
    """Append a single JSONL line."""
    _ensure_dir()
    with open(_jsonl_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _append_md(entry: Dict[str, Any]) -> None:
    """Append a human-readable markdown entry to the derived view."""
    md = _md_path()
    os.makedirs(os.path.dirname(md) or ".", exist_ok=True)

    # Ensure header exists
    if not os.path.exists(md) or os.path.getsize(md) == 0:
        with open(md, "w", encoding="utf-8") as f:
            f.write("# Inbox\n")

    priority_tag = ""
    if entry.get("priority") in ("high", "urgent"):
        priority_tag = f" [{entry['priority'].upper()}]"

    approval_tag = ""
    if entry.get("needs_approval"):
        approval_tag = " [NEEDS APPROVAL]"

    entry_type = entry.get("type", "message")
    subject = entry.get("subject", entry.get("task", ""))
    body = entry.get("body", entry.get("task", ""))
    sender = entry.get("from", "system")

    lines = [
        "",
        "---",
        "",
        f"## {entry['created_at'][:16].replace('T', ' ')} UTC | {sender} | {entry_type}{priority_tag}{approval_tag}",
        "",
        f"**Subject:** {subject}",
        "",
        body,
        "",
        f"*id: {entry['id']} — status: {entry.get('status', 'unread')}*",
        "",
    ]

    with open(md, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


class InboxTool:
    """Unified agent-to-operator messaging and task queue.

    The AGI loop uses this tool to:
    - Send messages to the operator when it wants to say something
    - Request the operator to do something or approve an action
    - Add tasks to a shared queue
    - Fetch and acknowledge pending tasks
    """

    @staticmethod
    def definition() -> dict:
        return {
            "name": "inbox",
            "description": (
                "Unified inbox for agent-to-operator communication. "
                "Backed by data/shared/inbox.jsonl with a derived "
                "inbox.md for human reading.\n\n"
                "ACTIONS:\n"
                "- send: message the operator. Requires 'subject' and 'body'. "
                "Set 'type' to classify (message/warning/tool_request/idea). "
                "Set 'needs_approval' if you need the operator to approve something.\n"
                "- add_task: add a task to the shared queue. Requires 'task'. "
                "Optionally set 'priority' and 'profile'.\n"
                "- next_task: fetch and auto-complete the oldest pending task. "
                "Optionally filter by 'profile'. Returns the task text or NO_TASK.\n"
                "- ack: acknowledge a task or message by its ID. Requires 'task_id'.\n\n"
                "All actions support 'dry_run' to preview without writing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["send", "add_task", "next_task", "ack"],
                        "description": (
                            "'send' to message the operator, "
                            "'add_task' to create a pending task, "
                            "'next_task' to pop the oldest pending task, "
                            "'ack' to acknowledge a task or message by ID."
                        ),
                    },
                    "type": {
                        "type": "string",
                        "enum": ["message", "tool_request", "warning", "idea"],
                        "description": (
                            "Message type (used with 'send' action only). Default 'message'. "
                            "'message' for general comms, "
                            "'tool_request' for permission/tool expansion requests, "
                            "'warning' for boundary or safety concerns, "
                            "'idea' for proposals or suggestions."
                        ),
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "normal", "high", "urgent"],
                        "description": (
                            "Priority level for 'send' and 'add_task'. Default 'normal'. "
                            "Use 'urgent' only for safety or boundary issues."
                        ),
                    },
                    "subject": {
                        "type": "string",
                        "description": f"Short one-line summary (max {_MAX_SUBJECT} chars). Required for 'send'.",
                    },
                    "body": {
                        "type": "string",
                        "description": f"Full message text (max {_MAX_BODY} chars). Required for 'send'.",
                    },
                    "task": {
                        "type": "string",
                        "description": "Task description text. Required for 'add_task'.",
                    },
                    "task_id": {
                        "type": "string",
                        "description": "Entry ID to acknowledge. Required for 'ack'.",
                    },
                    "profile": {
                        "type": "string",
                        "description": (
                            "Profile name. Used as sender identity for send/add_task, "
                            "and as filter for next_task. Defaults to 'system'."
                        ),
                    },
                    "needs_approval": {
                        "type": "boolean",
                        "description": (
                            "Set true if this requires operator approval "
                            "before proceeding (for 'send' action). Default false."
                        ),
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": (
                            "If true, preview the action without writing to the inbox. "
                            "Works with all actions. Default false."
                        ),
                    },
                },
                "required": ["action"],
            },
        }

    @staticmethod
    def execute(arguments: dict) -> str:
        action = arguments.get("action", "")

        if action not in _VALID_ACTIONS:
            return (
                f"Error: unknown action '{action}'. "
                f"Use one of: {', '.join(sorted(_VALID_ACTIONS))}."
            )

        if action == "send":
            return _action_send(arguments)
        if action == "add_task":
            return _action_add_task(arguments)
        if action == "next_task":
            return _action_next_task(arguments)
        if action == "ack":
            return _action_ack(arguments)

        return f"Error: unhandled action '{action}'."


# ---- action implementations ------------------------------------------------

def _action_send(arguments: dict) -> str:
    """Send a direct message to the operator."""
    msg_type = arguments.get("type", "message")
    if msg_type not in _VALID_TYPES:
        return (
            f"Error: invalid type '{msg_type}'. "
            f"Must be one of: {', '.join(sorted(_VALID_TYPES))}."
        )

    priority = arguments.get("priority", "normal")
    if priority not in _VALID_PRIORITIES:
        return (
            f"Error: invalid priority '{priority}'. "
            f"Must be one of: {', '.join(sorted(_VALID_PRIORITIES))}."
        )

    subject = (arguments.get("subject") or "").strip()
    if not subject:
        return "Error: 'subject' is required for action 'send'."
    if len(subject) > _MAX_SUBJECT:
        return f"Error: 'subject' exceeds {_MAX_SUBJECT} characters ({len(subject)})."

    body = (arguments.get("body") or "").strip()
    if not body:
        return "Error: 'body' is required for action 'send'."
    if len(body) > _MAX_BODY:
        return f"Error: 'body' exceeds {_MAX_BODY} characters ({len(body)})."

    needs_approval = bool(arguments.get("needs_approval", False))
    sender = arguments.get("_from", arguments.get("profile", "system"))

    entry = {
        "id": uuid.uuid4().hex[:12],
        "from": sender,
        "type": msg_type,
        "priority": priority,
        "subject": subject,
        "body": body,
        "needs_approval": needs_approval,
        "status": "unread",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    _append_one(entry)
    _append_md(entry)

    return f"Message sent (id={entry['id']}): {subject}"


def _action_add_task(arguments: dict) -> str:
    """Add a task to the shared queue."""
    task = (arguments.get("task") or "").strip()
    if not task:
        return "Error: 'task' is required for action 'add_task'."

    profile = arguments.get("profile", arguments.get("_from", "system"))
    dry_run = bool(arguments.get("dry_run", False))

    entry = {
        "id": uuid.uuid4().hex[:12],
        "from": profile,
        "type": "task",
        "priority": arguments.get("priority", "normal"),
        "subject": task[:_MAX_SUBJECT],
        "task": task,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "done_at": None,
    }

    if dry_run:
        return f"[DRY_RUN] Would add task (id={entry['id']}): {task}"

    _append_one(entry)
    _append_md(entry)

    return f"Task added (id={entry['id']}): {task}"


def _action_next_task(arguments: dict) -> str:
    """Fetch and mark done the oldest pending task."""
    profile = arguments.get("profile")
    dry_run = bool(arguments.get("dry_run", False))

    entries = _read_all()
    target_idx = None
    for i, e in enumerate(entries):
        if e.get("type") != "task" or e.get("status") != "pending":
            continue
        if profile and e.get("from") != profile:
            continue
        target_idx = i
        break

    if target_idx is None:
        return "NO_TASK — no pending tasks found."

    t = entries[target_idx]
    if dry_run:
        return f"[DRY_RUN] Would fetch and mark done: id={t['id']} | {t.get('task', t.get('subject', ''))}"

    entries[target_idx]["status"] = "done"
    entries[target_idx]["done_at"] = datetime.now(timezone.utc).isoformat()
    _write_all(entries)

    return f"TASK_FOUND id={t['id']} | {t.get('task', t.get('subject', ''))}"


def _action_ack(arguments: dict) -> str:
    """Acknowledge a task or message by ID."""
    task_id = (arguments.get("task_id") or "").strip()
    if not task_id:
        return "Error: 'task_id' is required for action 'ack'."

    dry_run = bool(arguments.get("dry_run", False))
    entries = _read_all()
    target_idx = None
    for i, e in enumerate(entries):
        if e.get("id") == task_id:
            target_idx = i
            break

    if target_idx is None:
        return f"Error: entry '{task_id}' not found."

    entry = entries[target_idx]
    if entry.get("status") == "acknowledged":
        return f"Entry '{task_id}' is already acknowledged."

    if dry_run:
        return f"[DRY_RUN] Would acknowledge id={task_id} | {entry.get('subject', entry.get('task', ''))}"

    entries[target_idx]["status"] = "acknowledged"
    entries[target_idx]["acked_at"] = datetime.now(timezone.utc).isoformat()
    _write_all(entries)

    return f"Acknowledged (id={task_id}): {entry.get('subject', entry.get('task', ''))}"
