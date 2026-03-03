"""continuation_update tool — maintain a single per-profile pickup file.

Backed by: data/{profile}_continuation.md (one file per profile, never more).
"""

import os
import re
import time
from typing import Optional

from src.data_paths import continuation_path as _continuation_path, profile_dir

_SAFE_NAME = re.compile(r"^[a-zA-Z0-9_-]+$")


def _ensure_dir() -> None:
    pass  # profile_dir auto-creates


def _file_for(profile: str) -> str:
    return _continuation_path(profile)


class ContinuationUpdateTool:
    """Writes or updates a single per-profile continuation markdown file."""

    @staticmethod
    def definition() -> dict:
        return {
            "name": "continuation_update",
            "description": (
                "Maintain a single evolving 'pickup later' markdown file per profile. "
                "File: data/{profile}_continuation.md (one per profile, never more).\n\n"
                "MODES:\n"
                "- append: add a timestamped block to the end of the file. Good for "
                "logging progress, notes, or session summaries.\n"
                "- replace_section: upsert a named section (## heading). If the section "
                "exists, its content is replaced in-place. If not, it's appended. "
                "Requires 'section' parameter with the heading name.\n\n"
                "Use this to leave notes for yourself across sessions, track ongoing "
                "work, or maintain a running context file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "profile": {
                        "type": "string",
                        "description": (
                            "Profile name (e.g. 'callum', 'astraea'). Required. "
                            "Must be alphanumeric with hyphens/underscores only."
                        ),
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["append", "replace_section"],
                        "description": (
                            "'append' adds a timestamped block at the end; "
                            "'replace_section' upserts a named section by heading. "
                            "Default: 'append'."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": "Markdown content to write. Required.",
                    },
                    "section": {
                        "type": "string",
                        "description": (
                            "Section heading name (without ##). Required when "
                            "mode='replace_section'. Example: 'Current Tasks'."
                        ),
                    },
                },
                "required": ["profile", "content"],
            },
        }

    @staticmethod
    def execute(arguments: dict) -> str:
        profile = arguments.get("profile", "")
        mode = arguments.get("mode", "append")
        content = arguments.get("content", "")
        section = arguments.get("section", "")

        if not _SAFE_NAME.match(profile):
            return f"Error: invalid profile name '{profile}'"
        if not content.strip():
            return "Error: 'content' must not be empty."

        _ensure_dir()
        path = _file_for(profile)

        if mode == "append":
            return _mode_append(path, content)
        if mode == "replace_section":
            if not section.strip():
                return "Error: 'section' is required for mode 'replace_section'."
            return _mode_replace_section(path, content, section.strip())
        return f"Error: unknown mode '{mode}'. Use 'append' or 'replace_section'."


# ---- mode implementations --------------------------------------------------

def _mode_append(path: str, content: str) -> str:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    block = f"\n## {ts}\n\n{content.strip()}\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(block)
    return f"Appended timestamped block to {path}"


def _mode_replace_section(path: str, content: str, section: str) -> str:
    heading = f"## {section}"
    new_block = f"{heading}\n\n{content.strip()}\n"

    existing = ""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            existing = f.read()

    # Find and replace the section if it exists
    pattern = re.compile(
        r"^(## " + re.escape(section) + r")\s*\n(.*?)(?=\n## |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(existing)
    if match:
        updated = existing[: match.start()] + new_block + existing[match.end():]
    else:
        # Append new section at the end
        updated = existing.rstrip("\n") + "\n\n" + new_block if existing.strip() else new_block

    with open(path, "w", encoding="utf-8") as f:
        f.write(updated)

    action = "Replaced" if match else "Added"
    return f"{action} section '{section}' in {path}"
