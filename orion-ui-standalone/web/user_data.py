"""Per-user data isolation layer.

All user-specific data lives under ``data/users/{user_id}/``.
Global templates (profiles, directives, prompts) stay in their
original locations and are *read-only* shared resources.

Directory layout per user
─────────────────────────
data/users/{user_id}/
    chats/
        index.json          # chat metadata index
        {chat_id}.json      # individual chats
    memory/
        vault.jsonl          # user's memory vault
        faiss/               # user's FAISS indexes
    notes/
        index.json           # knowledge notes index
        folders.json         # knowledge folders
        {note_id}.json       # individual notes
    settings.json            # per-user settings (preferences, agent configs, avatars)
    profiles/                # user-customised agent overrides (copy-on-write)
        {name}.yaml
    prompts/                 # user-customised system prompts (copy-on-write)
        {name}.system.md
    directives/              # user-customised soul scripts (copy-on-write)
        {name}.md
    uploads/                 # user-uploaded images (avatars, backgrounds)
    trash/
        profiles/
            index.json
"""

import json
import logging
import os
import re
from pathlib import Path

log = logging.getLogger("soulscript.user_data")

# ── Root paths ───────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"
_USERS_DIR = _DATA_DIR / "users"

# Global template directories (read-only shared resources)
GLOBAL_PROFILES_DIR = _PROJECT_ROOT / "profiles"
GLOBAL_PROMPTS_DIR = _PROJECT_ROOT / "prompts"
GLOBAL_DIRECTIVES_DIR = _PROJECT_ROOT / "directives"

# Regex: Supabase UUIDs or short IDs.  Block path traversal.
_SAFE_ID = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


def _validate_user_id(user_id: str) -> str:
    """Sanitise and validate a user_id to prevent path traversal."""
    uid = (user_id or "").strip()
    if not uid or not _SAFE_ID.match(uid):
        raise ValueError(f"Invalid user_id: {user_id!r}")
    return uid


# ── Per-user directory builders ──────────────────────────────────

def user_root(user_id: str) -> Path:
    uid = _validate_user_id(user_id)
    p = _USERS_DIR / uid
    p.mkdir(parents=True, exist_ok=True)
    return p


def user_chats_dir(user_id: str) -> Path:
    p = user_root(user_id) / "chats"
    p.mkdir(parents=True, exist_ok=True)
    return p


def user_memory_dir(user_id: str) -> Path:
    p = user_root(user_id) / "memory"
    p.mkdir(parents=True, exist_ok=True)
    return p


def user_faiss_dir(user_id: str) -> Path:
    p = user_root(user_id) / "memory" / "faiss"
    p.mkdir(parents=True, exist_ok=True)
    return p


def user_vault_path(user_id: str) -> Path:
    user_memory_dir(user_id)  # ensure exists
    return user_root(user_id) / "memory" / "vault.jsonl"


def user_notes_dir(user_id: str) -> Path:
    p = user_root(user_id) / "notes"
    p.mkdir(parents=True, exist_ok=True)
    return p


def user_settings_path(user_id: str) -> Path:
    return user_root(user_id) / "settings.json"


def user_profiles_dir(user_id: str) -> Path:
    p = user_root(user_id) / "profiles"
    p.mkdir(parents=True, exist_ok=True)
    return p


def user_prompts_dir(user_id: str) -> Path:
    p = user_root(user_id) / "prompts"
    p.mkdir(parents=True, exist_ok=True)
    return p


def user_directives_dir(user_id: str) -> Path:
    p = user_root(user_id) / "directives"
    p.mkdir(parents=True, exist_ok=True)
    return p


def user_uploads_dir(user_id: str) -> Path:
    p = user_root(user_id) / "uploads"
    p.mkdir(parents=True, exist_ok=True)
    return p


def user_trash_dir(user_id: str) -> Path:
    p = user_root(user_id) / "trash" / "profiles"
    p.mkdir(parents=True, exist_ok=True)
    return p


def user_folders_file(user_id: str) -> Path:
    return user_notes_dir(user_id) / "folders.json"


# ── Migration helper ─────────────────────────────────────────────

def ensure_user_dirs(user_id: str) -> Path:
    """Create all per-user subdirectories on first login.

    Returns the user root path.
    """
    root = user_root(user_id)
    for sub in ("chats", "memory", "memory/faiss", "notes",
                "profiles", "prompts", "directives", "uploads",
                "trash/profiles"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root
