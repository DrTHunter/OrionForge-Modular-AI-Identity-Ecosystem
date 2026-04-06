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
import shutil
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

# Seed vault — canonical default memories copied into every new user vault
SEED_VAULT_PATH = _DATA_DIR / "memory" / "seed_vault.jsonl"

# Seed chats — example conversations copied into every new user's chat list
SEED_CHATS_DIR = _DATA_DIR / "chats" / "seed"

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


def seed_user_vault(user_id: str) -> bool:
    """Copy seed memories into a new user's vault if it is empty.

    The seed file (``data/memory/seed_vault.jsonl``) contains the
    canonical set of platform/operator memories that every user
    should start with.  This is a byte-level copy — IDs and
    timestamps are preserved so all users share the same baseline.

    Returns True if seeding was performed, False if skipped (vault
    already has content or no seed file exists).
    """
    vault = user_vault_path(user_id)

    # Skip if vault already has content (not a new user)
    if vault.exists() and vault.stat().st_size > 0:
        return False

    if not SEED_VAULT_PATH.exists():
        log.warning("[seed] Seed vault not found at %s — skipping", SEED_VAULT_PATH)
        return False

    try:
        shutil.copy2(SEED_VAULT_PATH, vault)
        log.info("[seed] Seeded vault for user %s with %s",
                 user_id[:8], SEED_VAULT_PATH.name)
        return True
    except Exception as exc:
        log.warning("[seed] Failed to seed vault for %s: %s", user_id[:8], exc)
        return False


def seed_user_chats(user_id: str) -> bool:
    """Copy seed chats into a new user's chat directory.

    The seed directory (``data/chats/seed/``) contains an index.json
    and individual chat JSON files that serve as example conversations
    for new users.

    Returns True if seeding was performed, False if skipped.
    """
    chats_dir = user_chats_dir(user_id)
    index_path = chats_dir / "index.json"

    # Skip if user already has a chat index with content
    if index_path.exists() and index_path.stat().st_size > 0:
        try:
            existing = json.loads(index_path.read_text(encoding="utf-8"))
            if existing.get("chats"):
                return False
        except Exception:
            pass

    seed_index = SEED_CHATS_DIR / "index.json"
    if not seed_index.exists():
        log.warning("[seed] Seed chats index not found at %s — skipping", seed_index)
        return False

    try:
        # Copy index
        shutil.copy2(seed_index, index_path)

        # Copy each chat file referenced in the seed index
        seed_data = json.loads(seed_index.read_text(encoding="utf-8"))
        for chat in seed_data.get("chats", []):
            chat_file = SEED_CHATS_DIR / f"{chat['id']}.json"
            if chat_file.exists():
                shutil.copy2(chat_file, chats_dir / chat_file.name)

        log.info("[seed] Seeded chats for user %s (%d chats)",
                 user_id[:8], len(seed_data.get("chats", [])))
        return True
    except Exception as exc:
        log.warning("[seed] Failed to seed chats for %s: %s", user_id[:8], exc)
        return False
