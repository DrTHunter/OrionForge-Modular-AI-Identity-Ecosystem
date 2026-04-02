"""Encrypt / decrypt API keys at rest using Fernet (AES-128-CBC + HMAC-SHA256).

Keys are encrypted before writing to disk and decrypted on load.  The
encryption master key is derived from the ``EMAIL_ENCRYPTION_KEY`` env-var
(already set on Fly.io) via PBKDF2-HMAC-SHA256 with a fixed application
salt.  If the env-var is absent a deterministic fallback is used so
local-dev still works (keys are *obfuscated* rather than truly secret).

Stored format
─────────────
  Encrypted values are prefixed with ``enc:`` followed by the Fernet token.
  Legacy plaintext values (no prefix) are transparently read and will be
  re-encrypted on the next save cycle.

Per-user isolation
──────────────────
  Each authenticated user gets their own encrypted key file at::

      data/users/<user_id>/api_keys.enc.json

  When auth is disabled (single-user / local dev) the user-id falls
  back to ``__local__`` so the file lives at
  ``data/users/__local__/api_keys.enc.json``.
"""

import base64
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

log = logging.getLogger("soulscript")

# ── Derive Fernet key from env ───────────────────────────────────
_SALT = b"orionforge-keyvault-v1"
_ITERATIONS = 260_000  # OWASP minimum for PBKDF2-SHA256

_fernet: Fernet | None = None


def _get_fernet() -> Fernet:
    global _fernet
    if _fernet is not None:
        return _fernet
    raw = os.environ.get("EMAIL_ENCRYPTION_KEY", "").strip()
    if not raw:
        raw = "orionforge-local-dev-key"
    dk = hashlib.pbkdf2_hmac("sha256", raw.encode(), _SALT, _ITERATIONS)
    _fernet = Fernet(base64.urlsafe_b64encode(dk))
    return _fernet


# ── Encrypt / decrypt helpers ────────────────────────────────────

def encrypt_value(plaintext: str) -> str:
    """Encrypt a single string value.  Empty → empty."""
    if not plaintext:
        return ""
    return "enc:" + _get_fernet().encrypt(plaintext.encode("utf-8")).decode("ascii")


def decrypt_value(stored: str) -> str:
    """Decrypt a stored value.  Handles legacy plaintext (no ``enc:`` prefix)."""
    if not stored:
        return ""
    if not stored.startswith("enc:"):
        return stored  # legacy plaintext
    try:
        return _get_fernet().decrypt(stored[4:].encode("ascii")).decode("utf-8")
    except (InvalidToken, Exception) as exc:
        log.warning("[key_vault] Decryption failed — returning empty: %s", exc)
        return ""


def mask_value(plaintext: str) -> str:
    """Return a UI-safe masked representation of *plaintext*."""
    if not plaintext:
        return ""
    if len(plaintext) > 8:
        return plaintext[:4] + "•" * (len(plaintext) - 8) + plaintext[-4:]
    return "•" * len(plaintext)


# ── Sensitive field names ────────────────────────────────────────
# Keys in settings sub-dicts that hold secret material.
_SECRET_FIELDS: dict[str, set[str]] = {
    "api_keys": {"openai", "anthropic", "deepseek", "google_gemini",
                 "openrouter", "inworld"},
    "image": {"openai_api_key", "google_api_key", "stability_api_key",
              "ideogram_api_key", "replicate_api_key", "fal_api_key",
              "leonardo_api_key", "midjourney_api_key"},
    "tts": {"elevenlabs_api_key", "openedai_cloud_api_key"},
}


def encrypt_settings_secrets(settings: dict) -> dict:
    """Return a *copy* of *settings* with all secret fields encrypted."""
    out = {}
    for k, v in settings.items():
        if k in _SECRET_FIELDS and isinstance(v, dict):
            section = dict(v)
            for field in _SECRET_FIELDS[k]:
                raw = section.get(field, "")
                if raw and not raw.startswith("enc:"):
                    section[field] = encrypt_value(raw)
            out[k] = section
        else:
            out[k] = v
    return out


def decrypt_settings_secrets(settings: dict) -> dict:
    """Return a *copy* of *settings* with all secret fields decrypted."""
    out = {}
    for k, v in settings.items():
        if k in _SECRET_FIELDS and isinstance(v, dict):
            section = dict(v)
            for field in _SECRET_FIELDS[k]:
                stored = section.get(field, "")
                if stored and stored.startswith("enc:"):
                    section[field] = decrypt_value(stored)
            out[k] = section
        else:
            out[k] = v
    return out


def strip_secrets_for_template(settings: dict) -> dict:
    """Return a *copy* of *settings* with all secret fields blanked out.

    Safe to pass to Jinja2 templates — no keys leak into HTML source.
    Non-secret fields (ollama_url, voice selections, etc.) are preserved.
    """
    out = {}
    for k, v in settings.items():
        if k in _SECRET_FIELDS and isinstance(v, dict):
            section = dict(v)
            for field in _SECRET_FIELDS[k]:
                section[field] = ""
            out[k] = section
        else:
            out[k] = v
    return out


# ── Per-user encrypted key storage ───────────────────────────────
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _user_keys_path(user_id: str) -> Path:
    safe_id = "".join(c for c in user_id if c.isalnum() or c in "-_")
    if not safe_id:
        safe_id = "__local__"
    return _DATA_DIR / "users" / safe_id / "api_keys.enc.json"


def save_user_keys(user_id: str, category: str, keys: dict[str, str]) -> None:
    """Encrypt and persist keys for *user_id* under *category*.

    *category* is one of ``"api_keys"``, ``"image"``, ``"tts"``.
    """
    path = _user_keys_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing
    existing: dict[str, dict] = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, ValueError):
            existing = {}

    # Encrypt secret fields, leave non-secret fields as-is
    secret_fields = _SECRET_FIELDS.get(category, set())
    encrypted = {}
    for k, v in keys.items():
        if k in secret_fields and v and not v.startswith("enc:"):
            encrypted[k] = encrypt_value(v)
        else:
            encrypted[k] = v

    existing[category] = encrypted
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)


def load_user_keys(user_id: str, category: str) -> dict[str, str]:
    """Load and decrypt keys for *user_id* under *category*."""
    path = _user_keys_path(user_id)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, ValueError):
        return {}
    section = data.get(category, {})
    secret_fields = _SECRET_FIELDS.get(category, set())
    decrypted = {}
    for k, v in section.items():
        if k in secret_fields:
            decrypted[k] = decrypt_value(v)
        else:
            decrypted[k] = v
    return decrypted


def load_user_keys_masked(user_id: str, category: str) -> dict[str, str]:
    """Load keys for *user_id* returning masked values for secrets."""
    raw = load_user_keys(user_id, category)
    secret_fields = _SECRET_FIELDS.get(category, set())
    masked = {}
    for k, v in raw.items():
        if k in secret_fields:
            masked[k] = mask_value(v)
        else:
            masked[k] = v
    return masked


def delete_user_keys(user_id: str) -> None:
    """Remove all stored keys for a user (called on account wipe)."""
    path = _user_keys_path(user_id)
    if path.exists():
        path.unlink()
        log.info("[key_vault] Deleted encrypted keys for user %s", user_id)
