"""Personal MCP access tokens.

Each user gets one long-lived token that they paste into Claude Code / Claude
Desktop / Gemini to connect to the hosted OrionForge MCP endpoint (``/mcp``).

Security model
──────────────
- The raw token is shown to the user exactly once, on mint. We store only its
  SHA-256 hash, so a leaked registry file cannot be replayed as a token.
- Lookup is by hash; the final compare is constant-time (``hmac.compare_digest``)
  to avoid leaking timing information, mirroring ``auth.verify_bridge_key``.
- One active token per user — minting again replaces the previous one (rotation).

Registry: ``data/mcp_tokens.json`` (gitignored)
    { "<sha256(token)>": {"user_id", "created", "prefix", "label"} }
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger("soulscript.mcp_tokens")

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# Registry location. In production set ORION_MCP_TOKENS_FILE to a path on the
# persistent volume (e.g. /persist/mcp_tokens.json) so tokens survive deploys —
# data/mcp_tokens.json itself is NOT on the Fly volume (only certain subdirs are).
_TOKENS_FILE = Path(
    os.environ.get("ORION_MCP_TOKENS_FILE") or (_DATA_DIR / "mcp_tokens.json")
)

_TOKEN_PREFIX = "oforge_"
_lock = threading.Lock()


# ── Registry I/O ────────────────────────────────────────────────────
def _load() -> dict:
    if not _TOKENS_FILE.exists():
        return {}
    try:
        with open(_TOKENS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("[mcp_tokens] could not read registry: %s", exc)
        return {}


def _save(registry: dict) -> None:
    _TOKENS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _TOKENS_FILE.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    os.replace(tmp, _TOKENS_FILE)  # atomic


def _hash(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ── Public API ──────────────────────────────────────────────────────
def mint_token(user_id: str, label: str = "") -> str:
    """Create (or rotate) the user's token and return the raw value once.

    Any previously issued token for this user is revoked.
    """
    uid = (user_id or "").strip()
    if not uid:
        raise ValueError("mint_token: empty user_id")
    raw = _TOKEN_PREFIX + secrets.token_urlsafe(32)
    entry = {
        "user_id": uid,
        "created": datetime.now(timezone.utc).isoformat(),
        # A short, non-secret prefix so the UI can show "which" token without
        # storing the secret. token_urlsafe is unpadded, so this is safe to show.
        "prefix": raw[: len(_TOKEN_PREFIX) + 6],
        "label": (label or "").strip()[:64],
    }
    with _lock:
        registry = _load()
        # Drop any existing tokens for this user (one active token per user).
        registry = {h: e for h, e in registry.items() if e.get("user_id") != uid}
        registry[_hash(raw)] = entry
        _save(registry)
    log.info("[mcp_tokens] minted token for user %s", uid[:8])
    return raw


def verify_token(raw: str) -> Optional[str]:
    """Return the user_id for a valid token, else None."""
    if not raw or not isinstance(raw, str):
        return None
    h = _hash(raw.strip())
    registry = _load()
    entry = registry.get(h)
    if not entry:
        return None
    # Constant-time confirmation that the stored hash matches what we computed.
    for stored_hash in registry:
        if hmac.compare_digest(stored_hash, h):
            return entry.get("user_id") or None
    return None


def revoke_token(user_id: str) -> bool:
    """Remove the user's token(s). Returns True if anything was removed."""
    uid = (user_id or "").strip()
    if not uid:
        return False
    with _lock:
        registry = _load()
        kept = {h: e for h, e in registry.items() if e.get("user_id") != uid}
        removed = len(kept) != len(registry)
        if removed:
            _save(kept)
    return removed


def get_token_status(user_id: str) -> dict:
    """Return non-secret status for the user's token.

    ``{"exists": bool, "prefix": str, "created": str, "label": str}`` — never the
    raw token (we don't have it; only its hash is stored).
    """
    uid = (user_id or "").strip()
    if not uid:
        return {"exists": False}
    registry = _load()
    for entry in registry.values():
        if entry.get("user_id") == uid:
            return {
                "exists": True,
                "prefix": entry.get("prefix", ""),
                "created": entry.get("created", ""),
                "label": entry.get("label", ""),
            }
    return {"exists": False}
