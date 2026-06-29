"""Supabase Auth — JWT verification & session helpers.

Uses Supabase JS client on the frontend for login/signup;
stores the access_token in an httpOnly cookie;
verifies tokens server-side via Supabase JWKS or shared JWT secret.
"""

import hmac
import json
import logging
from pathlib import Path
from typing import Optional

import httpx
import jwt  # PyJWT

log = logging.getLogger("soulscript.auth")

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_AUTH_FILE = _CONFIG_DIR / "auth.json"

# ── Cached Supabase JWKS (RS256 public keys) ────────────────────
_jwks_cache: dict | None = None


def _load_auth_config() -> dict:
    """Load auth.json configuration, with env-var overrides for secrets."""
    cfg: dict = {}
    if _AUTH_FILE.exists():
        with open(_AUTH_FILE, "r", encoding="utf-8-sig") as f:
            cfg = json.load(f)
    # Env vars take priority over file values
    import os
    if os.environ.get("SUPABASE_URL"):
        cfg["supabase_url"] = os.environ["SUPABASE_URL"]
    if os.environ.get("SUPABASE_ANON_KEY"):
        cfg["supabase_anon_key"] = os.environ["SUPABASE_ANON_KEY"]
    if not cfg:
        return {"auth_enabled": False}
    return cfg


def get_auth_config() -> dict:
    """Public accessor for auth config (safe subset)."""
    cfg = _load_auth_config()
    return {
        "supabase_url": cfg.get("supabase_url", ""),
        "supabase_anon_key": cfg.get("supabase_anon_key", ""),
        "auth_enabled": cfg.get("auth_enabled", False),
        "free_tier_features": cfg.get("free_tier_features", []),
        "paid_features": cfg.get("paid_features", []),
    }


async def _fetch_jwks(supabase_url: str) -> dict:
    """Fetch Supabase JWKS for token verification (RS256)."""
    global _jwks_cache
    if _jwks_cache:
        return _jwks_cache
    url = f"{supabase_url}/auth/v1/.well-known/jwks.json"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=10)
            resp.raise_for_status()
            _jwks_cache = resp.json()
            return _jwks_cache
    except Exception as exc:
        log.warning("[auth] Failed to fetch JWKS: %s", exc)
        return {}


def verify_supabase_token(token: str) -> Optional[dict]:
    """Verify a Supabase JWT and return the decoded payload.

    Supabase uses HS256 with the JWT secret by default.
    Falls back to unverified decode if no secret configured (dev mode).
    """
    cfg = _load_auth_config()
    jwt_secret = cfg.get("jwt_secret", "")

    if jwt_secret:
        # Verify with the shared JWT secret (HS256)
        try:
            payload = jwt.decode(
                token,
                jwt_secret,
                algorithms=["HS256"],
                audience="authenticated",
                options={"verify_exp": True},
            )
            return payload
        except jwt.ExpiredSignatureError:
            log.debug("[auth] Token expired")
            return None
        except jwt.InvalidTokenError as exc:
            log.debug("[auth] Invalid token: %s", exc)
            return None
    else:
        # Dev mode — decode without verification but still check structure
        try:
            payload = jwt.decode(
                token,
                options={
                    "verify_signature": False,
                    "verify_exp": True,
                },
            )
            # Ensure it looks like a Supabase token
            if payload.get("iss") and "supabase" in payload.get("iss", ""):
                return payload
            return payload  # Accept it in dev mode
        except jwt.ExpiredSignatureError:
            log.debug("[auth] Token expired")
            return None
        except jwt.InvalidTokenError as exc:
            log.debug("[auth] Invalid token: %s", exc)
            return None


def extract_user_from_token(payload: dict) -> dict:
    """Extract user info from decoded JWT payload."""
    return {
        "id": payload.get("sub", ""),
        "email": payload.get("email", ""),
        "role": payload.get("role", "authenticated"),
        "aud": payload.get("aud", ""),
    }


async def refresh_supabase_session(refresh_token: str) -> Optional[dict]:
    """Exchange a Supabase refresh token for a fresh session.

    Returns the Supabase token response (access_token, refresh_token,
    expires_in, …) on success, or None when refresh is unavailable or
    fails. Keeps a login alive past the 1-hour access-token TTL.
    """
    if not refresh_token:
        return None
    cfg = _load_auth_config()
    supabase_url = (cfg.get("supabase_url", "") or "").rstrip("/")
    anon_key = cfg.get("supabase_anon_key", "")
    if not supabase_url or not anon_key:
        return None
    url = f"{supabase_url}/auth/v1/token?grant_type=refresh_token"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url,
                headers={"apikey": anon_key, "Content-Type": "application/json"},
                json={"refresh_token": refresh_token},
                timeout=10,
            )
        if resp.status_code != 200:
            log.debug("[auth] refresh failed: HTTP %s", resp.status_code)
            return None
        data = resp.json()
        return data if data.get("access_token") else None
    except Exception as exc:
        log.warning("[auth] token refresh error: %s", exc)
        return None


# ── Single-user allowlist ───────────────────────────────────────
# Only emails in this list (case-insensitive) may log in.
# Sources, in priority order:
#   1. ALLOWED_EMAILS env var (comma-separated)
#   2. auth.json -> "allowed_emails" array
#   3. Hard-coded default below (owner email)
_DEFAULT_ALLOWED_EMAILS = ("dr.trent.hunter@gmail.com",)


def _allowed_emails() -> set[str]:
    import os
    env_val = os.environ.get("ALLOWED_EMAILS", "").strip()
    if env_val:
        return {e.strip().lower() for e in env_val.split(",") if e.strip()}
    cfg = _load_auth_config()
    cfg_list = cfg.get("allowed_emails") or []
    if isinstance(cfg_list, list) and cfg_list:
        return {str(e).strip().lower() for e in cfg_list if str(e).strip()}
    return {e.lower() for e in _DEFAULT_ALLOWED_EMAILS}


def is_email_allowed(email: str) -> bool:
    """Return True if the email is permitted to log in.

    When open_registration is enabled in auth.json (or OPEN_REGISTRATION=1 env),
    all emails are allowed and the allowlist is ignored.
    When the allowlist is empty, registration is also open (opt-in default).
    """
    if not email:
        return False
    import os
    if os.environ.get("OPEN_REGISTRATION", "").strip() in ("1", "true", "yes"):
        return True
    cfg = _load_auth_config()
    if cfg.get("open_registration"):
        return True
    emails = _allowed_emails()
    if not emails:
        return True  # empty list = open
    return email.strip().lower() in emails


# ── Optional VS Code bridge API key ─────────────────────────────
# A static key that lets the VS Code bridge authenticate without a
# short-lived Supabase JWT. Entirely opt-in: when no key is configured
# the whole mechanism is inert and the normal login flow is untouched.
# Sources, in priority order:
#   1. ORION_BRIDGE_API_KEY env var
#   2. auth.json -> "bridge_api_key"
def get_bridge_api_key() -> str:
    """Return the configured bridge API key, or "" when disabled."""
    import os
    env_val = os.environ.get("ORION_BRIDGE_API_KEY", "").strip()
    if env_val:
        return env_val
    cfg = _load_auth_config()
    return str(cfg.get("bridge_api_key", "") or "").strip()


def verify_bridge_key(provided: str) -> Optional[dict]:
    """Return the owner identity if the provided key matches, else None.

    Uses a constant-time comparison and maps a valid key to the owner
    account so bridge calls share the same chats and memory vault.
    Returns None when bridge-key auth is disabled (no key configured).

    The owner user id is resolved from, in priority order:
      1. ORION_BRIDGE_USER_ID env var (keeps the id out of the repo)
      2. auth.json -> "bridge_user_id"
      3. "__bridge__" (isolated data space) when neither is set
    """
    import os
    expected = get_bridge_api_key()
    if not expected or not provided:
        return None
    if not hmac.compare_digest(provided.strip(), expected):
        return None
    cfg = _load_auth_config()
    emails = sorted(_allowed_emails())
    owner_email = emails[0] if emails else ""
    user_id = (
        os.environ.get("ORION_BRIDGE_USER_ID", "").strip()
        or str(cfg.get("bridge_user_id", "") or "").strip()
        or "__bridge__"
    )
    return {
        "id": user_id,
        "email": owner_email,
        "role": "authenticated",
        "aud": "authenticated",
        "via": "bridge",
    }


# ── Public paths that don't require authentication ──────────────
PUBLIC_PATHS = {
    "/login",
    "/auth/callback",
    "/api/auth/config",
    "/api/auth/session",
    "/api/auth/set-session",
    "/api/auth/logout",
    "/api/auth/user",
    "/api/stripe/webhook",
    "/api/stripe/config",
    "/plans",
    "/static",
    "/uploads",
    "/favicon.ico",
    # Hosted MCP endpoint — does its own bearer-token auth (web/app.py mount).
    # Must bypass the cookie AuthMiddleware, which would 302-redirect to /login.
    "/mcp",
}


def is_public_path(path: str) -> bool:
    """Check if a request path is public (no auth required)."""
    for pp in PUBLIC_PATHS:
        if path == pp or path.startswith(pp + "/"):
            return True
    return False
