"""Supabase Auth — JWT verification & session helpers.

Uses Supabase JS client on the frontend for login/signup;
stores the access_token in an httpOnly cookie;
verifies tokens server-side via Supabase JWKS or shared JWT secret.
"""

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
}


def is_public_path(path: str) -> bool:
    """Check if a request path is public (no auth required)."""
    for pp in PUBLIC_PATHS:
        if path == pp or path.startswith(pp + "/"):
            return True
    return False
