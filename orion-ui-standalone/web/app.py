"""SoulScript Engine — Clean MVP Web Layer.

A minimal FastAPI application demonstrating AI identity persistence
through FAISS-backed memory retrieval and soul script injection.

Prompt Injection Order
──────────────────────
1. Base Prompt        — Agent's system prompt (prompts/{agent}.system.md)
2. Soul Script        — FAISS semantic retrieval from directive-mode knowledge
3. Always-On Knowledge — Verbatim text from always-mode attached knowledge
4. Memory Vault       — FAISS semantic search over agent memories (vault.jsonl)
5. Conversation       — Recent user/assistant messages (truncated to budget)
"""

import asyncio
import json
import logging
import os
import queue
import re
import subprocess
import sys
import threading
import time as _time
import uuid
import base64 as _b64
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# Load .env before any module reads os.environ
from dotenv import load_dotenv
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

import httpx
import yaml
from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from starlette.middleware.base import BaseHTTPMiddleware

from web.image_gen import _generate_image
from web.video_gen import _generate_video
from web.auth import get_auth_config, verify_supabase_token, extract_user_from_token, is_public_path, is_email_allowed, verify_bridge_key, refresh_supabase_session
from web.stripe_billing import (
    get_user_tier, get_user_subscription, user_has_feature,
    create_checkout_session, create_billing_portal_session,
    handle_webhook_event, TIER_INFO, STRIPE_PUBLISHABLE_KEY,
    get_user_credits, add_user_credits, deduct_user_credits,
    CREDIT_PACKS, TOOL_CREDIT_COSTS, create_credits_checkout_session,
    fulfill_credits_for_session,
    STORE_CATALOG, LLM_MARKUP_MULTIPLIER, estimate_llm_credit_cost,
    estimate_llm_credit_cost_safe, estimate_image_credit_cost, estimate_video_credit_cost,
    estimate_tts_credit_cost, estimate_stt_credit_cost,
    get_credit_history, get_trial_status, FREE_TRIAL_DAYS, user_has_purchased_credits,
    get_user_purchases, user_owns_item, purchase_tool, purchase_skin,
    purchase_agent, purchase_pack, user_owns_agent, user_has_tool_access, SKIN_PRICES,
    get_store_agent_ids, get_user_unlocked_agents, FREE_AGENT_IDS,
    STORE_PACKS,
    touch_user_activity, wipe_user_data, wipe_user_by_email,
    purge_inactive_users, list_all_users, INACTIVE_ACCOUNT_DAYS,
)
from src.memory.types import VALID_SCOPES, VALID_CATEGORIES
from web.key_vault import (
    encrypt_settings_secrets, decrypt_settings_secrets,
    strip_secrets_for_template,
    save_user_keys, load_user_keys, load_user_keys_masked,
    delete_user_keys, mask_value,
)
from web.user_data import (
    ensure_user_dirs, seed_user_vault, seed_user_chats,
    user_chats_dir, user_memory_dir, user_faiss_dir, user_vault_path,
    user_notes_dir, user_settings_path, user_profiles_dir,
    user_prompts_dir, user_directives_dir, user_uploads_dir,
    user_trash_dir, user_folders_file,
    GLOBAL_PROFILES_DIR, GLOBAL_PROMPTS_DIR, GLOBAL_DIRECTIVES_DIR,
)
import contextvars

# ── Per-request user identity (set by AuthMiddleware) ────────────
_current_user_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_user_id", default="__local__"
)

# ── Project paths ────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR   = _PROJECT_ROOT / "config"
_DATA_DIR     = _PROJECT_ROOT / "data"
_PROFILES_DIR    = _PROJECT_ROOT / "profiles"
_PROMPTS_DIR     = _PROJECT_ROOT / "prompts"
_DIRECTIVES_DIR  = _PROJECT_ROOT / "directives"
_CHATS_DIR       = _DATA_DIR / "chats"
_NOTES_DIR    = _DATA_DIR / "user_notes"
_VAULT_PATH   = _DATA_DIR / "memory" / "vault.jsonl"
_FAISS_DIR    = _DATA_DIR / "memory" / "faiss"
_TRASH_DIR    = _DATA_DIR / "trash" / "profiles"

CONNECTIONS_FILE = _CONFIG_DIR / "connections.json"
SETTINGS_FILE    = _CONFIG_DIR / "settings.json"
PRICING_FILE     = _CONFIG_DIR / "pricing.yaml"

log = logging.getLogger("soulscript")

# ── Ensure data directories exist ────────────────────────────────
for _d in [_CHATS_DIR, _NOTES_DIR, _VAULT_PATH.parent, _FAISS_DIR, _TRASH_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── FastAPI app ──────────────────────────────────────────────────
from contextlib import asynccontextmanager

@asynccontextmanager
async def _lifespan(application: FastAPI):
    """Build NotesFAISS in background so Soul Script retrieval works soon after boot."""
    try:
        _seed_platform_keys_from_env()
    except Exception as exc:
        log.warning("[startup] Platform-key seeding failed: %s", exc)
    # Auto-sync OpenRouter pricing from their live API
    try:
        result = await _sync_openrouter_pricing(force=True)
        log.info("[startup] OpenRouter pricing sync: %s", result)
    except Exception as exc:
        log.warning("[startup] OpenRouter pricing sync failed: %s", exc)
    # Build FAISS index in background thread so health check passes quickly.
    # Use a lock file so only one worker rebuilds; others wait then load.
    import threading
    try:
        import fcntl
    except ImportError:
        fcntl = None
    def _bg_faiss():
        lock_path = Path("data/memory/faiss/.rebuild.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        if not fcntl:
            # No file locking on Windows — just rebuild directly
            try:
                _rebuild_notes_faiss()
                log.info("[startup] NotesFAISS build completed (background)")
            except Exception as exc:
                log.warning("[startup] NotesFAISS build skipped: %s", exc)
            return
        with open(lock_path, "w") as lf:
            got_lock = False
            try:
                fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                got_lock = True
            except OSError:
                log.info("[startup] Another worker is rebuilding NotesFAISS — waiting")
                fcntl.flock(lf, fcntl.LOCK_EX)  # blocking wait
            try:
                if got_lock:
                    _rebuild_notes_faiss()
                    log.info("[startup] NotesFAISS build completed (background)")
                else:
                    log.info("[startup] NotesFAISS rebuild done by sibling — loading cached index")
                from src.storage.note_collector import invalidate_notes_faiss
                invalidate_notes_faiss()
            except Exception as exc:
                log.warning("[startup] NotesFAISS build skipped: %s", exc)
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)
    threading.Thread(target=_bg_faiss, daemon=True).start()
    # Auto-purge inactive accounts (>90 days)
    try:
        result = purge_inactive_users()
        if result.get("purged_count", 0) > 0:
            log.info("[startup] Inactive account cleanup: purged %d user(s)", result["purged_count"])
    except Exception as exc:
        log.warning("[startup] Inactive account purge failed: %s", exc)
    yield

app = FastAPI(title="SoulScript Engine", version="0.2.0", lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# ── Auth file path ───────────────────────────────────────────────
_AUTH_FILE = _CONFIG_DIR / "auth.json"


# ── Authentication Middleware ────────────────────────────────────
# Paths that require auth but NOT an active subscription
SUBSCRIPTION_EXEMPT_PATHS = {
    "/plans",
    "/api/stripe/checkout",
    "/api/stripe/portal",
    "/api/stripe/subscription",
    "/api/stripe/config",
    "/api/credits/balance",
    "/api/credits/buy",
    "/api/auth/logout",
    "/api/auth/session",
    "/api/auth/user",
    "/admin",
    "/api/admin",
}

def _is_subscription_exempt(path: str) -> bool:
    for sp in SUBSCRIPTION_EXEMPT_PATHS:
        if path == sp or path.startswith(sp + "/"):
            return True
    return False


def _session_cookie_domain(request: Request) -> str | None:
    """Return ".orionforge.chat" for branded hosts so one login covers the
    apex and every subdomain; None (host-only) for fly.dev / localhost."""
    host = (request.headers.get("host") or "").split(":")[0].lower()
    if host == "orionforge.chat" or host.endswith(".orionforge.chat"):
        return ".orionforge.chat"
    return None


def _write_session_cookies(request: Request, response, access_token: str, refresh_token: str = "") -> None:
    """Set the auth cookies with a domain that spans the orionforge.chat site."""
    domain = _session_cookie_domain(request)
    response.set_cookie(
        key="sb_access_token",
        value=access_token,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=3600,
        path="/",
        domain=domain,
    )
    if refresh_token:
        response.set_cookie(
            key="sb_refresh_token",
            value=refresh_token,
            httponly=True,
            samesite="lax",
            secure=False,
            max_age=60 * 60 * 24 * 30,
            path="/",
            domain=domain,
        )


def _clear_session_cookies(request: Request, response) -> None:
    """Delete auth cookies for both the domain-scoped and legacy host-only forms."""
    domain = _session_cookie_domain(request)
    response.delete_cookie("sb_access_token", path="/", domain=domain)
    response.delete_cookie("sb_refresh_token", path="/", domain=domain)
    response.delete_cookie("sb_access_token", path="/")
    response.delete_cookie("sb_refresh_token", path="/")


class AuthMiddleware(BaseHTTPMiddleware):
    """Redirect unauthenticated users to /login, unsubscribed to /plans."""

    async def dispatch(self, request: Request, call_next):
        auth_cfg = get_auth_config()
        if not auth_cfg.get("auth_enabled", False):
            response = await call_next(request)
            return response

        path = request.url.path

        # Let public paths through (login page, auth API, static assets)
        if is_public_path(path):
            response = await call_next(request)
            return response

        # ── Optional VS Code bridge key auth (isolated, opt-in) ──
        # Active only when a bridge key is configured server-side
        # (ORION_BRIDGE_API_KEY env or auth.json "bridge_api_key").
        # When the X-Bridge-Key header is absent, this block is skipped
        # entirely and the normal Supabase login flow runs unchanged.
        if path.startswith("/api/"):
            bridge_key = request.headers.get("x-bridge-key", "")
            if bridge_key:
                bridge_user = verify_bridge_key(bridge_key)
                if bridge_user is None:
                    return JSONResponse({"error": "Invalid bridge key"}, status_code=401)
                request.state.user = bridge_user
                user_id = bridge_user.get("id", "") or "__bridge__"
                _current_user_id.set(user_id)
                try:
                    ensure_user_dirs(user_id)
                    seed_user_vault(user_id)
                    seed_user_chats(user_id)
                except Exception as exc:
                    log.warning("[auth] bridge dir init failed for %s: %s", user_id, exc)
                request.state.trial = get_trial_status(user_id)
                request.state.is_subscribed = True
                return await call_next(request)

        # Check for auth cookie. If the access token is missing or expired,
        # fall back to the refresh token so the session survives past the
        # 1-hour access-token TTL instead of bouncing the user to /login.
        token = request.cookies.get("sb_access_token")
        payload = verify_supabase_token(token) if token else None

        refreshed_session = None
        if not payload:
            refresh_token = request.cookies.get("sb_refresh_token")
            if refresh_token:
                new_sess = await refresh_supabase_session(refresh_token)
                if new_sess and new_sess.get("access_token"):
                    new_payload = verify_supabase_token(new_sess["access_token"])
                    if new_payload:
                        payload = new_payload
                        refreshed_session = new_sess

        if not payload:
            if path.startswith("/api/"):
                return JSONResponse({"error": "Not authenticated"}, status_code=401)
            from urllib.parse import quote
            return RedirectResponse(url=f"/login?next={quote(path, safe='/')}", status_code=302)

        # Attach user info to request state
        request.state.user = extract_user_from_token(payload)

        # ── Single-user allowlist gate ──
        # Only the configured owner email may access the app. Anyone else
        # gets their auth cookie wiped and is bounced to /login.
        if not is_email_allowed(request.state.user.get("email", "")):
            log.warning(
                "[auth] Denied login for non-allowlisted email: %s",
                request.state.user.get("email", ""),
            )
            if path.startswith("/api/"):
                resp = JSONResponse({"error": "Access denied for this account"}, status_code=403)
            else:
                resp = RedirectResponse(url="/login?denied=1", status_code=302)
            _clear_session_cookies(request, resp)
            return resp

        # Record user activity for inactive-account cleanup
        user_id = request.state.user.get("id", "")
        _current_user_id.set(user_id or "__local__")
        touch_user_activity(user_id)

        # Ensure per-user data directories exist (fast no-op after first login)
        if user_id:
            try:
                is_new = ensure_user_dirs(user_id)
                seed_user_vault(user_id)
                seed_user_chats(user_id)
                # Seed welcome credits ($2 = 200 credits) on first ever login
                if get_user_credits(user_id) == 0:
                    add_user_credits(user_id, 200, reason="welcome_grant:new_user_$2")
                    log.info("[auth] Seeded 200 welcome credits for new user %s", user_id[:8])
            except Exception as exc:
                log.warning("[auth] Failed to create user dirs for %s: %s", user_id, exc)

        # Attach trial info for templates
        trial = get_trial_status(user_id)
        sub_info = get_user_subscription(user_id)
        request.state.trial = trial
        request.state.is_subscribed = bool(
            sub_info.get("subscription") and sub_info["subscription"].get("status") in ("active", "trialing")
        )

        # No subscription paywall (pay-per-use) ────────────────────────────────────
        # Every authenticated user has full access; usage is billed in
        # credits at 2× the API cost per request — no monthly ($9.99/mo) plan.

        response = await call_next(request)
        # If we minted a fresh access token via the refresh flow, persist it so
        # the browser carries the renewed session on subsequent requests.
        if refreshed_session is not None:
            _write_session_cookies(
                request,
                response,
                refreshed_session.get("access_token", ""),
                refreshed_session.get("refresh_token", ""),
            )
        return response


app.add_middleware(AuthMiddleware)

# ── CSRF protection ─────────────────────────────────────────────
# Cross-origin form submissions (the classic CSRF vector) send
# Content-Type: application/x-www-form-urlencoded or multipart/form-data.
# Our API endpoints expect application/json, which browsers cannot
# send cross-origin without a CORS preflight.  We block the dangerous
# form content types on state-changing /api/ routes.  Stripe webhooks
# are exempt (they use signature verification).
_CSRF_EXEMPT = {"/api/stripe/webhook"}

class CSRFMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            path = request.url.path
            if path.startswith("/api/") and path not in _CSRF_EXEMPT:
                ct = (request.headers.get("content-type") or "").lower()
                # Block cross-origin form-encoded submissions
                if "application/x-www-form-urlencoded" in ct or "multipart/form-data" in ct:
                    return JSONResponse(
                        {"error": "CSRF validation failed — form submissions not allowed on API routes"},
                        status_code=403,
                    )
        return await call_next(request)

app.add_middleware(CSRFMiddleware)

# ── Uploads directory (chat backgrounds, etc.) ──────────────────
_UPLOADS_DIR = _DATA_DIR / "uploads"
_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
_VIDEO_UPLOADS_DIR = _UPLOADS_DIR / "videos"
_VIDEO_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(_UPLOADS_DIR)), name="uploads")


# ── Per-user upload serving ──────────────────────────────────────
@app.get("/api/uploads/{filename:path}")
async def api_user_upload(filename: str, request: Request):
    """Serve a file from the authenticated user's uploads directory."""
    uid = _get_user_id(request)
    if uid and uid != "__local__":
        path = user_uploads_dir(uid) / filename
    else:
        path = _UPLOADS_DIR / filename
    if not path.exists() or not path.is_file():
        # Fallback to global uploads
        path = _UPLOADS_DIR / filename
    if not path.exists() or not path.is_file():
        return JSONResponse({"error": "Not found"}, 404)
    import mimetypes
    media_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return Response(content=path.read_bytes(), media_type=media_type)


# ── Migrate base64 avatar images to files (one-time on startup) ─
def _migrate_base64_avatars():
    """Convert any base64 data-URL avatars in settings.json to saved files."""
    try:
        if not SETTINGS_FILE.exists():
            return
        with open(SETTINGS_FILE, "r", encoding="utf-8-sig") as f:
            settings = json.load(f)
        changed = False
        # Migrate agent avatars
        for name, entry in settings.get("agent_avatars", {}).items():
            img = entry.get("image", "")
            if img.startswith("data:image"):
                try:
                    header, b64 = img.split(",", 1)
                    ext = ".png"
                    if "jpeg" in header or "jpg" in header:
                        ext = ".jpg"
                    elif "webp" in header:
                        ext = ".webp"
                    raw = _b64.b64decode(b64)
                    fname = f"avatar_{name}_{uuid.uuid4().hex[:8]}{ext}"
                    with open(_UPLOADS_DIR / fname, "wb") as f:
                        f.write(raw)
                    entry["image"] = f"/uploads/{fname}"
                    changed = True
                except Exception:
                    pass
        # Migrate user profile avatar
        up = settings.get("user_profile", {})
        img = up.get("image", "")
        if img.startswith("data:image"):
            try:
                header, b64 = img.split(",", 1)
                ext = ".png"
                if "jpeg" in header or "jpg" in header:
                    ext = ".jpg"
                elif "webp" in header:
                    ext = ".webp"
                raw = _b64.b64decode(b64)
                fname = f"user_avatar_{uuid.uuid4().hex[:8]}{ext}"
                with open(_UPLOADS_DIR / fname, "wb") as f:
                    f.write(raw)
                up["image"] = f"/uploads/{fname}"
                changed = True
            except Exception:
                pass
        if changed:
            SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2)
    except Exception:
        pass

_migrate_base64_avatars()

# ── FAISS memory (per-user instances) ────────────────────────────
_faiss_memory = None
_vault_store = None
_user_vault_stores: dict[str, object] = {}
_user_faiss_memories: dict[str, object] = {}

def _get_vault_store(user_id: str | None = None):
    """Return a VaultStore instance scoped to the given user."""
    uid = user_id or _current_user_id.get("__local__")
    if uid and uid != "__local__":
        if uid not in _user_vault_stores:
            from src.memory.vault import VaultStore
            _user_vault_stores[uid] = VaultStore(str(user_vault_path(uid)))
        return _user_vault_stores[uid]
    # Fallback: global vault for local dev
    global _vault_store
    if _vault_store is None:
        from src.memory.vault import VaultStore
        _vault_store = VaultStore(str(_VAULT_PATH))
    return _vault_store

def _get_faiss_memory(user_id: str | None = None):
    uid = user_id or _current_user_id.get("__local__")
    if uid and uid != "__local__":
        if uid not in _user_faiss_memories:
            try:
                from src.memory.faiss_memory import FAISSMemory
                _user_faiss_memories[uid] = FAISSMemory(
                    vault_path=str(user_vault_path(uid)),
                    faiss_dir=str(user_faiss_dir(uid)),
                )
                log.info("[vault] FAISSMemory loaded for user %s — %d memories",
                         uid[:8], len(_user_faiss_memories[uid].list_all()))
            except Exception as exc:
                log.warning("[vault] FAISSMemory unavailable for user %s (%s) — using VaultStore fallback", uid[:8], exc)
                return None
        return _user_faiss_memories[uid]
    # Fallback: global FAISS for local dev
    global _faiss_memory
    if _faiss_memory is None:
        try:
            from src.memory.faiss_memory import FAISSMemory
            _faiss_memory = FAISSMemory(
                vault_path=str(_VAULT_PATH),
                faiss_dir=str(_FAISS_DIR),
            )
            log.info("[vault] FAISSMemory loaded — %d memories", len(_faiss_memory.list_all()))
        except Exception as exc:
            log.warning("[vault] FAISSMemory unavailable (%s) — using VaultStore fallback", exc)
    return _faiss_memory


# ═══════════════════════════════════════════════════════════════════
#  TTL CACHE — eliminates redundant disk reads within the same request
# ═══════════════════════════════════════════════════════════════════
_TTL_CACHE: dict[str, tuple[float, object]] = {}   # key → (expires_at, data)
_TTL_SECONDS = 30.0                                 # covers repeated navigations; writes call _cache_invalidate

def _cache_get(key: str):
    entry = _TTL_CACHE.get(key)
    if entry and entry[0] > _time.monotonic():
        return entry[1]
    return None

def _cache_set(key: str, value):
    _TTL_CACHE[key] = (_time.monotonic() + _TTL_SECONDS, value)

def _cache_invalidate(*prefixes: str):
    """Drop any cache keys that start with one of the given prefixes."""
    for k in [k for k in _TTL_CACHE if any(k.startswith(p) for p in prefixes)]:
        _TTL_CACHE.pop(k, None)


# ═══════════════════════════════════════════════════════════════════
#  JSON HELPERS
# ═══════════════════════════════════════════════════════════════════

def _read_json(path: Path, default=None):
    if not path.exists():
        return default if default is not None else {}
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        log.warning("[io] Corrupt JSON in %s — returning default", path)
        return default if default is not None else {}

def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ═══════════════════════════════════════════════════════════════════
#  CONNECTIONS
# ═══════════════════════════════════════════════════════════════════

def _load_connections() -> dict:
    cached = _cache_get("connections")
    if cached is not None:
        return cached
    data = _read_json(CONNECTIONS_FILE, {"connections": [], "agent_connections": {}})
    _cache_set("connections", data)
    return data

def _save_connections(data: dict):
    _write_json(CONNECTIONS_FILE, data)
    _cache_invalidate("connections")

PLATFORM_OLLAMA_URL = "http://orionforge-engine-ollama.flycast:11434"

# ── Env-var → provider mapping for platform-hosted keys ──────────
# Only OpenRouter (LLM gateway) and ElevenLabs (TTS) are platform-hosted.
# All other providers (OpenAI, Anthropic, Google, DeepSeek, Ollama) are
# user-BYOK only — configurable in Settings → API Keys.
_ENV_KEY_MAP = {
    "openrouter":   ("OPENROUTER_API_KEY",   "https://openrouter.ai/api/v1"),
    "elevenlabs":   ("ELEVENLABS_API_KEY",   "https://api.elevenlabs.io"),
}

def _seed_platform_keys_from_env():
    """Populate connections.json with platform keys found in env vars.

    Called once at startup so Fly.io secrets automatically create / update
    the platform-hosted connections without requiring the admin UI.
    """
    store = _load_connections()
    changed = False
    for provider, (env_var, default_url) in _ENV_KEY_MAP.items():
        key = os.environ.get(env_var, "").strip()
        if not key:
            continue
        # Find existing platform connection for this provider
        existing = None
        for c in store["connections"]:
            if c.get("provider") == provider and (
                c.get("platform_hosted") or str(c.get("id", "")).startswith("platform_")
            ):
                existing = c
                break
        if existing:
            needs_update = existing.get("api_key") != key
            # Fix base URL if it was stored with a trailing /v1 path
            if existing.get("url", "").rstrip("/") != default_url.rstrip("/"):
                existing["url"] = default_url
                needs_update = True
            if needs_update:
                existing["api_key"] = key
                existing["enabled"] = True
                existing["platform_hosted"] = True
                changed = True
        else:
            store["connections"].append({
                "id": f"platform_{provider}",
                "name": f"Platform — {provider.replace('_', ' ').title()}",
                "type": "external",
                "provider": provider,
                "url": default_url,
                "api_key": key,
                "models": [],
                "enabled": True,
                "platform_hosted": True,
            })
            changed = True
    if changed:
        _save_connections(store)
        log.info("[startup] Seeded %d platform keys from environment",
                 sum(1 for p, (e, _) in _ENV_KEY_MAP.items() if os.environ.get(e, "").strip()))


def _normalize_ollama_url(url: str | None) -> str:
    raw = (url or "").strip()
    if not raw:
        return PLATFORM_OLLAMA_URL
    lowered = raw.lower()
    if "localhost" in lowered or "127.0.0.1" in lowered:
        return PLATFORM_OLLAMA_URL
    if "orionforge-engine-ollama.flycast" in lowered:
        return PLATFORM_OLLAMA_URL
    return raw.rstrip("/")

def _resolve_connection(connection_id: str | None, agent: str) -> dict | None:
    """Pick the best API connection for a request.

    Priority:
      1. __userkey_<provider> — dynamic connection built from user's API keys in settings
      2. Explicit connection_id (if provided and enabled)
      3. Agent-mapped connection
      4. First enabled user connection (type=external, not platform_hosted)
      5. First enabled platform-hosted connection (fallback)
    """
    # Handle dynamic user-key connections (e.g. __userkey_openai, __userkey_deepseek)
    if connection_id and connection_id.startswith("__userkey_"):
        provider = connection_id[len("__userkey_"):]
        # Per-user encrypted key vault first, fall back to settings.json
        uid = _current_user_id.get("__local__")
        keys = load_user_keys(uid, "api_keys") if uid and uid != "__local__" else {}
        if not keys:
            keys = _load_settings().get("api_keys", {})
        api_key = keys.get(provider, "")
        base_url = _USER_PROVIDER_URLS.get(provider, "")
        if api_key and base_url:
            catalog_entry = _USER_MODEL_CATALOG.get(provider, (provider.title(), provider, None))
            display_name, section, special = catalog_entry
            pricing = _load_pricing()
            models = _openrouter_models(pricing) if special == "aggregate" else _models_for_provider(pricing, section)
            return {
                "id": connection_id,
                "name": f"User — {display_name}",
                "type": "external",
                "provider": provider,
                "url": base_url,
                "api_key": api_key,
                "models": models,
                "enabled": True,
            }
        return None

    store = _load_connections()
    conns = store.get("connections", [])
    agent_map = store.get("agent_connections", {})

    if connection_id:
        return next((c for c in conns if c["id"] == connection_id and c.get("enabled")), None)
    mapped_id = agent_map.get(agent)
    if mapped_id:
        found = next((c for c in conns if c["id"] == mapped_id and c.get("enabled")), None)
        if found:
            return found
    # Prefer user's own keys (non-platform)
    user_conn = next((c for c in conns if c.get("enabled") and c.get("type") == "external" and not c.get("platform_hosted")), None)
    if user_conn:
        return user_conn
    # Fall back to platform-hosted connection
    return next((c for c in conns if c.get("enabled") and c.get("platform_hosted")), None)


def _prepare_tools_for_connection(tool_defs: list[dict], conn: dict) -> list[dict]:
    """Swap the SearXNG web_search function tool for OpenRouter's native
    server tool when the active connection is OpenRouter.

    OpenRouter's ``openrouter:web_search`` runs server-side — the model
    decides when to search and OpenRouter handles execution transparently.
    This removes the need for a self-hosted SearXNG instance.

    For non-OpenRouter connections the original tool_defs are returned
    unchanged (SearXNG fallback).
    """
    if conn.get("provider") != "openrouter":
        return tool_defs

    # Build excluded_domains from the user's web_search config
    excluded: list[str] = []
    try:
        from src.tools.web_search import get_effective_config
        cfg = get_effective_config()
        raw = cfg.get("ignored_sites", "")
        excluded = [s.strip() for s in raw.split(",") if s.strip()]
    except Exception:
        pass

    # Remove the function-based web_search def (avoid duplicate search tools)
    filtered = [d for d in tool_defs
                if not (d.get("type") == "function"
                        and d.get("function", {}).get("name") == "web_search")]

    # Inject OpenRouter's native server tool
    or_search: dict = {
        "type": "openrouter:web_search",
        "parameters": {
            "max_results": 5,
            "search_context_size": "medium",
        },
    }
    if excluded:
        or_search["parameters"]["excluded_domains"] = excluded
    filtered.append(or_search)

    log.info("[tools] Swapped web_search → openrouter:web_search (%d excluded domains)", len(excluded))
    return filtered


# ═══════════════════════════════════════════════════════════════════
#  SETTINGS
# ═══════════════════════════════════════════════════════════════════

def _get_user_id(request: Request | None = None) -> str:
    """Return the current user id from request context or contextvars."""
    if request is not None:
        user = getattr(request.state, "user", None)
        if user:
            return user.get("id", "__local__") or "__local__"
    return _current_user_id.get("__local__")

def _load_settings(user_id: str | None = None) -> dict:
    uid = user_id or _current_user_id.get("__local__")
    cache_key = f"settings:{uid}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    # Per-user settings file
    if uid and uid != "__local__":
        path = user_settings_path(uid)
        if path.exists():
            data = _read_json(path, {})
            data = decrypt_settings_secrets(data)
            _cache_set(cache_key, data)
            return data
    # Fallback to global settings (for backward compat / local dev)
    data = _read_json(SETTINGS_FILE, {})
    data = decrypt_settings_secrets(data)
    _cache_set(cache_key, data)
    return data

def _save_settings(data: dict, user_id: str | None = None):
    uid = user_id or _current_user_id.get("__local__")
    if uid and uid != "__local__":
        path = user_settings_path(uid)
        _write_json(path, encrypt_settings_secrets(data))
    else:
        _write_json(SETTINGS_FILE, encrypt_settings_secrets(data))
    _cache_invalidate(f"settings:{uid}", "settings:")


# ═══════════════════════════════════════════════════════════════════
#  PRICING
# ═══════════════════════════════════════════════════════════════════

def _load_pricing() -> dict:
    hit = _cache_get("pricing")
    if hit is not None:
        return hit
    if not PRICING_FILE.exists():
        return {}
    try:
        with open(PRICING_FILE, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        _cache_set("pricing", data)
        return data
    except yaml.YAMLError as exc:
        log.warning("[pricing] Failed to parse %s: %s — returning empty dict", PRICING_FILE, exc)
        return {}

def _save_pricing(data: dict):
    _cache_invalidate("pricing")
    PRICING_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Dump to string first, validate round-trip, then write
    raw = yaml.dump(data, default_flow_style=False, sort_keys=False,
                    allow_unicode=True)
    # Validate round-trip before overwriting file
    try:
        yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        log.error("[pricing] yaml.dump produced invalid YAML — aborting save: %s", exc)
        return
    with open(PRICING_FILE, "w", encoding="utf-8") as f:
        f.write(raw)


# ── OpenRouter pricing auto-sync ────────────────────────────────
_OR_PRICING_LAST_SYNC: float = 0.0       # epoch timestamp of last sync
_OR_PRICING_SYNC_INTERVAL = 6 * 3600     # re-sync every 6 hours

async def _sync_openrouter_pricing(force: bool = False) -> dict:
    """Fetch live per-model pricing from OpenRouter and merge into pricing.yaml.

    OpenRouter returns per-token rates in `pricing.prompt` / `pricing.completion`.
    We convert to per-1M-token rates and store under the 'openrouter' provider key.
    Returns {"synced": N, "total_models": M} on success.
    """
    import time
    global _OR_PRICING_LAST_SYNC

    if not force and (time.time() - _OR_PRICING_LAST_SYNC) < _OR_PRICING_SYNC_INTERVAL:
        return {"synced": 0, "skipped": True, "reason": "within sync interval"}

    # Get OpenRouter API key from platform connection or env
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        store = _load_connections()
        for c in store.get("connections", []):
            if c.get("provider") == "openrouter" and c.get("platform_hosted") and c.get("api_key"):
                api_key = c["api_key"]
                break
    if not api_key:
        return {"synced": 0, "error": "No OpenRouter API key available"}

    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get("https://openrouter.ai/api/v1/models", headers=headers)
            resp.raise_for_status()
            models_data = resp.json().get("data", [])
    except Exception as exc:
        log.warning("[openrouter-sync] Failed to fetch models: %s", exc)
        return {"synced": 0, "error": str(exc)}

    pricing = _load_pricing()
    if not pricing:
        # File was corrupt or missing — reload from image default
        log.warning("[openrouter-sync] pricing.yaml returned empty, skipping sync to avoid data loss")
        return {"synced": 0, "error": "pricing.yaml empty or corrupt — sync skipped"}
    or_section = pricing.get("openrouter", {})

    # Preserve existing _default and other meta keys
    synced = 0
    for m in models_data:
        model_id = m.get("id", "")
        if not model_id:
            continue
        pr = m.get("pricing") or {}
        prompt_per_tok = pr.get("prompt")
        completion_per_tok = pr.get("completion")
        if prompt_per_tok is None and completion_per_tok is None:
            continue
        try:
            input_per_1m = float(prompt_per_tok or 0) * 1_000_000
            output_per_1m = float(completion_per_tok or 0) * 1_000_000
        except (ValueError, TypeError):
            continue
        # Only store models with non-zero pricing
        if input_per_1m == 0 and output_per_1m == 0:
            # Free models — store them too so cost displays $0
            pass
        or_section[model_id] = {
            "input_per_1m": round(input_per_1m, 4),
            "cached_input_per_1m": round(input_per_1m * 0.5, 4),  # estimate cached at 50%
            "output_per_1m": round(output_per_1m, 4),
            "training_per_1m": 0.0,
        }
        synced += 1

    # Ensure _default exists
    if "_default" not in or_section:
        or_section["_default"] = {
            "input_per_1m": 2.50,
            "cached_input_per_1m": 1.25,
            "output_per_1m": 10.00,
            "training_per_1m": 0.0,
        }

    pricing["openrouter"] = or_section
    _save_pricing(pricing)

    # Clear metering cache so new prices take effect immediately
    try:
        from src.observability.metering import reset_pricing_cache
        reset_pricing_cache()
    except Exception:
        pass

    _OR_PRICING_LAST_SYNC = time.time()
    log.info("[openrouter-sync] Synced pricing for %d models (total %d in catalog)",
             synced, len(models_data))
    return {"synced": synced, "total_models": len(models_data)}


# ═══════════════════════════════════════════════════════════════════
#  PROFILES
# ═══════════════════════════════════════════════════════════════════

def _list_agents(user_id: str | None = None) -> list[str]:
    uid = user_id or _current_user_id.get("__local__")
    agents = set(p.stem for p in _PROFILES_DIR.glob("*.yaml"))
    # Include per-user custom agents
    if uid and uid != "__local__":
        agents |= set(p.stem for p in user_profiles_dir(uid).glob("*.yaml"))
    return sorted(agents)

# Private agents restricted to specific user IDs. Even though their profile
# files live in the shared profiles dir, they stay hidden from everyone except
# the listed UIDs. "__local__" (single-user local mode) always retains access.
RESTRICTED_AGENTS: dict[str, set[str]] = {
    "elysia_cannon": {"a2cddfbd-dabb-4d50-b515-b600a08ce8e8"},
    "orion_cannon": {"a2cddfbd-dabb-4d50-b515-b600a08ce8e8"},
}


def _can_access_agent(agent: str, user_id: str | None) -> bool:
    """Whether `agent` is visible to `user_id`.

    Non-restricted agents are always allowed here (their visibility is still
    governed by the store/unlock rules in _list_unlocked_agents).
    """
    allowed = RESTRICTED_AGENTS.get(agent)
    if allowed is None:
        return True
    if user_id == "__local__":
        return True
    return bool(user_id) and user_id in allowed


def _list_unlocked_agents(request: Request) -> list[str]:
    """Return agents the current user can access.

    Admins see everything. Regular users see:
    - Free agents (codex_animus)
    - Purchased store agents
    - User-created agents (not in the store catalog)

    Restricted agents (RESTRICTED_AGENTS) are filtered out for everyone except
    their allowed UIDs, regardless of admin status. The list is then ordered by
    _prioritize_default_agent so the pinned agents (K-OS first) lead.
    """
    all_agents = _list_agents()
    user = getattr(request.state, "user", None)
    user_id = (user.get("id", "") if user else "") or _current_user_id.get("__local__")

    def _finalize(agents: list[str]) -> list[str]:
        return _prioritize_default_agent(
            [a for a in agents if _can_access_agent(a, user_id)]
        )

    if _check_admin(request):
        return _finalize(all_agents)
    if not user:
        return _finalize(all_agents)
    store_ids = get_store_agent_ids()
    unlocked = get_user_unlocked_agents(user_id)
    agents = [
        a for a in all_agents
        if a not in store_ids  # user-created agents (not in store)
        or a in unlocked       # free or purchased store agents
    ]
    return _finalize(agents)


# Agents pinned to the top of the chat dropdown, in this exact order. Any pinned
# agent the current user can't access is simply skipped; every other agent keeps
# its existing (alphabetical) order after the pinned block.
PINNED_AGENT_ORDER = [
    "codex_animus", "k_os", "elysia", "orion", "marcus",
    "aristotle", "janus", "dalvarr", "lux_umbra",
]


def _prioritize_default_agent(agents: list[str]) -> list[str]:
    """Order agents for the dropdown.

    Pinned agents (PINNED_AGENT_ORDER) are placed first, in that exact order, so
    K-OS is the default selection. Remaining agents keep their existing order.
    """
    present = set(agents)
    pinned = [a for a in PINNED_AGENT_ORDER if a in present]
    rest = [a for a in agents if a not in pinned]
    return pinned + rest

def _load_profile(name: str, user_id: str | None = None) -> dict:
    uid = user_id or _current_user_id.get("__local__")
    key = f"profile:{uid}:{name}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    # Check per-user override first
    if uid and uid != "__local__":
        user_path = user_profiles_dir(uid) / f"{name}.yaml"
        if user_path.exists():
            with open(user_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            _cache_set(key, data)
            return data
    # Fall back to global template
    path = _PROFILES_DIR / f"{name}.yaml"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    _cache_set(key, data)
    return data

def _save_profile(name: str, data: dict, user_id: str | None = None):
    uid = user_id or _current_user_id.get("__local__")
    if uid and uid != "__local__":
        dest = user_profiles_dir(uid) / f"{name}.yaml"
    else:
        dest = _PROFILES_DIR / f"{name}.yaml"
    with open(dest, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    _cache_invalidate(f"profile:{uid}:{name}")

def _load_system_prompt(name: str, user_id: str | None = None) -> str:
    uid = user_id or _current_user_id.get("__local__")
    # Check per-user override first
    if uid and uid != "__local__":
        user_path = user_prompts_dir(uid) / f"{name}.system.md"
        if user_path.exists():
            return user_path.read_text(encoding="utf-8")
    # Fall back to global template
    path = _PROMPTS_DIR / f"{name}.system.md"
    return path.read_text(encoding="utf-8") if path.exists() else ""

def _save_system_prompt(name: str, text: str, user_id: str | None = None):
    uid = user_id or _current_user_id.get("__local__")
    if uid and uid != "__local__":
        dest = user_prompts_dir(uid) / f"{name}.system.md"
    else:
        dest = _PROMPTS_DIR / f"{name}.system.md"
    dest.write_text(text, encoding="utf-8")

def _load_soul_script(name: str, user_id: str | None = None) -> str:
    """Load the soul script (directive) text for an agent."""
    uid = user_id or _current_user_id.get("__local__")
    # Check per-user override first
    if uid and uid != "__local__":
        user_path = user_directives_dir(uid) / f"{name}.md"
        if user_path.exists():
            return user_path.read_text(encoding="utf-8")
    # Fall back to global template
    path = _DIRECTIVES_DIR / f"{name}.md"
    return path.read_text(encoding="utf-8") if path.exists() else ""

def _save_soul_script(name: str, text: str, user_id: str | None = None):
    """Save the soul script (directive) text for an agent."""
    uid = user_id or _current_user_id.get("__local__")
    if uid and uid != "__local__":
        dest_dir = user_directives_dir(uid)
    else:
        dest_dir = _DIRECTIVES_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / f"{name}.md").write_text(text, encoding="utf-8")
    # Re-index soul scripts into NotesFAISS so they are searchable at chat time
    try:
        _rebuild_notes_faiss()
        from src.storage.note_collector import invalidate_notes_faiss
        invalidate_notes_faiss()
    except Exception as exc:
        log.warning("[soul_script] FAISS reindex after soul script save failed: %s", exc)

def _get_agent_config(agent: str, user_id: str | None = None) -> dict:
    return _load_settings(user_id=user_id).get("agent_configs", {}).get(agent, {})

def _save_agent_config(agent: str, cfg: dict, user_id: str | None = None):
    uid = user_id or _current_user_id.get("__local__")
    settings = _load_settings(user_id=uid)
    settings.setdefault("agent_configs", {})[agent] = cfg
    _save_settings(settings, user_id=uid)


# ═══════════════════════════════════════════════════════════════════
#  AGENT TRASH (soft-delete with 30-day retention)
# ═══════════════════════════════════════════════════════════════════

_TRASH_INDEX_FILE = _TRASH_DIR / "index.json"
_TRASH_RETENTION_DAYS = 30


def _load_trash_index() -> list[dict]:
    return _read_json(_TRASH_INDEX_FILE, [])


def _save_trash_index(data: list[dict]):
    _write_json(_TRASH_INDEX_FILE, data)


def _trash_agent(name: str):
    """Soft-delete an agent by moving files to the trash directory."""
    import shutil
    now = datetime.now(timezone.utc).isoformat()
    trash_id = f"{name}_{uuid.uuid4().hex[:8]}"
    agent_trash_dir = _TRASH_DIR / trash_id
    agent_trash_dir.mkdir(parents=True, exist_ok=True)

    # Move profile yaml
    profile_src = _PROFILES_DIR / f"{name}.yaml"
    if profile_src.exists():
        shutil.move(str(profile_src), str(agent_trash_dir / f"{name}.yaml"))

    # Move system prompt
    prompt_src = _PROMPTS_DIR / f"{name}.system.md"
    if prompt_src.exists():
        shutil.move(str(prompt_src), str(agent_trash_dir / f"{name}.system.md"))

    # Save agent config snapshot so we can restore it
    settings = _load_settings()
    agent_cfg = settings.get("agent_configs", {}).pop(name, {})
    agent_avatar = settings.get("agent_avatars", {}).pop(name, {})
    _save_settings(settings)

    # Write config snapshot into trash folder
    _write_json(agent_trash_dir / "config_snapshot.json", {
        "agent_config": agent_cfg,
        "agent_avatar": agent_avatar,
    })

    # Update trash index
    idx = _load_trash_index()
    idx.append({
        "id": trash_id,
        "name": name,
        "display_name": agent_cfg.get("display_name", name),
        "deleted_at": now,
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=_TRASH_RETENTION_DAYS)).isoformat(),
    })
    _save_trash_index(idx)
    log.info("[trash] Agent '%s' moved to trash as '%s'", name, trash_id)


def _restore_agent(trash_id: str) -> str | None:
    """Restore an agent from trash. Returns the agent name or None on failure."""
    import shutil
    idx = _load_trash_index()
    entry = next((e for e in idx if e["id"] == trash_id), None)
    if not entry:
        return None

    agent_trash_dir = _TRASH_DIR / trash_id
    if not agent_trash_dir.exists():
        # Clean up orphan index entry
        idx = [e for e in idx if e["id"] != trash_id]
        _save_trash_index(idx)
        return None

    name = entry["name"]

    # Check name collision — if a new agent with the same name exists, bail
    if (_PROFILES_DIR / f"{name}.yaml").exists():
        return None

    # Move files back
    profile_file = agent_trash_dir / f"{name}.yaml"
    if profile_file.exists():
        shutil.move(str(profile_file), str(_PROFILES_DIR / f"{name}.yaml"))

    prompt_file = agent_trash_dir / f"{name}.system.md"
    if prompt_file.exists():
        shutil.move(str(prompt_file), str(_PROMPTS_DIR / f"{name}.system.md"))

    # Restore config & avatar
    snapshot_file = agent_trash_dir / "config_snapshot.json"
    if snapshot_file.exists():
        snapshot = _read_json(snapshot_file, {})
        settings = _load_settings()
        if snapshot.get("agent_config"):
            settings.setdefault("agent_configs", {})[name] = snapshot["agent_config"]
        if snapshot.get("agent_avatar"):
            settings.setdefault("agent_avatars", {})[name] = snapshot["agent_avatar"]
        _save_settings(settings)

    # Clean up trash folder
    shutil.rmtree(str(agent_trash_dir), ignore_errors=True)

    # Remove from index
    idx = [e for e in idx if e["id"] != trash_id]
    _save_trash_index(idx)
    log.info("[trash] Agent '%s' restored from trash", name)
    return name


def _permanently_delete_from_trash(trash_id: str):
    """Permanently remove a trashed agent."""
    import shutil
    agent_trash_dir = _TRASH_DIR / trash_id
    if agent_trash_dir.exists():
        shutil.rmtree(str(agent_trash_dir), ignore_errors=True)
    idx = _load_trash_index()
    idx = [e for e in idx if e["id"] != trash_id]
    _save_trash_index(idx)
    log.info("[trash] Permanently deleted trash entry '%s'", trash_id)


def _purge_expired_trash():
    """Remove any trash entries older than the retention period."""
    idx = _load_trash_index()
    now = datetime.now(timezone.utc)
    to_purge = []
    remaining = []
    for entry in idx:
        try:
            expires = datetime.fromisoformat(entry["expires_at"])
            if now >= expires:
                to_purge.append(entry)
            else:
                remaining.append(entry)
        except (KeyError, ValueError):
            to_purge.append(entry)
    if to_purge:
        for entry in to_purge:
            _permanently_delete_from_trash(entry["id"])
        log.info("[trash] Purged %d expired agent(s) from trash", len(to_purge))


# Run purge on startup
_purge_expired_trash()


# ═══════════════════════════════════════════════════════════════════
#  CHATS
# ═══════════════════════════════════════════════════════════════════

def _load_chat_index(user_id: str | None = None) -> dict:
    uid = user_id or _current_user_id.get("__local__")
    if uid and uid != "__local__":
        return _read_json(user_chats_dir(uid) / "index.json", {"folders": [], "chats": []})
    return _read_json(_CHATS_DIR / "index.json", {"folders": [], "chats": []})

def _save_chat_index(data: dict, user_id: str | None = None):
    uid = user_id or _current_user_id.get("__local__")
    if uid and uid != "__local__":
        _write_json(user_chats_dir(uid) / "index.json", data)
    else:
        _write_json(_CHATS_DIR / "index.json", data)

def _load_chat(chat_id: str, user_id: str | None = None) -> dict | None:
    uid = user_id or _current_user_id.get("__local__")
    if uid and uid != "__local__":
        path = user_chats_dir(uid) / f"{chat_id}.json"
    else:
        path = _CHATS_DIR / f"{chat_id}.json"
    return _read_json(path) if path.exists() else None

def _save_chat(chat_id: str, data: dict, user_id: str | None = None):
    uid = user_id or _current_user_id.get("__local__")
    if uid and uid != "__local__":
        _write_json(user_chats_dir(uid) / f"{chat_id}.json", data)
    else:
        _write_json(_CHATS_DIR / f"{chat_id}.json", data)

def _create_new_chat(agent: str, mode: str = "chat", meta: dict | None = None, user_id: str | None = None) -> dict:
    uid = user_id or _current_user_id.get("__local__")
    chat_id = str(uuid.uuid4())[:8]
    now = datetime.now(timezone.utc).isoformat()
    chat_data = {
        "id": chat_id, "title": "New Chat", "folder_id": None,
        "agent": agent, "mode": mode,
        "created": now, "updated": now, "messages": [],
    }
    if meta:
        chat_data.update(meta)
    _save_chat(chat_id, chat_data, user_id=uid)
    idx = _load_chat_index(user_id=uid)
    idx["chats"].append({
        "id": chat_id, "title": "New Chat", "folder_id": None,
        "agent": agent, "mode": mode, "created": now, "updated": now,
    })
    _save_chat_index(idx, user_id=uid)
    return chat_data


# ── In-memory burst session store ────────────────────────────────
_chat_sessions: dict[str, dict] = {}


def _update_chat_index_entry(chat_id: str, updates: dict, user_id: str | None = None):
    """Update a chat entry in the index."""
    uid = user_id or _current_user_id.get("__local__")
    idx = _load_chat_index(user_id=uid)
    for c in idx["chats"]:
        if c["id"] == chat_id:
            c.update(updates)
            break
    _save_chat_index(idx, user_id=uid)


def _append_agent_line_to_chat(chat_id: str, line: str):
    """Append an agent output line to a persistent chat (called from reader thread)."""
    try:
        role = "agent"
        parsed = None
        if line.startswith("[step] "):
            role = "step"
            try: parsed = json.loads(line[7:])
            except Exception: pass
        elif line.startswith("[tool-result] "):
            role = "tool-result"
            try: parsed = json.loads(line[14:])
            except Exception: pass
        elif line.startswith("[memory-flush] "):
            role = "memory-flush"
            try: parsed = json.loads(line[15:])
            except Exception: pass
        elif line.startswith("[tick-start] "):
            role = "tick-start"
            try: parsed = json.loads(line[13:])
            except Exception: pass
        elif line.startswith("[tick-end] "):
            role = "tick-end"
            try: parsed = json.loads(line[10:])
            except Exception: pass
        elif line.startswith("[burst] Done") or line.startswith("[burst] Starting"):
            role = "system"
        elif "ERROR" in line or "error" in line.lower():
            role = "error"
        msg = {"role": role, "text": line, "time": datetime.now(timezone.utc).isoformat()}
        if parsed:
            msg["data"] = parsed
        chat_data = _load_chat(chat_id)
        if chat_data:
            chat_data["messages"].append(msg)
            chat_data["updated"] = datetime.now(timezone.utc).isoformat()
            _save_chat(chat_id, chat_data)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════
#  KNOWLEDGE (Notes) — with folder organisation
# ═══════════════════════════════════════════════════════════════════

_FOLDERS_FILE = _NOTES_DIR / "folders.json"

# Pinned default folder — guaranteed to always exist, cannot be deleted.
_SOUL_SCRIPTS_FOLDER_ID = "soul_scripts"

_DEFAULT_FOLDERS: list[dict] = [
    {"id": _SOUL_SCRIPTS_FOLDER_ID, "name": "Soul Scripts", "emoji": "🧬",
     "pinned": True, "color": "#818cf8"},
]


def _load_folders(user_id: str | None = None) -> list[dict]:
    """Load knowledge folders from disk, seeding defaults if missing."""
    uid = user_id or _current_user_id.get("__local__")
    if uid and uid != "__local__":
        fpath = user_folders_file(uid)
    else:
        fpath = _FOLDERS_FILE
    folders = _read_json(fpath, [])
    # Ensure pinned defaults always present
    existing_ids = {f["id"] for f in folders}
    changed = False
    for default in _DEFAULT_FOLDERS:
        if default["id"] not in existing_ids:
            folders.insert(0, dict(default))
            changed = True
    # Ensure pinned folders are always first
    if changed:
        pinned = [f for f in folders if f.get("pinned")]
        regular = [f for f in folders if not f.get("pinned")]
        folders = pinned + regular
        _save_folders(folders, user_id=uid)
    return folders


def _save_folders(data: list[dict], user_id: str | None = None):
    uid = user_id or _current_user_id.get("__local__")
    if uid and uid != "__local__":
        _write_json(user_folders_file(uid), data)
    else:
        _write_json(_FOLDERS_FILE, data)


def _load_notes_index(user_id: str | None = None) -> list[dict]:
    uid = user_id or _current_user_id.get("__local__")
    if uid and uid != "__local__":
        return _read_json(user_notes_dir(uid) / "index.json", [])
    return _read_json(_NOTES_DIR / "index.json", [])

def _save_notes_index(data: list[dict], user_id: str | None = None):
    uid = user_id or _current_user_id.get("__local__")
    if uid and uid != "__local__":
        _write_json(user_notes_dir(uid) / "index.json", data)
    else:
        _write_json(_NOTES_DIR / "index.json", data)

def _load_note(note_id: str, user_id: str | None = None) -> dict | None:
    uid = user_id or _current_user_id.get("__local__")
    if uid and uid != "__local__":
        path = user_notes_dir(uid) / f"{note_id}.json"
    else:
        path = _NOTES_DIR / f"{note_id}.json"
    return _read_json(path) if path.exists() else None

def _save_note(note_id: str, data: dict, user_id: str | None = None):
    uid = user_id or _current_user_id.get("__local__")
    if uid and uid != "__local__":
        _write_json(user_notes_dir(uid) / f"{note_id}.json", data)
    else:
        _write_json(_NOTES_DIR / f"{note_id}.json", data)


# Seed folders on import so the file exists on first run
_load_folders()


def _migrate_legacy_sections():
    """One-time migration: convert old free-text section names to folder IDs.

    Previously, notes had free-text like "Soul Scripts", "Knowledge", etc.
    Now `section` stores a folder ID.  This creates matching folders for
    any legacy text values and updates notes to use the folder IDs.
    """
    _MIGRATION_MARKER = _NOTES_DIR / ".folders_migrated"
    if _MIGRATION_MARKER.exists():
        return  # already migrated

    folders = _load_folders()
    folder_names = {f["name"].lower(): f["id"] for f in folders}
    # Also map the pinned folder names
    idx = _load_notes_index()
    changed = False

    for entry in idx:
        section = entry.get("section", "Uncategorized")
        # If it's already a valid folder ID, skip
        if section in {f["id"] for f in folders} or section == "Uncategorized":
            continue
        # Map legacy name → folder ID, creating folder if needed
        key = section.lower()
        if key in folder_names:
            fid = folder_names[key]
        else:
            # Create a new folder for this legacy section name
            fid = str(uuid.uuid4())[:8]
            folders.append({"id": fid, "name": section, "emoji": "📁",
                            "pinned": False, "color": "#6366f1"})
            folder_names[key] = fid
        entry["section"] = fid
        # Also update the individual note file
        note = _load_note(entry["id"])
        if note:
            note["section"] = fid
            _save_note(entry["id"], note)
        changed = True

    if changed:
        _save_notes_index(idx)
        _save_folders(folders)

    _MIGRATION_MARKER.write_text("migrated", encoding="utf-8")


_migrate_legacy_sections()


# ═══════════════════════════════════════════════════════════════════
#  PROMPT ASSEMBLY — The core of SoulScript identity persistence
# ═══════════════════════════════════════════════════════════════════

def _build_chat_messages(agent: str, messages: list[dict]) -> tuple[list[dict], dict, list[dict]]:
    """Assemble the full prompt with identity + knowledge + memory layers.

    Returns (llm_messages, layer_metadata, tool_defs) where:
      - llm_messages: the message array for the LLM
      - layer_metadata: what was injected at each stage (Prompt Inspector UI)
      - tool_defs: OpenAI-format tool definitions for the ``tools`` API param

    Injection order (highest priority first):
      1. Base system prompt    — the agent's personality / instructions
      2. Soul Script retrieval — FAISS search over directive-mode knowledge
      3. Always-on knowledge   — verbatim attached knowledge notes
      4. Memory Vault context  — FAISS search over persistent agent memories
      5. Memory Save Protocol  — hardcoded [MEMORY_SAVE: ...] tag instructions
      6. Image Generation      — [IMAGE_GEN: ...] tag instructions (if provider set)
      7. Conversation history  — recent user/assistant turns (budget-trimmed)
      8. Tool definitions      — OpenAI function-calling schemas (separate API param)
    """
    layers = {
        "base_prompt": {"chars": 0, "preview": ""},
        "soul_script": {"chunks": 0, "chars": 0, "preview": ""},
        "always_on":   {"chunks": 0, "chars": 0, "preview": ""},
        "vault":       {"memories": 0, "chars": 0, "snippets": []},
        "conversation": {"turns": 0, "chars": 0},
        "tools":       {"count": 0, "names": []},
    }

    # ── 1. Base prompt ──
    system_prompt = _load_system_prompt(agent)

    # ── 1b. User identity — let the agent know who it's talking to ──
    user_name = _load_settings().get("user_profile", {}).get("name", "").strip()
    if user_name:
        system_prompt += f"\n\nThe user's name is {user_name}."

    layers["base_prompt"]["chars"] = len(system_prompt)
    layers["base_prompt"]["preview"] = system_prompt[:300]

    # Get latest user message for semantic search
    latest_user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            latest_user_msg = m.get("text", "")
            break

    # ── 2 & 3. Soul Script + Always-on knowledge ──
    always_block = ""
    directive_block = ""
    try:
        from src.storage.note_collector import collect_notes
        always_block, directive_block = collect_notes(agent, query=latest_user_msg)
        if directive_block:
            system_prompt += "\n\n" + directive_block
            layers["soul_script"]["chunks"] = directive_block.count("---") + 1
            layers["soul_script"]["chars"] = len(directive_block)
            layers["soul_script"]["preview"] = directive_block[:400]
        if always_block:
            system_prompt += "\n\n" + always_block
            layers["always_on"]["chunks"] = always_block.count("---") + 1
            layers["always_on"]["chars"] = len(always_block)
            layers["always_on"]["preview"] = always_block[:400]
    except Exception as exc:
        log.warning("[prompt] Note collection failed: %s", exc)

    # ── 4. Memory Vault context ──
    if latest_user_msg:
        try:
            fm = _get_faiss_memory()
            if fm:
                results = fm.search(latest_user_msg, scope=agent, top_k=5)
                if results:
                    snippets = [r["text"] for r in results if r.get("text")]
                    if snippets:
                        vault_block = "\n\n---\n\n".join(snippets)
                        system_prompt += (
                            "\n\n## Memory Vault Context\n\n"
                            "The following memories were retrieved from your persistent "
                            "memory vault based on relevance to the current conversation:\n\n"
                            + vault_block
                        )
                        layers["vault"]["memories"] = len(snippets)
                        layers["vault"]["chars"] = len(vault_block)
                        layers["vault"]["snippets"] = [s[:150] for s in snippets]
        except Exception as exc:
            log.warning("[prompt] Vault search failed: %s", exc)

    # ── Memory save instruction ──
    system_prompt += (
        "\n\n## Memory Save Protocol\n\n"
        "You have a persistent memory vault. When you want to save something "
        "to memory (because the user asked you to remember it, or because it is "
        "important biographical/preference/project info worth keeping), include "
        "one or more memory-save tags in your response like this:\n\n"
        "```\n[MEMORY_SAVE: category=preference | The user prefers dark mode and minimal UIs]\n```\n\n"
        "Valid categories: bio, preference, project, lore, session, meta, health, self, other.\n"
        "The system will automatically extract these and write them to your vault. "
        "You can include multiple MEMORY_SAVE tags in a single response.\n"
        "Always confirm to the user what you saved.\n"
        "The MEMORY_SAVE tag will be hidden from the user — they only see your natural text."
    )

    # ── Image generation instruction (only if a provider is configured) ──
    img_cfg = _load_settings().get("image", {})
    img_preferred = img_cfg.get("preferred", "none")
    if img_preferred and img_preferred != "none":
        _IMG_MODEL_LABELS = {
            "openai_dalle3": "DALL-E 3", "openai_dalle2": "DALL-E 2",
            "openai_gpt_image": "GPT Image", "google_imagen": "Google Imagen 3",
            "stability_ultra": "Stable Image Ultra", "stability_core": "Stable Image Core",
            "stability_sd3_large": "SD3 Large", "stability_sd3_large_turbo": "SD3 Large Turbo",
            "stability_sd3_medium": "SD3 Medium",
            "ideogram": "Ideogram V2", "ideogram_turbo": "Ideogram V2 Turbo",
            "replicate_flux_pro": "Flux Pro", "replicate_flux_schnell": "Flux Schnell",
            "replicate_flux_dev": "Flux Dev", "replicate_playground": "Playground v2.5",
            "fal_flux_pro": "Flux Pro (FAL)", "fal_flux_schnell": "Flux Schnell (FAL)",
            "fal_flux_dev": "Flux Dev (FAL)",
            "leonardo_diffusion_xl": "Leonardo Diffusion XL",
            "leonardo_lightning_xl": "Leonardo Lightning XL",
            "leonardo_vision_xl": "Leonardo Vision XL",
            "leonardo_kino_xl": "Leonardo Kino XL",
            "midjourney": "Midjourney",
        }
        model_label = _IMG_MODEL_LABELS.get(img_preferred, img_preferred)
        system_prompt += (
            f"\n\n## Image Generation\n\n"
            f"You can generate images using {model_label}. When the user asks you to "
            "create, generate, draw, illustrate, or design an image, include an image "
            "generation tag in your response like this:\n\n"
            "```\n[IMAGE_GEN: A photorealistic sunset over a mountain lake with reflections]\n```\n\n"
            "Write a detailed, descriptive prompt inside the tag — the more detail the better. "
            "You may include exactly ONE [IMAGE_GEN: ...] tag per response. "
            "Place it after your introductory text. The system will generate the image "
            "and display it inline in the chat. The tag itself will be hidden from the user."
        )

    # ── Video generation instruction (only if a provider is configured) ──
    vid_cfg = _load_settings().get("video", {})
    vid_preferred = vid_cfg.get("preferred", "none")
    if vid_preferred and vid_preferred != "none":
        _VID_MODEL_LABELS = {
            "google_veo2":  "Google Veo 2",
            "google_veo3":  "Google Veo 3",
            "google_veo31": "Google Veo 3.1",
        }
        vid_model_label = _VID_MODEL_LABELS.get(vid_preferred, vid_preferred)
        system_prompt += (
            f"\n\n## Video Generation\n\n"
            f"You can generate videos using {vid_model_label}. When the user asks you to "
            "create, generate, make, or produce a video, include a video generation tag "
            "in your response like this:\n\n"
            "```\n[VIDEO_GEN: A cinematic aerial shot of a glowing city at night with light trails]\n```\n\n"
            "Write a rich, detailed prompt inside the tag — describe the subject, action, "
            "style, camera motion, and mood. You may include exactly ONE [VIDEO_GEN: ...] "
            "tag per response. Place it after your introductory text. The system will "
            "generate the video and display it inline in the chat. "
            "The tag itself will be hidden from the user. "
            "Video generation takes 1–3 minutes; let the user know to expect a wait."
        )

    # ── 5. Conversation history (truncated to budget) ──
    MAX_CONTEXT_CHARS = 30_000
    conversation: list[dict] = []
    budget = MAX_CONTEXT_CHARS
    for m in reversed(messages):
        text = m.get("text", "")
        if len(text) > budget:
            break
        budget -= len(text)
        conversation.insert(0, {"role": m["role"], "content": text})

    layers["conversation"]["turns"] = len(conversation)
    layers["conversation"]["chars"] = MAX_CONTEXT_CHARS - budget

    # ── 6. Tool registry ──
    tool_defs: list[dict] = []
    try:
        from src.tools.registry import get_tool_defs_for_agent
        tool_defs = get_tool_defs_for_agent(agent)
        layers["tools"]["count"] = len(tool_defs)
        layers["tools"]["names"] = [d["function"]["name"] for d in tool_defs]
    except Exception as exc:
        log.warning("[prompt] Tool registry failed: %s", exc)

    return [{"role": "system", "content": system_prompt}] + conversation, layers, tool_defs


def _extract_and_save_memories(agent: str, response_text: str) -> list[dict]:
    """Parse [MEMORY_SAVE: ...] tags from response and write to vault.

    Returns list of saved memory summaries (for optional UI feedback).
    """
    import re
    pattern = r'\[MEMORY_SAVE:\s*(?:category=([\w]+)\s*\|)?\s*(.+?)\]'
    matches = re.findall(pattern, response_text, re.DOTALL)
    if not matches:
        return []

    fm = _get_faiss_memory()
    if not fm:
        # Fall back to VaultStore for memory saves
        vs = _get_vault_store()
        if not vs:
            log.warning("[memory] Vault not available — cannot save memories")
            return []

    saved = []
    for category_raw, text_raw in matches:
        category = (category_raw or "other").strip().lower()
        text = text_raw.strip()
        if not text or len(text) < 5:
            continue
        valid_cats = {"bio", "preference", "project", "lore", "session", "meta", "health", "self", "other"}
        if category not in valid_cats:
            category = "other"
        try:
            if fm:
                mem = fm.add(
                    text=text,
                    scope=agent,
                    category=category,
                    source="chat",
                    tags=["auto-saved"],
                )
            else:
                mem = vs.create_memory(
                    text=text,
                    scope=agent,
                    category=category,
                    source="chat",
                    tags=["auto-saved"],
                )
            saved.append({"id": mem.id, "text": text[:120], "category": category})
            log.info("[memory] Saved to vault: scope=%s cat=%s text=%.60s", agent, category, text)
        except Exception as exc:
            log.error("[memory] Failed to save memory: %s", exc)
    return saved


def _strip_memory_tags(text: str) -> str:
    """Remove [MEMORY_SAVE: ...] tags from text shown to user."""
    import re
    return re.sub(r'\[MEMORY_SAVE:\s*(?:category=[\w]+\s*\|)?\s*.+?\]', '', text).strip()


# ═══════════════════════════════════════════════════════════════════
#  AUTH ROUTES
# ═══════════════════════════════════════════════════════════════════

@app.get("/login", response_class=HTMLResponse)
async def page_login(request: Request):
    """Login / signup page."""
    auth_cfg = get_auth_config()
    return templates.TemplateResponse(request, "login.html", {
        "auth_config": auth_cfg,
    })


@app.get("/api/auth/config")
async def api_auth_config():
    """Return public auth config for frontend."""
    cfg = get_auth_config()
    return JSONResponse({
        "supabase_url": cfg.get("supabase_url", ""),
        "supabase_anon_key": cfg.get("supabase_anon_key", ""),
        "auth_enabled": cfg.get("auth_enabled", False),
        "free_tier_features": cfg.get("free_tier_features", []),
        "paid_features": cfg.get("paid_features", []),
    })


@app.post("/api/auth/set-session")
async def api_auth_set_session(request: Request):
    """Store Supabase JWT in httpOnly cookie after frontend auth."""
    try:
        body = await request.json()
        access_token = body.get("access_token", "")
        refresh_token = body.get("refresh_token", "")
        expires_at = body.get("expires_at", 0)

        if not access_token:
            return JSONResponse({"error": "No access token"}, status_code=400)

        # Verify the token is valid
        payload = verify_supabase_token(access_token)
        if not payload:
            return JSONResponse({"error": "Invalid token"}, status_code=401)

        user = extract_user_from_token(payload)

        # Enforce single-user email allowlist before issuing a session cookie.
        if not is_email_allowed(user.get("email", "")):
            log.warning(
                "[auth] set-session denied for non-allowlisted email: %s",
                user.get("email", ""),
            )
            return JSONResponse(
                {"error": "This account is not authorized to sign in."},
                status_code=403,
            )

        response = JSONResponse({"ok": True, "user": user})

        # Set httpOnly cookies scoped to the whole orionforge.chat site so a
        # single login covers soulscript.orionforge.chat and any subdomain.
        _write_session_cookies(request, response, access_token, refresh_token)

        return response
    except Exception as exc:
        log.error("[auth] set-session failed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/auth/session")
async def api_auth_session(request: Request):
    """Check if the current user has a valid session."""
    token = request.cookies.get("sb_access_token")
    if not token:
        return JSONResponse({"authenticated": False})

    payload = verify_supabase_token(token)
    if not payload:
        return JSONResponse({"authenticated": False})

    user = extract_user_from_token(payload)
    return JSONResponse({"authenticated": True, "user": user})


@app.post("/api/auth/logout")
async def api_auth_logout(request: Request):
    """Clear auth cookies to log out."""
    response = JSONResponse({"ok": True})
    _clear_session_cookies(request, response)
    return response


@app.get("/api/auth/user")
async def api_auth_user(request: Request):
    """Get current user info (used by base template)."""
    # /api/auth/user is a PUBLIC path, so AuthMiddleware short-circuits before
    # it can populate request.state.user. Resolve the user straight from the
    # session cookie (like /api/auth/session) so the nav can show the email and
    # reveal the Admin tab; falling back to request.state.user when present.
    user = getattr(request.state, "user", None)
    if not user:
        token = request.cookies.get("sb_access_token")
        payload = verify_supabase_token(token) if token else None
        if payload:
            user = extract_user_from_token(payload)
    if user:
        # is_admin lets the nav reveal the Admin tab based on the authoritative
        # backend check (ADMIN_USER_IDS / ADMIN_EMAILS) rather than hardcoding.
        return JSONResponse({
            "authenticated": True,
            "user": user,
            "is_admin": _user_is_admin(user),
        })
    return JSONResponse({"authenticated": False})


@app.get("/auth/callback", response_class=HTMLResponse)
async def auth_callback(request: Request):
    """OAuth callback page.

    Supabase redirects here with tokens in the URL fragment (#access_token=...).
    Since fragments aren't sent to the server, a small JS page extracts them
    and POSTs to /api/auth/set-session to create the httpOnly cookie.
    """
    auth_cfg = get_auth_config()
    return HTMLResponse(f"""
<!DOCTYPE html>
<html><head><title>Signing in…</title>
<script src="https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2"></script>
<style>body{{background:#0f0f14;color:#e4e4e7;display:flex;align-items:center;justify-content:center;min-height:100vh;font-family:system-ui,sans-serif}}.box{{text-align:center}}.spinner{{display:inline-block;width:32px;height:32px;border:3px solid #2a2a3a;border-top-color:#6366f1;border-radius:50%;animation:spin .8s linear infinite}}@keyframes spin{{to{{transform:rotate(360deg)}}}}</style>
</head><body>
<div class="box">
  <div class="spinner" id="spinner"></div>
  <p style="margin-top:1rem;color:#a1a1aa" id="status">Completing sign-in…</p>
</div>
<script>
(async () => {{
  try {{
    const sb = window.supabase.createClient(
      "{auth_cfg['supabase_url']}",
      "{auth_cfg['supabase_anon_key']}"
    );
    // Supabase JS auto-detects the hash fragment and hydrates the session
    const {{ data, error }} = await sb.auth.getSession();
    if (error || !data.session) {{
      document.getElementById('status').textContent = 'Sign-in failed. Redirecting to login…';
      setTimeout(() => window.location.href = '/login', 2000);
      return;
    }}
    const resp = await fetch('/api/auth/set-session', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{
        access_token: data.session.access_token,
        refresh_token: data.session.refresh_token,
        expires_at: data.session.expires_at,
      }}),
    }});
    if (resp.ok) {{
      // Read redirect target: localStorage (survives OAuth) > query param > /chat
      var nextUrl = '/chat';
      try {{
        var stored = localStorage.getItem('orion_auth_next');
        if (stored) {{ nextUrl = stored; localStorage.removeItem('orion_auth_next'); }}
      }} catch(e) {{}}
      if (nextUrl === '/chat') {{
        var qp = new URLSearchParams(window.location.search).get('next');
        if (qp) {{ nextUrl = qp; }}
      }}
      window.location.href = nextUrl;
    }} else if (resp.status === 403) {{
      // Email not on allowlist — sign out of Supabase then bounce to login.
      try {{ await sb.auth.signOut(); }} catch(e) {{}}
      try {{ localStorage.removeItem('orion_auth_next'); }} catch(e) {{}}
      document.getElementById('status').textContent = 'Access denied. Redirecting…';
      setTimeout(() => window.location.href = '/login?denied=1', 1200);
    }} else {{
      document.getElementById('status').textContent = 'Session error. Redirecting…';
      setTimeout(() => window.location.href = '/login', 2000);
    }}
  }} catch (e) {{
    console.error('[callback]', e);
    document.getElementById('status').textContent = 'Error. Redirecting…';
    setTimeout(() => window.location.href = '/login', 2000);
  }}
}})();
</script>
</body></html>
""")


# ═══════════════════════════════════════════════════════════════════
#  STRIPE BILLING ROUTES
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/stripe/config")
async def api_stripe_config():
    """Return public Stripe config for frontend."""
    return JSONResponse({
        "publishable_key": STRIPE_PUBLISHABLE_KEY,
        "configured": bool(STRIPE_PUBLISHABLE_KEY),
        "tiers": TIER_INFO,
    })


@app.get("/api/stripe/subscription")
async def api_stripe_subscription(request: Request):
    """Get current user's subscription status."""
    user = getattr(request.state, "user", None)
    if not user:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    sub_info = get_user_subscription(user["id"])
    return JSONResponse(sub_info)


@app.post("/api/stripe/checkout")
async def api_stripe_checkout(request: Request):
    """Pro subscriptions are disabled — Orion Forge is pay-per-use (credits)."""
    return JSONResponse(
        {
            "error": "Monthly subscriptions are disabled. Orion Forge is pay-per-use — "
                     "buy credits in the Store and pay only 2× the API cost.",
            "redirect": "/store",
        },
        status_code=400,
    )


@app.post("/api/stripe/portal")
async def api_stripe_portal(request: Request):
    """Create a Stripe Billing Portal session for subscription management."""
    user = getattr(request.state, "user", None)
    if not user:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    base_url = str(request.base_url).rstrip("/")
    result = create_billing_portal_session(
        user_id=user["id"],
        return_url=f"{base_url}/plans",
    )
    if "error" in result:
        return JSONResponse(result, status_code=400)
    return JSONResponse(result)


@app.post("/api/stripe/webhook")
async def api_stripe_webhook(request: Request):
    """Handle Stripe webhook events (subscription changes, payments)."""
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")
    result = handle_webhook_event(payload, sig_header)
    if "error" in result:
        return JSONResponse(result, status_code=400)
    return JSONResponse(result)


# ═══════════════════════════════════════════════════════════════════
#  STORE & CREDITS API
# ═══════════════════════════════════════════════════════════════════

@app.get("/store", response_class=HTMLResponse)
async def page_store(request: Request):
    """Store page — browse tools, buy credit packs, purchase skins."""
    user = getattr(request.state, "user", None)
    credits = get_user_credits(user["id"]) if user else 0
    purchases = get_user_purchases(user["id"]) if user else {"tools": [], "skins": ["default"]}
    auth_cfg = get_auth_config()
    packs_list = [{"id": k, **v} for k, v in CREDIT_PACKS.items()]
    return templates.TemplateResponse(request, "store.html", {
        "page": "store",
        "user": user,
        "credits": credits,
        "catalog": STORE_CATALOG,
        "credit_packs": packs_list,
        "tool_costs": TOOL_CREDIT_COSTS,
        "purchases": purchases,
        "skin_prices": SKIN_PRICES,
        "store_packs": STORE_PACKS,
        "stripe_publishable_key": STRIPE_PUBLISHABLE_KEY,
        "auth_config": auth_cfg,
    })


@app.get("/api/credits/balance")
async def api_credits_balance(request: Request):
    """Return the current user's credit balance."""
    user = getattr(request.state, "user", None)
    if not user:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    balance = get_user_credits(user["id"])
    return JSONResponse({"credits": balance})


@app.post("/api/credits/buy")
async def api_credits_buy(request: Request):
    """Create a Stripe checkout session for a credit pack."""
    user = getattr(request.state, "user", None)
    if not user:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    body = await request.json()
    pack_id = body.get("pack_id", "")
    base_url = str(request.base_url).rstrip("/")
    result = create_credits_checkout_session(
        user_id=user["id"],
        user_email=user.get("email", ""),
        pack_id=pack_id,
        # pack + Stripe session id ride back so the success page can fire
        # Google Analytics purchase events ({CHECKOUT_SESSION_ID} is replaced
        # by Stripe with the real session id on redirect).
        success_url=f"{base_url}/store?credits_success=1&pack={pack_id}&sid={{CHECKOUT_SESSION_ID}}",
        cancel_url=f"{base_url}/store?credits_canceled=1",
    )
    if "error" in result:
        return JSONResponse(result, status_code=400)
    return JSONResponse(result)


@app.post("/api/credits/confirm")
async def api_credits_confirm(request: Request):
    """Fallback credit fulfillment after the Stripe checkout success redirect.

    The webhook is the primary fulfillment path; this lets the success page
    deliver credits immediately and resiliently by verifying the Checkout
    Session server-side. Idempotent with the webhook, so it never double-credits.
    """
    user = getattr(request.state, "user", None)
    if not user:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    body = await request.json()
    session_id = body.get("session_id", "")
    if not session_id:
        return JSONResponse({"error": "Missing session_id"}, status_code=400)
    result = fulfill_credits_for_session(session_id, expected_user_id=user["id"])
    return JSONResponse({"ok": bool(result.get("ok")), "result": result, "credits": get_user_credits(user["id"])})


@app.get("/api/store/catalog")
async def api_store_catalog(request: Request):
    """Return the full store catalog and credit packs."""
    user = getattr(request.state, "user", None)
    credits = get_user_credits(user["id"]) if user else 0
    return JSONResponse({
        "catalog": STORE_CATALOG,
        "credit_packs": CREDIT_PACKS,
        "credits": credits,
    })


@app.get("/api/credits/history")
async def api_credits_history(request: Request):
    """Return the user's credit transaction history."""
    user = getattr(request.state, "user", None)
    if not user:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    history = get_credit_history(user["id"])
    return JSONResponse({"history": history})


@app.post("/api/credits/deduct-tool")
async def api_credits_deduct_tool(request: Request):
    """One-time purchase of a tool from the store (deducts credits, unlocks forever)."""
    user = getattr(request.state, "user", None)
    if not user:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    body = await request.json()
    tool_id = body.get("tool_id", "")
    result = purchase_tool(user["id"], tool_id)
    if "error" in result:
        status = 402 if "Insufficient" in result["error"] else 400
        return JSONResponse(result, status_code=status)
    return JSONResponse({"success": True, "tool_id": tool_id, "cost": result["cost"], "balance": result["balance"]})


@app.post("/api/store/purchase-skin")
async def api_purchase_skin(request: Request):
    """One-time purchase of a skin (deducts credits, unlocks forever)."""
    user = getattr(request.state, "user", None)
    if not user:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    body = await request.json()
    skin_id = body.get("skin_id", "")
    result = purchase_skin(user["id"], skin_id)
    if "error" in result:
        status = 402 if "Insufficient" in result.get("error", "") else 400
        return JSONResponse(result, status_code=status)
    return JSONResponse({"success": True, "skin_id": skin_id, "cost": result["cost"], "balance": result["balance"]})


@app.post("/api/store/purchase-agent")
async def api_purchase_agent(request: Request):
    """One-time purchase of an AI agent (deducts credits, seeds profile + knowledge)."""
    user = getattr(request.state, "user", None)
    if not user:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    body = await request.json()
    agent_catalog_id = body.get("agent_id", "")

    result = purchase_agent(user["id"], agent_catalog_id)
    if "error" in result:
        status = 402 if "Insufficient" in result.get("error", "") else 400
        return JSONResponse(result, status_code=status)

    agent_id = result["agent_id"]

    # ── Seed soul script + prompt into Knowledge notes ──
    try:
        _seed_agent_knowledge(agent_id)
    except Exception as exc:
        log.warning("[store] Agent knowledge seeding failed for '%s': %s", agent_id, exc)

    return JSONResponse({
        "success": True,
        "agent_id": agent_id,
        "catalog_id": agent_catalog_id,
        "cost": result["cost"],
        "balance": result["balance"],
    })


@app.post("/api/store/purchase-pack")
async def api_purchase_pack(request: Request):
    """Purchase a bundle pack (agents + tools at discount)."""
    user = getattr(request.state, "user", None)
    if not user:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    body = await request.json()
    pack_id = body.get("pack_id", "")

    result = purchase_pack(user["id"], pack_id)
    if "error" in result:
        status = 402 if "Insufficient" in result.get("error", "") else 400
        return JSONResponse(result, status_code=status)

    # Seed knowledge for each newly unlocked agent
    for agent_id in result.get("agents_unlocked", []):
        try:
            _seed_agent_knowledge(agent_id)
        except Exception as exc:
            log.warning("[store] Pack agent seeding failed for '%s': %s", agent_id, exc)

    return JSONResponse({
        "success": True,
        "pack_id": pack_id,
        "cost": result["cost"],
        "balance": result["balance"],
        "agents_unlocked": result["agents_unlocked"],
        "tools_unlocked": result["tools_unlocked"],
    })


def _seed_agent_knowledge(agent_id: str):
    """Seed an agent's soul script and prompt into the Knowledge tab on purchase.

    Creates notes in the 'Soul Scripts' and 'Prompts' folders, and attaches
    them to the agent's config so they're injected during conversations.
    """
    import uuid
    from datetime import datetime, timezone

    notes_dir = _DATA_DIR / "user_notes"
    notes_dir.mkdir(parents=True, exist_ok=True)

    # ── Ensure folders exist ──
    folders_file = notes_dir / "folders.json"
    folders = _read_json(folders_file, [])
    folder_ids = {f["id"] for f in folders}

    if "soul_scripts" not in folder_ids:
        folders.append({"id": "soul_scripts", "name": "Soul Scripts", "emoji": "🧬", "pinned": True, "color": "#818cf8"})
    if "prompts" not in folder_ids:
        folders.append({"id": "prompts", "name": "Prompts", "emoji": "📝", "pinned": True, "color": "#f472b6"})
    _write_json(folders_file, folders)

    # ── Read the soul script (directive) ──
    directive_path = _DIRECTIVES_DIR / f"{agent_id}.md"
    soul_script_text = directive_path.read_text(encoding="utf-8") if directive_path.exists() else ""

    # ── Read the system prompt ──
    prompt_path = _PROMPTS_DIR / f"{agent_id}.system.md"
    prompt_text = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""

    # ── Load existing note index ──
    index_file = notes_dir / "index.json"
    index = _read_json(index_file, [])
    now_iso = datetime.now(timezone.utc).isoformat()

    note_ids = []
    ss_ids = set()

    # ── Create soul script note ──
    if soul_script_text:
        ss_id = uuid.uuid4().hex[:8]
        ss_title = f"{agent_id.capitalize()} — Soul Script"
        index.append({
            "id": ss_id, "title": ss_title, "emoji": "🧬",
            "preview": "", "section": "soul_scripts",
            "created": now_iso, "updated": now_iso,
        })
        # Convert markdown to simple HTML for the note
        ss_html = soul_script_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        ss_html = f"<pre>{ss_html}</pre>"
        _write_json(notes_dir / f"{ss_id}.json", {
            "id": ss_id, "title": ss_title, "emoji": "🧬",
            "content_html": ss_html, "preview": "",
            "section": "soul_scripts",
            "created": now_iso, "updated": now_iso,
        })
        note_ids.append(ss_id)
        ss_ids.add(ss_id)

    # ── Create prompt note ──
    if prompt_text:
        pr_id = uuid.uuid4().hex[:8]
        pr_title = f"{agent_id.capitalize()} — System Prompt"
        index.append({
            "id": pr_id, "title": pr_title, "emoji": "📝",
            "preview": "", "section": "prompts",
            "created": now_iso, "updated": now_iso,
        })
        pr_html = prompt_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        pr_html = f"<pre>{pr_html}</pre>"
        _write_json(notes_dir / f"{pr_id}.json", {
            "id": pr_id, "title": pr_title, "emoji": "📝",
            "content_html": pr_html, "preview": "",
            "section": "prompts",
            "created": now_iso, "updated": now_iso,
        })
        note_ids.append(pr_id)

    _write_json(index_file, index)

    # ── Attach notes to agent config ──
    if note_ids:
        cfg = _get_agent_config(agent_id)
        existing = cfg.get("attached_notes", [])
        cfg["attached_notes"] = list(set(existing + note_ids))
        modes = cfg.get("note_modes", {})
        essential = cfg.get("essential_notes", [])
        for nid in note_ids:
            # Soul script notes default to directive mode (semantic retrieval)
            modes[nid] = "directive" if nid in ss_ids else "always"
            essential.append(nid)
        cfg["note_modes"] = modes
        cfg["essential_notes"] = list(set(essential))
        _save_agent_config(agent_id, cfg)

    log.info("[store] Seeded knowledge for agent '%s': %d notes created", agent_id, len(note_ids))


@app.get("/api/store/purchases")
async def api_store_purchases(request: Request):
    """Return all items the user has purchased."""
    user = getattr(request.state, "user", None)
    if not user:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    purchases = get_user_purchases(user["id"])
    return JSONResponse(purchases)


@app.get("/plans")
async def page_plans(request: Request):
    """Legacy subscription page — Orion Forge is now pay-per-use, so this
    redirects to the credits Store. There is no monthly subscription."""
    return RedirectResponse(url="/store", status_code=302)


# ═══════════════════════════════════════════════════════════════════
#  PAGE ROUTES
# ═══════════════════════════════════════════════════════════════════

@app.get("/", response_class=RedirectResponse)
async def root():
    return RedirectResponse(url="/chat", status_code=302)

@app.get("/chat", response_class=HTMLResponse)
async def page_chat(request: Request):
    agents = _list_unlocked_agents(request)
    store = _load_connections()
    conns = [c for c in store.get("connections", []) if c.get("enabled")]
    platform_conns = [
        c for c in conns
        if c.get("platform_hosted") and c.get("provider") not in ("elevenlabs", "edge-tts")
    ]
    user_conns = [
        c for c in conns
        if not c.get("platform_hosted") and c.get("provider") not in ("elevenlabs", "edge-tts")
    ]
    settings = _load_settings()
    stt_cfg = settings.get("stt", {})

    # Merge voice fields from agent profiles into avatar_map so the
    # frontend can route TTS to the correct voice per agent.
    avatar_map = dict(settings.get("agent_avatars", {}))
    for agent_name in agents:
        prof = _load_profile(agent_name)
        entry = dict(avatar_map.get(agent_name, {}))
        if prof.get("inworld_voice"):
            entry["inworld_voice"] = prof["inworld_voice"]
        if prof.get("edge_voice"):
            entry["edge_voice"] = prof["edge_voice"]
        if prof.get("voice_id"):
            entry["voice_id"] = prof["voice_id"]
        avatar_map[agent_name] = entry

    return templates.TemplateResponse(request, "chat.html", {
        "page": "chat",
        "agents": agents, "connections": conns,
        "platform_connections": platform_conns,
        "user_connections": user_conns,
        "chat_index": _load_chat_index(),
        "agent_connections": store.get("agent_connections", {}),
        "avatar_map": avatar_map,
        "user_profile": settings.get("user_profile", {}),
        "chat_background": settings.get("chat_background") or "",
        "pinned_models": settings.get("pinned_models", []),
        "stt_provider": stt_cfg.get("provider", "elevenlabs"),
        "chat_defaults": settings.get("chat_defaults", {}),
    })

@app.get("/profiles", response_class=HTMLResponse)
async def page_profiles(request: Request):
    agents = _list_unlocked_agents(request)
    settings = _load_settings()
    agent_data = {}
    for name in agents:
        profile = _load_profile(name)
        cfg = settings.get("agent_configs", {}).get(name, {})
        agent_data[name] = {
            "profile": profile, "config": cfg,
            "system_prompt": "",
            "soul_script": "",
            "display_name": cfg.get("display_name", name),
            "description": cfg.get("description", ""),
        }
    store = _load_connections()
    all_models = []
    for c in store.get("connections", []):
        if c.get("enabled"):
            for m in c.get("models", []):
                if m not in all_models:
                    all_models.append(m)
    notes = [n for n in _load_notes_index() if not n.get("trashed")]
    # Builtin notes (agent-specific .md files in notes/ dir)
    notes_dir = _DATA_DIR / "user_notes"
    builtin_notes = {}
    for name in agents:
        agent_notes = []
        for md_file in sorted((_PROJECT_ROOT / "notes").glob("*.md")) if (_PROJECT_ROOT / "notes").exists() else []:
            agent_notes.append({"file": md_file.name, "attached": True, "mode": "always"})
        builtin_notes[name] = agent_notes
    # Gather all registered tool names for the tools toggle section
    try:
        from src.tools.registry import list_registered_tools
        all_tool_names = list_registered_tools()
    except Exception:
        all_tool_names = ["continuation_update", "cost_tracker", "directives", "echo", "memory", "web_search"]
    # Tool display info from catalogue
    tool_display = {}
    for t in _TOOL_CATALOGUE:
        tool_display[t["name"]] = {
            "icon": t.get("icon", "🔧"),
            "display_name": t.get("display_name", t["name"]),
            "description": t.get("description", "")[:100],
            "status": t.get("status", "planned"),
        }
    # Also include registry tools not in the catalogue
    for tn in all_tool_names:
        if tn not in tool_display:
            tool_display[tn] = {"icon": "🔧", "display_name": tn, "description": "", "status": "ready"}
    return templates.TemplateResponse(request, "profiles.html", {
        "page": "profiles",
        "agents": agents, "agent_data": agent_data,
        "all_models": all_models, "notes": notes,
        "avatar_map": settings.get("agent_avatars", {}),
        "user_profile": settings.get("user_profile", {}),
        "connections": [c for c in store.get("connections", []) if c.get("enabled")],
        "builtin_notes": builtin_notes,
        "all_tool_names": all_tool_names,
        "tool_display": tool_display,
    })

@app.get("/vault", response_class=HTMLResponse)
async def page_vault(request: Request, q: str = "", scope: str = "", category: str = "", sort: str = "newest"):
    fm = _get_faiss_memory()
    vs = _get_vault_store()
    memories, scopes, categories = [], [], []
    # Load max_total_memories from memory profile (0 = unlimited)
    mp = _load_memory_profile()
    max_total = mp.get("retention_policy", {}).get("max_total_memories", 5000)
    stats = {"active_count": 0, "max_active": max_total, "utilization_pct": 0,
             "by_scope": {}, "raw_lines": 0, "compactable_lines": 0,
             "bloat_ratio": "1.0x", "deleted_count": 0}
    if fm:
        try:
            raw_stats = fm.stats()
            raw_stats["max_active"] = max_total
            if max_total and max_total > 0:
                raw_stats["utilization_pct"] = min(100, round(raw_stats.get("active_count", 0) / max_total * 100))
            else:
                raw_stats["utilization_pct"] = 0
            stats = raw_stats
        except Exception:
            pass
        try:
            if q:
                raw_results = fm.search(q, scope=scope or None, top_k=50)
                # Filter out low-relevance results (cosine similarity threshold)
                MIN_SCORE = 0.25
                memories = [r for r in raw_results if r.get("score", 0) >= MIN_SCORE]
            else:
                all_mems = fm.list_all(scope=scope or None)
                if category:
                    all_mems = [m for m in all_mems if getattr(m, "category", "") == category]
                memories = [m.__dict__ if hasattr(m, "__dict__") else m for m in all_mems]

            all_raw = fm.list_all()
            scopes = sorted({getattr(m, "scope", "") for m in all_raw} - {""})
            categories = sorted({getattr(m, "category", "") for m in all_raw} - {""})
        except Exception as e:
            import logging
            logging.getLogger(__name__).exception("[vault] list failed: %s", e)
            memories, scopes, categories = [], [], []
    elif vs:
        # ── VaultStore fallback (no FAISS / no semantic search) ──
        all_active = vs.read_active()
        active_count = len(all_active)
        stats["active_count"] = active_count
        stats["raw_lines"] = len(vs.read_all())
        stats["compactable_lines"] = stats["raw_lines"] - active_count
        if max_total and max_total > 0:
            stats["utilization_pct"] = min(100, round(active_count / max_total * 100))

        filtered = all_active
        if scope:
            filtered = [m for m in filtered if m.scope == scope]
        if category:
            filtered = [m for m in filtered if m.category == category]
        if q:
            q_lower = q.lower()
            filtered = [m for m in filtered if q_lower in (m.text or "").lower()]
        memories = [m.__dict__ for m in filtered]

        scopes = sorted({m.scope for m in all_active if m.scope} - {""})
        categories = sorted({m.category for m in all_active if m.category} - {""})

    # ── Sort memories ──────────────────────────────────────────
    def _sort_key(m):
        """Extract sort key from a Memory object or dict."""
        if isinstance(m, dict):
            get = m.get
        else:
            get = lambda k, d="": getattr(m, k, d)
        if sort == "oldest":
            return get("created_at", "")
        elif sort == "scope":
            return (get("scope", ""), get("created_at", ""))
        elif sort == "category":
            return (get("category", ""), get("created_at", ""))
        elif sort == "tier":
            return (get("tier", "canon"), get("created_at", ""))
        elif sort == "alpha":
            return (get("text", "") or "").lower()
        elif sort == "source":
            return (get("source", "") or "", get("created_at", ""))
        elif sort == "tag":
            tags = get("tags", []) or []
            return ((tags[0] if tags else "~"), get("created_at", ""))
        elif sort == "updated":
            return get("updated_at", "") or get("created_at", "") or ""
        else:  # newest (default)
            return get("created_at", "")

    reverse = sort not in ("oldest", "alpha", "tag")  # ascending for oldest, alpha, tag
    if sort == "updated":
        reverse = True
    memories = sorted(memories, key=_sort_key, reverse=reverse)

    return templates.TemplateResponse(request, "vault.html", {
        "page": "vault",
        "memories": memories, "stats": stats,
        "scopes": scopes, "categories": categories,
        "search_query": q, "current_scope": scope, "current_category": category,
        "current_sort": sort,
    })

@app.get("/knowledge", response_class=HTMLResponse)
async def page_knowledge(request: Request):
    notes = [n for n in _load_notes_index() if not n.get("trashed")]
    folders = _load_folders()
    return templates.TemplateResponse(request, "knowledge.html", {
        "page": "knowledge", "notes": notes,
        "folders": folders,
    })

@app.get("/knowledge/{note_id}/edit", response_class=HTMLResponse)
async def page_knowledge_edit(request: Request, note_id: str):
    note = _load_note(note_id)
    if not note:
        return RedirectResponse(url="/knowledge", status_code=302)
    folders = _load_folders()
    return templates.TemplateResponse(request, "knowledge_edit.html", {
        "page": "knowledge", "note": note,
        "folders": folders,
    })

@app.get("/settings", response_class=HTMLResponse)
async def page_settings(request: Request, tab: str = "api_keys"):
    store = _load_connections()
    agents = _list_unlocked_agents(request)
    # Strip all secret key values before passing to template
    safe_settings = strip_secrets_for_template(_load_settings())
    return templates.TemplateResponse(request, "settings.html", {
        "page": "settings",
        "connections": store.get("connections", []),
        "settings": safe_settings, "tab": tab,
        "agents": agents,
    })

@app.get("/pricing", response_class=RedirectResponse)
async def page_pricing_redirect():
    """Redirect old pricing page to tools (cost_tracker tool)."""
    return RedirectResponse(url="/tools#cost_tracker", status_code=302)


# ── Skins page ────────────────────────────────────────────────────

@app.get("/skins", response_class=HTMLResponse)
async def page_skins(request: Request):
    settings = _load_settings()
    active_skin = settings.get("skin", "default")
    user = getattr(request.state, "user", None)
    purchases = get_user_purchases(user["id"]) if user else {"tools": [], "skins": ["default"]}
    credits = get_user_credits(user["id"]) if user else 0
    return templates.TemplateResponse(request, "skins.html", {
        "page": "skins",
        "active_skin": active_skin,
        "owned_skins": purchases.get("skins", ["default"]),
        "skin_prices": SKIN_PRICES,
        "credits": credits,
    })

@app.put("/api/skin")
async def api_set_skin(request: Request):
    """Save active UI skin to settings.json (only if owned)."""
    body = await request.json()
    skin_id = body.get("skin", "default")
    user = getattr(request.state, "user", None)
    if user:
        if not user_owns_item(user["id"], skin_id, "skins") and skin_id != "default":
            return JSONResponse({"error": "You don't own this skin. Purchase it in the Store."}, status_code=403)
    settings = _load_settings()
    settings["skin"] = skin_id
    _save_settings(settings)
    return {"ok": True, "skin": skin_id}

@app.get("/api/skin")
async def api_get_skin():
    """Return the active skin id."""
    settings = _load_settings()
    return {"skin": settings.get("skin", "default")}


# ── Tools page ────────────────────────────────────────────────────

# Hardcoded tool catalogue (matches agent-runtime registry).
# Tools can be uploaded/implemented one at a time later.
_TOOL_CATALOGUE = [
    {
        "name": "web_search",
        "display_name": "Web Search Tool",
        "icon": "🔍",
        "icon_bg": "rgba(34,211,238,0.12)",
        "icon_color": "#22d3ee",
        "description": "Search the web via SearXNG and scrape/summarise results. Supports fast, normal, and deep modes. Includes knowledge-gate to prevent unnecessary searches.",
        "status": "ready",
        "parameters": [
            {"name": "action", "type": "string", "required": True, "description": "search or scrape", "enum": ["search", "scrape"]},
            {"name": "query", "type": "string", "required": False, "description": "Search query string", "enum": []},
            {"name": "mode", "type": "string", "required": False, "description": "Search depth preset", "enum": ["fast", "normal", "deep"]},
            {"name": "url", "type": "string", "required": False, "description": "URL to scrape directly", "enum": []},
            {"name": "knowledge_check", "type": "string", "required": False, "description": "What you already know about this topic", "enum": []},
            {"name": "reason", "type": "string", "required": False, "description": "Why the internet is needed", "enum": []},
            {"name": "parser", "type": "library", "required": False, "description": "HTML parsing powered by BeautifulSoup4 (bs4)", "enum": ["beautifulsoup4"]},
        ],
    },
    {
        "name": "email",
        "display_name": "Email Tool",
        "icon": "📧",
        "icon_bg": "rgba(244,114,182,0.12)",
        "icon_color": "#f472b6",
        "description": "Send emails through configured SMTP accounts or API relay. Supports multiple accounts, confirmation gating, and per-agent defaults.",
        "status": "ready",
        "parameters": [
            {"name": "action", "type": "string", "required": True, "description": "send, status, or accounts", "enum": ["send", "status", "accounts"]},
            {"name": "subject", "type": "string", "required": False, "description": "Email subject line (required for send)", "enum": []},
            {"name": "body", "type": "string", "required": False, "description": "Email body text (required for send)", "enum": []},
            {"name": "recipients", "type": "array", "required": False, "description": "List of recipient email addresses (required for send)", "enum": []},
            {"name": "account_id", "type": "string", "required": False, "description": "Specific account ID to send from", "enum": []},
            {"name": "confirmation", "type": "string", "required": False, "description": "Set to 'confirmed' after user approval", "enum": ["confirmed"]},
        ],
    },
    {
        "name": "memory",
        "display_name": "Memory Tool",
        "icon": "🧠",
        "icon_bg": "rgba(129,140,248,0.12)",
        "icon_color": "#818cf8",
        "description": "Store, search, and manage durable memories in the FAISS-backed vault. Memories persist across sessions and are searchable by meaning via vector embeddings.",
        "status": "ready",
        "parameters": [
            {"name": "action", "type": "string", "required": True, "description": "Operation to perform", "enum": ["add", "add_many", "remember", "search", "recall", "get", "update", "delete", "bulk_delete", "list", "stats", "compact", "rebuild_index"]},
            {"name": "text", "type": "string", "required": False, "description": "Memory content (for add/remember/update)", "enum": []},
            {"name": "query", "type": "string", "required": False, "description": "Search query text (for search action)", "enum": []},
            {"name": "scope", "type": "string", "required": False, "description": "Memory scope — 'shared' for cross-agent memories, or an agent name for agent-specific memories", "enum": sorted(VALID_SCOPES)},
            {"name": "category", "type": "string", "required": False, "description": "Memory category", "enum": sorted(VALID_CATEGORIES)[:5] + ["…"]},
            {"name": "tags", "type": "array", "required": False, "description": "Optional tags for filtering", "enum": []},
            {"name": "source", "type": "string", "required": False, "description": "Origin of the memory", "enum": ["chat", "manual", "tool", "operator", "promotion"]},
            {"name": "memory_id", "type": "string", "required": False, "description": "Memory ID (for get/update/delete)", "enum": []},
            {"name": "memory_ids", "type": "array", "required": False, "description": "List of memory IDs (for bulk_delete)", "enum": []},
            {"name": "memories", "type": "array", "required": False, "description": "Array of memory objects (for add_many batch)", "enum": []},
            {"name": "limit", "type": "integer", "required": False, "description": "Max results (default 10 for search, 20 for recall/list)", "enum": []},
        ],
    },
    {
        "name": "echo",
        "display_name": "Echo",
        "icon": "📢",
        "icon_bg": "rgba(16,185,129,0.12)",
        "icon_color": "#10b981",
        "description": "Simple echo tool for testing. Returns the input text unchanged. Useful for verifying tool dispatch works correctly.",
        "status": "ready",
        "parameters": [
            {"name": "text", "type": "string", "required": True, "description": "Text to echo back", "enum": []},
        ],
    },
    {
        "name": "directives",
        "display_name": "Directives",
        "icon": "📜",
        "icon_bg": "rgba(251,191,36,0.12)",
        "icon_color": "#fbbf24",
        "description": "Manage runtime directives — the living governance document that shapes agent behavior. Read, update, or query active directives.",
        "status": "ready",
        "tags": ["agi_loop"],
        "parameters": [
            {"name": "action", "type": "string", "required": True, "description": "Operation", "enum": ["read", "update", "list"]},
            {"name": "directive_id", "type": "string", "required": False, "description": "Target directive identifier", "enum": []},
            {"name": "content", "type": "string", "required": False, "description": "New directive content", "enum": []},
        ],
    },
    {
        "name": "continuation_update",
        "display_name": "Continuation Update",
        "icon": "🔄",
        "icon_bg": "rgba(52,211,153,0.12)",
        "icon_color": "#34d399",
        "description": "Maintain a single evolving 'pickup later' markdown file per profile. Append timestamped blocks or upsert named sections.",
        "status": "ready",
        "tags": ["agi_loop"],
        "parameters": [
            {"name": "profile", "type": "string", "required": True, "description": "Profile name (e.g. 'orion')", "enum": []},
            {"name": "mode", "type": "string", "required": False, "description": "Write mode", "enum": ["append", "replace_section"]},
            {"name": "content", "type": "string", "required": True, "description": "Markdown content to write", "enum": []},
            {"name": "section", "type": "string", "required": False, "description": "Section heading (replace_section mode only)", "enum": []},
        ],
    },
    {
        "name": "runtime_info",
        "display_name": "Runtime Info",
        "icon": "📊",
        "icon_bg": "rgba(56,189,248,0.12)",
        "icon_color": "#38bdf8",
        "description": "Read-only snapshot of runtime state: agent identity, model/provider, policy, allowed tools, memory config, directive state, and vault health. Includes diff tracking between calls. Available even in stasis mode.",
        "status": "ready",
        "tags": ["agi_loop"],
        "parameters": [],
    },
    {
        "name": "inbox",
        "display_name": "Inbox",
        "icon": "💬",
        "icon_bg": "rgba(239,68,68,0.12)",
        "icon_color": "#ef4444",
        "description": "Unified agent-to-operator messaging and task queue. The AGI loop uses this to send messages, flag warnings, request approval, queue tasks, and fetch pending work.",
        "status": "ready",
        "tags": ["agi_loop"],
        "parameters": [
            {"name": "action", "type": "string", "required": True, "description": "Action to perform", "enum": ["send", "add_task", "next_task", "ack"]},
            {"name": "type", "type": "string", "required": False, "description": "Message type (for send)", "enum": ["message", "task", "tool_request", "warning", "idea"]},
            {"name": "priority", "type": "string", "required": False, "description": "Priority level", "enum": ["low", "normal", "high", "urgent"]},
            {"name": "subject", "type": "string", "required": False, "description": "Short summary (max 120 chars, required for send)", "enum": []},
            {"name": "body", "type": "string", "required": False, "description": "Full message text (max 2000 chars, required for send)", "enum": []},
            {"name": "task", "type": "string", "required": False, "description": "Task description (required for add_task)", "enum": []},
            {"name": "task_id", "type": "string", "required": False, "description": "Task/message ID (required for ack)", "enum": []},
            {"name": "needs_approval", "type": "boolean", "required": False, "description": "Requires operator approval before proceeding", "enum": []},
        ],
    },
    {
        "name": "computer_use",
        "icon": "🖥️",
        "icon_bg": "rgba(99,102,241,0.12)",
        "icon_color": "#6366f1",
        "description": "Control the computer — take screenshots, move the mouse, click, type, and interact with desktop applications programmatically.",
        "status": "concept",
        "parameters": [
            {"name": "action", "type": "string", "required": True, "description": "Action to perform", "enum": ["screenshot", "click", "type", "move", "scroll", "key"]},
            {"name": "x", "type": "integer", "required": False, "description": "X coordinate", "enum": []},
            {"name": "y", "type": "integer", "required": False, "description": "Y coordinate", "enum": []},
            {"name": "text", "type": "string", "required": False, "description": "Text to type or key to press", "enum": []},
        ],
    },
    {
        "name": "cost_tracker",
        "display_name": "Cost Tracker",
        "icon": "💰",
        "icon_bg": "rgba(16,185,129,0.12)",
        "icon_color": "#10b981",
        "description": "Manage token pricing, track costs per model, and view spending across all LLM API calls. Edit per-model rates and monitor usage in real time.",
        "status": "ready",
        "parameters": [
            {"name": "action", "type": "string", "required": True, "description": "Operation to perform", "enum": ["get_pricing", "set_pricing", "list_models", "cost_summary", "cost_log", "session_cost"]},
            {"name": "provider", "type": "string", "required": False, "description": "Provider name (openai, anthropic, deepseek, ollama)", "enum": []},
            {"name": "model", "type": "string", "required": False, "description": "Model name", "enum": []},
            {"name": "period", "type": "string", "required": False, "description": "Time period for cost summary", "enum": ["today", "this_week", "this_month", "all_time"]},
        ],
    },
    {
        "name": "agi_loop",
        "display_name": "AGI Loop",
        "icon": "🔁",
        "icon_bg": "rgba(168,85,247,0.12)",
        "icon_color": "#a855f7",
        "description": "Autonomous tick-based execution loop. Runs the agent on a timer with budget caps, error-streak safeguards, and tier-escalation. Agents can query status, tick history, or request pause/resume.",
        "status": "ready",
        "tags": ["agi_loop"],
        "parameters": [
            {"name": "action", "type": "string", "required": True, "description": "Action to perform", "enum": ["status", "tick_history", "request_pause", "request_resume"]},
            {"name": "limit", "type": "integer", "required": False, "description": "Max tick history entries to return (default 10)", "enum": []},
        ],
    },
    {
        "name": "model_router",
        "display_name": "Model Router",
        "icon": "🔀",
        "icon_bg": "rgba(245,158,11,0.12)",
        "icon_color": "#f59e0b",
        "description": "Task-aware tiered model routing with scored classification, stuck-loop escalation, budget gates, and disabled-tier skipping. Classifies prompts by task type and routes to the optimal model tier. Agents can query routing decisions, inspect the tier map, or check budget status.",
        "status": "ready",
        "tags": ["agi_loop"],
        "parameters": [
            {"name": "action", "type": "string", "required": True, "description": "Action to perform", "enum": ["resolve", "list_tiers", "get_map", "classify", "budget"]},
            {"name": "text", "type": "string", "required": False, "description": "Stimulus text to classify (for resolve/classify)", "enum": []},
        ],
    },
]

@app.get("/tools", response_class=HTMLResponse)
async def page_tools(request: Request):
    agents = _list_agents()
    # Gather pricing data for the cost_tracker tool panel
    pricing = _load_pricing()
    store = _load_connections()
    connections = [c for c in store.get("connections", []) if c.get("enabled")]
    try:
        from src.observability.metering import read_cost_log, aggregate_costs
        from datetime import timedelta
        now = datetime.now(timezone.utc)
        all_events = read_cost_log(limit=100000)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        today_events = [e for e in all_events if e.get("ts", "") >= today_start]
        month_start = (now - timedelta(days=30)).isoformat()
        month_events = [e for e in all_events if e.get("ts", "") >= month_start]
        cost_stats = {
            "today": aggregate_costs(today_events),
            "this_month": aggregate_costs(month_events),
            "all_time": aggregate_costs(all_events),
        }
    except Exception:
        cost_stats = {"today": {}, "this_month": {}, "all_time": {}}
    # Web-search effective config for the config panel
    try:
        from src.tools.web_search import get_effective_config as _ws_cfg
        ws_config = _ws_cfg()
    except Exception:
        ws_config = {"searxng_url": "http://localhost:3000/search", "require_justification": True, "modes": {"fast": {"pages": 2, "return_count": 2, "word_limit": 800}, "normal": {"pages": 4, "return_count": 3, "word_limit": 1500}, "deep": {"pages": 8, "return_count": 5, "word_limit": 3000}}}
    # Identity + Memory profiles for the memory tool panel
    identity_profile = _load_identity_profile()
    memory_profile = _load_memory_profile()
    # FAISS memory stats
    try:
        fm = _get_faiss_memory()
        if fm:
            _fs = fm.stats()
            faiss_stats = {
                "total_memories": _fs.get("active_count", 0),
                "vault_path": str(_VAULT_PATH),
                "faiss_dir": str(_FAISS_DIR),
                "embedding_model": _fs.get("embedding_model", "all-mpnet-base-v2"),
                "faiss_vectors": _fs.get("faiss_vectors", 0),
                "vector_dimensions": fm._embedding_dim or 768,
                "index_type": type(fm.index).__name__ if fm.index else "IndexFlatIP",
                "in_sync": _fs.get("in_sync", True),
            }
        else:
            vs = _get_vault_store()
            _count = len(vs.read_active()) if vs else 0
            faiss_stats = {"total_memories": _count, "vault_path": str(_VAULT_PATH), "faiss_dir": str(_FAISS_DIR), "embedding_model": "N/A (VaultStore mode)", "faiss_vectors": 0, "vector_dimensions": 0, "index_type": "VaultStore", "in_sync": True}
    except Exception:
        faiss_stats = {"total_memories": 0, "vault_path": str(_VAULT_PATH), "faiss_dir": str(_FAISS_DIR), "embedding_model": "all-mpnet-base-v2", "faiss_vectors": 0, "vector_dimensions": 768, "index_type": "IndexFlatIP", "in_sync": True}
    # Email tool config
    try:
        from src.tools.email_tool import get_effective_config as _email_cfg
        _tools_user = getattr(request.state, "user", None)
        _tools_uid = _tools_user["id"] if _tools_user else ""
        email_config = _email_cfg(user_id=_tools_uid)
    except Exception:
        email_config = {"api_base_url": "http://127.0.0.1:8000", "timeout": 30, "require_confirmation": True, "accounts": []}
    return templates.TemplateResponse(request, "tools.html", {
        "page": "tools",
        "tools": _TOOL_CATALOGUE,
        "agents": agents,
        "total": len(_TOOL_CATALOGUE),
        "pricing": pricing,
        "connections": connections,
        "cost_stats": cost_stats,
        "web_search_config": ws_config,
        "identity_profile": identity_profile,
        "memory_profile": memory_profile,
        "faiss_stats": faiss_stats,
        "router_config": _load_model_router_config(),
        "email_config": email_config,
        "is_admin": _check_admin(request),
    })


# ── Web Search Config API ─────────────────────────────────────────
@app.get("/api/tools/web_search/config", response_class=JSONResponse)
async def api_web_search_config_get():
    """Return the effective web_search configuration."""
    from src.tools.web_search import get_effective_config
    return JSONResponse(get_effective_config())


@app.put("/api/tools/web_search/config", response_class=JSONResponse)
async def api_web_search_config_put(request: Request):
    """Save web_search configuration to settings.json."""
    body = await request.json()
    settings = _load_settings()
    if "tool_config" not in settings:
        settings["tool_config"] = {}
    settings["tool_config"]["web_search"] = body
    _save_settings(settings)
    from src.tools.web_search import get_effective_config
    return JSONResponse(get_effective_config())


# ── Email Tool Config API ──────────────────────────────────────────
@app.get("/api/tools/email/config", response_class=JSONResponse)
async def api_email_config_get(request: Request):
    """Return the effective email configuration (passwords masked)."""
    user = getattr(request.state, "user", None)
    uid = user["id"] if user else ""
    from src.tools.email_tool import get_effective_config as email_cfg
    return JSONResponse(email_cfg(user_id=uid))


@app.put("/api/tools/email/config", response_class=JSONResponse)
async def api_email_config_put(request: Request):
    """Save top-level email configuration (api_base_url, timeout, require_confirmation)."""
    body = await request.json()
    settings = _load_settings()
    if "tool_config" not in settings:
        settings["tool_config"] = {}
    if "email" not in settings["tool_config"]:
        settings["tool_config"]["email"] = {}
    ecfg = settings["tool_config"]["email"]
    # Only update non-account fields
    for key in ("api_base_url", "timeout", "require_confirmation"):
        if key in body:
            ecfg[key] = body[key]
    _save_settings(settings)
    from src.tools.email_tool import get_effective_config as email_cfg
    return JSONResponse(email_cfg())


@app.get("/api/tools/email/accounts", response_class=JSONResponse)
async def api_email_accounts_list(request: Request):
    """List email accounts for the authenticated user (passwords masked)."""
    user = getattr(request.state, "user", None)
    uid = user["id"] if user else ""
    from src.tools.email_tool import get_accounts
    return JSONResponse({"accounts": get_accounts(uid)})


@app.post("/api/tools/email/accounts", response_class=JSONResponse)
async def api_email_accounts_save(request: Request):
    """Create or update an email account for the authenticated user."""
    user = getattr(request.state, "user", None)
    uid = user["id"] if user else ""
    body = await request.json()
    from src.tools.email_tool import save_account
    saved = save_account(uid, body)
    # Mask password for response
    resp = dict(saved)
    if resp.get("password"):
        resp["password"] = "••••••••"
        resp["password_set"] = True
    return JSONResponse(resp)


@app.delete("/api/tools/email/accounts/{account_id}", response_class=JSONResponse)
async def api_email_accounts_delete(account_id: str, request: Request):
    """Delete an email account for the authenticated user."""
    user = getattr(request.state, "user", None)
    uid = user["id"] if user else ""
    from src.tools.email_tool import delete_account
    if delete_account(uid, account_id):
        return JSONResponse({"deleted": True, "id": account_id})
    return JSONResponse({"error": "Account not found"}, status_code=404)


@app.post("/api/tools/email/test", response_class=JSONResponse)
async def api_email_test(request: Request):
    """Send a test email through a specific account (scoped to user)."""
    user = getattr(request.state, "user", None)
    uid = user["id"] if user else ""
    body = await request.json()
    account_id = body.get("account_id", "")
    recipient = body.get("recipient", "")
    if not recipient or "@" not in recipient:
        return JSONResponse({"error": "Valid recipient email required"}, status_code=400)
    from src.tools.email_tool import get_account_by_id, get_default_account, _send_via_smtp
    account = get_account_by_id(uid, account_id) if account_id else get_default_account(uid)
    if not account:
        return JSONResponse({"error": "No account found"}, status_code=404)
    import json as _json
    result = _send_via_smtp(
        account, "Orion Forge — Email Test",
        "This is a test email sent from Orion Forge to verify your email configuration.",
        [recipient],
    )
    return JSONResponse(_json.loads(result))


# ── Identity Profile Config API ────────────────────────────────────
IDENTITY_PROFILE_FILE = _CONFIG_DIR / "identity_profile.json"

def _load_identity_profile() -> dict:
    return _read_json(IDENTITY_PROFILE_FILE, {})

def _save_identity_profile(data: dict):
    _write_json(IDENTITY_PROFILE_FILE, data)

@app.get("/api/tools/memory/identity", response_class=JSONResponse)
async def api_identity_profile_get():
    """Return the current identity FAISS profile."""
    return JSONResponse(_load_identity_profile())

@app.put("/api/tools/memory/identity", response_class=JSONResponse)
async def api_identity_profile_put(request: Request):
    """Save a partial or full identity profile update."""
    body = await request.json()
    profile = _load_identity_profile()
    def _deep_merge(base: dict, updates: dict):
        for k, v in updates.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                _deep_merge(base[k], v)
            else:
                base[k] = v
    _deep_merge(profile, body)
    _save_identity_profile(profile)
    return JSONResponse(profile)


# ── Memory Profile Config API ─────────────────────────────────────
MEMORY_PROFILE_FILE = _CONFIG_DIR / "memory_profile.json"

def _load_memory_profile() -> dict:
    return _read_json(MEMORY_PROFILE_FILE, {})

def _save_memory_profile(data: dict):
    _write_json(MEMORY_PROFILE_FILE, data)

@app.get("/api/tools/memory/profile", response_class=JSONResponse)
async def api_memory_profile_get():
    """Return the current memory profile configuration."""
    return JSONResponse(_load_memory_profile())

@app.put("/api/tools/memory/profile", response_class=JSONResponse)
async def api_memory_profile_put(request: Request):
    """Save a partial or full memory profile update."""
    body = await request.json()
    profile = _load_memory_profile()
    # Deep-merge incoming fields into existing profile
    def _deep_merge(base: dict, updates: dict):
        for k, v in updates.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                _deep_merge(base[k], v)
            else:
                base[k] = v
    _deep_merge(profile, body)

    # ── Enforce hard ceilings on retention_policy.max_total_memories ──
    HARD_MAX_TOTAL = 25_000  # must match types.HARD_MAX_TOTAL_MEMORIES
    rp = profile.get("retention_policy", {})
    mtm = rp.get("max_total_memories", 5000)
    if isinstance(mtm, (int, float)) and mtm != 0 and mtm > HARD_MAX_TOTAL:
        rp["max_total_memories"] = HARD_MAX_TOTAL
    profile["retention_policy"] = rp

    _save_memory_profile(profile)
    return JSONResponse(profile)


# ── Saved Profiles (split: identity vs memory) ────────────────────
_SAVED_IDENTITY_PROFILES_DIR = _CONFIG_DIR / "saved_profiles" / "identity"
_SAVED_IDENTITY_PROFILES_DIR.mkdir(parents=True, exist_ok=True)
_SAVED_MEMORY_PROFILES_DIR = _CONFIG_DIR / "saved_profiles" / "memory"
_SAVED_MEMORY_PROFILES_DIR.mkdir(parents=True, exist_ok=True)

import re as _re  # used by profile sanitisation below

_DEFAULT_PROFILE_STEM = "__default__"


def _seed_default_profile(directory, loader_fn):
    """Ensure a pinned __default__.json exists in *directory*.

    If it already exists but is missing keys present in the
    current live profile (e.g. ``category_policy`` added after
    the initial seed), merge the new keys in so the default
    stays in sync.
    """
    default_path = directory / f"{_DEFAULT_PROFILE_STEM}.json"
    if not default_path.exists():
        data = loader_fn()
        if not data:
            data = {}
        data["_pinned"] = True
        default_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    else:
        # Upgrade: back-fill any top-level keys the live profile has
        # that the saved default is missing.
        live = loader_fn() or {}
        try:
            saved = json.loads(default_path.read_text(encoding="utf-8"))
        except Exception:
            saved = {}
        changed = False
        for key in live:
            if key not in saved:
                saved[key] = live[key]
                changed = True
        if changed:
            saved["_pinned"] = True
            default_path.write_text(json.dumps(saved, indent=2), encoding="utf-8")


_seed_default_profile(_SAVED_IDENTITY_PROFILES_DIR, _load_identity_profile)
_seed_default_profile(_SAVED_MEMORY_PROFILES_DIR, _load_memory_profile)


def _list_profiles_in(directory):
    pinned = []
    regular = []
    for f in sorted(directory.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            entry = {"filename": f.stem, "name": data.get("name", f.stem), "version": data.get("version", ""), "description": data.get("description", ""), "pinned": bool(data.get("_pinned"))}
        except Exception:
            entry = {"filename": f.stem, "name": f.stem, "version": "", "description": "", "pinned": False}
        if entry["pinned"]:
            # Override display name for the default
            if entry["filename"] == _DEFAULT_PROFILE_STEM:
                entry["name"] = "Default"
            pinned.append(entry)
        else:
            regular.append(entry)
    return pinned + regular


# ── Identity saved profiles CRUD ──
@app.get("/api/tools/memory/identity/profiles", response_class=JSONResponse)
async def api_saved_identity_profiles_list():
    return JSONResponse(_list_profiles_in(_SAVED_IDENTITY_PROFILES_DIR))

@app.post("/api/tools/memory/identity/profiles", response_class=JSONResponse)
async def api_saved_identity_profiles_save(request: Request):
    body = await request.json()
    filename = body.get("filename", "").strip()
    if not filename:
        return JSONResponse({"error": "filename required"}, status_code=400)
    safe = _re.sub(r'[^\w\-]', '_', filename)
    profile = body.get("profile") or _load_identity_profile()
    dest = _SAVED_IDENTITY_PROFILES_DIR / f"{safe}.json"
    dest.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return JSONResponse({"ok": True, "filename": safe})

@app.get("/api/tools/memory/identity/profiles/{filename}", response_class=JSONResponse)
async def api_saved_identity_profiles_get(filename: str):
    safe = _re.sub(r'[^\w\-]', '_', filename)
    path = _SAVED_IDENTITY_PROFILES_DIR / f"{safe}.json"
    if not path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))

@app.delete("/api/tools/memory/identity/profiles/{filename}", response_class=JSONResponse)
async def api_saved_identity_profiles_delete(filename: str):
    safe = _re.sub(r'[^\w\-]', '_', filename)
    if safe == _DEFAULT_PROFILE_STEM:
        return JSONResponse({"error": "Cannot delete the default profile"}, status_code=403)
    path = _SAVED_IDENTITY_PROFILES_DIR / f"{safe}.json"
    if path.exists():
        path.unlink()
    return JSONResponse({"ok": True})


# ── Memory vault saved profiles CRUD ──
@app.get("/api/tools/memory/profiles", response_class=JSONResponse)
async def api_saved_memory_profiles_list():
    return JSONResponse(_list_profiles_in(_SAVED_MEMORY_PROFILES_DIR))

@app.post("/api/tools/memory/profiles", response_class=JSONResponse)
async def api_saved_memory_profiles_save(request: Request):
    body = await request.json()
    filename = body.get("filename", "").strip()
    if not filename:
        return JSONResponse({"error": "filename required"}, status_code=400)
    safe = _re.sub(r'[^\w\-]', '_', filename)
    profile = body.get("profile") or _load_memory_profile()
    dest = _SAVED_MEMORY_PROFILES_DIR / f"{safe}.json"
    dest.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return JSONResponse({"ok": True, "filename": safe})

@app.get("/api/tools/memory/profiles/{filename}", response_class=JSONResponse)
async def api_saved_memory_profiles_get(filename: str):
    safe = _re.sub(r'[^\w\-]', '_', filename)
    path = _SAVED_MEMORY_PROFILES_DIR / f"{safe}.json"
    if not path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))

@app.delete("/api/tools/memory/profiles/{filename}", response_class=JSONResponse)
async def api_saved_memory_profiles_delete(filename: str):
    safe = _re.sub(r'[^\w\-]', '_', filename)
    if safe == _DEFAULT_PROFILE_STEM:
        return JSONResponse({"error": "Cannot delete the default profile"}, status_code=403)
    path = _SAVED_MEMORY_PROFILES_DIR / f"{safe}.json"
    if path.exists():
        path.unlink()
    return JSONResponse({"ok": True})


# ── AGI Loop page (preview only — no backend connected) ──────────
AGI_LOOP_FILE = _CONFIG_DIR / "agi_loop.json"

_AGI_LOOP_DEFAULTS = {
    "interval_hours": 0,
    "interval_minutes": 30,
    "max_loops": 5,
    "ticks_per_loop": 15,
    "max_steps_per_tick": 3,
    "stimulus": "",
    "profile": "orion",
    "auto_pause_on_budget": True,
    "auto_pause_on_error_streak": 5,
    "stale_streak_limit": 2,
    "monthly_hard_cap": 20.00,
    "monthly_soft_cap": 16.00,
    "per_session_cap": 2.00,
    "per_tick_cap": 0.10,
    "tiers": [
        {
            "id": "t0", "label": "local_cheap", "enabled": True,
            "connection_id": "", "provider": "ollama",
            "primary_model": "qwen2.5:7b",
            "temperature": 0.6, "max_output_tokens": 2048,
            "max_iterations": 8, "retries_before_escalate": 3,
            "alt_models": [],
            "default_for": "Memory ops, reflection, summarization",
            "cost_per_call": "~$0.00",
        },
        {
            "id": "t1", "label": "local_strong", "enabled": True,
            "connection_id": "", "provider": "ollama",
            "primary_model": "llama3:70b",
            "temperature": 0.5, "max_output_tokens": 4096,
            "max_iterations": 8, "retries_before_escalate": 3,
            "alt_models": [],
            "default_for": "Planning, general reasoning",
            "cost_per_call": "~$0.00",
        },
        {
            "id": "t2", "label": "cheap_cloud", "enabled": True,
            "connection_id": "", "provider": "deepseek",
            "primary_model": "deepseek-chat",
            "temperature": 0.4, "max_output_tokens": 8192,
            "max_iterations": 6, "retries_before_escalate": 2,
            "alt_models": [],
            "default_for": "High-stakes decisions, tool use",
            "cost_per_call": "~$0.001",
        },
        {
            "id": "t3", "label": "expensive_cloud", "enabled": True,
            "connection_id": "", "provider": "openai",
            "primary_model": "gpt-4o",
            "temperature": 0.3, "max_output_tokens": 16384,
            "max_iterations": 4, "retries_before_escalate": 2,
            "alt_models": [],
            "default_for": "Final polish, critical review",
            "cost_per_call": "~$0.01\u20130.10",
        },
        {
            "id": "t4", "label": "code_light", "enabled": True,
            "connection_id": "", "provider": "deepseek",
            "primary_model": "deepseek-chat",
            "temperature": 0.3, "max_output_tokens": 4096,
            "max_iterations": 8, "retries_before_escalate": 3,
            "alt_models": [],
            "default_for": "Light coding \u2014 typos, renames, formatting",
            "cost_per_call": "~$0.001",
        },
        {
            "id": "t5", "label": "code_heavy", "enabled": True,
            "connection_id": "", "provider": "openai",
            "primary_model": "gpt-4o",
            "temperature": 0.2, "max_output_tokens": 16384,
            "max_iterations": 15, "retries_before_escalate": 2,
            "alt_models": [],
            "default_for": "Heavy coding \u2014 implement, debug, refactor",
            "cost_per_call": "~$0.01\u20130.10",
        },
    ],
}

def _load_agi_loop_config() -> dict:
    saved = _read_json(AGI_LOOP_FILE, {})
    merged = {**_AGI_LOOP_DEFAULTS, **saved}
    # Ensure tiers always present
    if "tiers" not in merged:
        merged["tiers"] = _AGI_LOOP_DEFAULTS["tiers"]
    return merged

def _save_agi_loop_config(data: dict):
    _write_json(AGI_LOOP_FILE, data)


@app.get("/agi-loop", response_class=HTMLResponse)
async def page_agi_loop(request: Request):
    agents = _list_agents()
    config = _load_agi_loop_config()
    connections = _load_connections().get("connections", [])
    # Get live loop state
    try:
        from src.tools.agi_loop import get_loop_state
        loop_status = get_loop_state().to_dict()
    except Exception:
        loop_status = {"running": False, "paused": False, "total_ticks": 0}
    router_config = _load_model_router_config()
    # Resolve allowed tools for the configured agent
    try:
        from src.tools.registry import get_allowed_tools
        allowed_tools = get_allowed_tools(config.get("profile", ""))
    except Exception:
        allowed_tools = []
    return templates.TemplateResponse(request, "agi_loop.html", {
        "page": "agi-loop",
        "agents": agents, "config": config,
        "connections": connections,
        "loop_status": loop_status,
        "router_config": router_config,
        "allowed_tools": allowed_tools,
    })


class AGILoopConfigUpdate(BaseModel):
    interval_hours: int = 0
    interval_minutes: int = 30
    max_loops: int = 5
    ticks_per_loop: int = 15
    max_steps_per_tick: int = 3
    stimulus: str = ""
    profile: str = "orion"
    auto_pause_on_budget: bool = True
    auto_pause_on_error_streak: int = 5
    stale_streak_limit: int = 2
    monthly_hard_cap: float = 20.00
    monthly_soft_cap: float = 16.00
    per_session_cap: float = 2.00
    per_tick_cap: float = 0.10
    tiers: list = []


@app.get("/api/agi-loop/config")
async def api_agi_loop_config_get():
    return JSONResponse(_load_agi_loop_config())


@app.post("/api/agi-loop/config")
async def api_agi_loop_config_save(body: AGILoopConfigUpdate):
    data = body.dict()
    _save_agi_loop_config(data)
    return JSONResponse({"ok": True, "config": data})


# ── AGI Loop Runner ───────────────────────────────────────────────

async def _execute_agi_tick(agent: str, stimulus: str, agi_config: dict) -> dict:
    """Execute a single AGI loop tick — send stimulus through the chat pipeline."""
    try:
        conn = _resolve_connection(None, agent)
        if not conn:
            return {"error": "No connection available", "response": "", "cost": 0.0, "tool_calls": []}

        # Use a dedicated AGI loop chat per agent
        chat_id = f"agi-loop-{agent}"
        chat_data = _load_chat(chat_id)
        if not chat_data:
            chat_data = {
                "id": chat_id, "agent": agent,
                "title": f"AGI Loop — {agent}",
                "mode": "agi_loop",
                "messages": [],
                "created": datetime.now(timezone.utc).isoformat(),
                "updated": datetime.now(timezone.utc).isoformat(),
            }
            _save_chat(chat_id, chat_data)
            idx = _load_chat_index()
            idx["chats"].append({
                "id": chat_id, "title": f"AGI Loop — {agent}",
                "agent": agent, "mode": "agi_loop",
                "folder_id": None,
                "created": chat_data["created"],
                "updated": chat_data["updated"],
            })
            _save_chat_index(idx)

        now = datetime.now(timezone.utc).isoformat()
        chat_data["messages"].append({"role": "user", "text": stimulus, "time": now})

        llm_messages, layers, tool_defs = await asyncio.to_thread(
            _build_chat_messages, agent, chat_data["messages"]
        )

        # Model resolution via router
        profile = _load_profile(agent)
        routed_model = None
        routed_provider = None
        task_type = "agi_tick"
        _direct_conn_id = None          # set when task_tier_map uses model:<conn>:<name>
        _router_tier_cfg = None         # resolved TierConfig (for connection override)
        try:
            from src.routing.model_router import ModelRouter
            router = ModelRouter.from_config()
            if router.enabled:
                decision = router.route(stimulus, fallback_task="agi_tick")
                task_type = decision.task_type
                if decision.is_direct_model:
                    routed_model = decision.model
                    _direct_conn_id = decision.direct_conn_id
                elif not decision.fallback:
                    routed_model = decision.model
                    routed_provider = decision.provider
                    _router_tier_cfg = decision  # carries connection_id
                # If nothing matched, use the agi_tick tier
                if not routed_model:
                    from src.routing.model_router import resolve_tier
                    tier = resolve_tier("agi_tick")
                    if tier:
                        routed_model = tier.get("primary_model")
                        routed_provider = tier.get("provider")
        except Exception:
            pass

        model = (
            routed_model
            or profile.get("model", "")
            or (conn["models"][0] if conn.get("models") else "gpt-4o-mini")
        )

        # Connection override for routed tier (or direct-model connection)
        if _direct_conn_id:
            direct_conn = _resolve_connection(_direct_conn_id, agent)
            if direct_conn:
                conn = direct_conn
        elif _router_tier_cfg and getattr(_router_tier_cfg, 'connection_id', ''):
            tier_conn = _resolve_connection(_router_tier_cfg.connection_id, agent)
            if tier_conn:
                conn = tier_conn

        # ── OpenRouter prefix fix ──────────────────────────────────────
        if conn.get("provider") == "openrouter" and model and "/" not in model:
            model = f"openai/{model}"
            log.info("[agi_tick] Auto-prefixed model for OpenRouter: %s", model)

        url = conn["url"].rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if conn.get("api_key"):
            headers["Authorization"] = f"Bearer {conn['api_key']}"

        max_steps = agi_config.get("max_steps_per_tick", 3)
        tool_call_log: list[dict] = []
        total_cost = 0.0

        # Inject runtime context
        try:
            from src.tools.runtime_info import RuntimeInfoTool
            from src.runtime_policy import RuntimePolicy
            policy_cfg = profile.get("policy", {})
            policy = RuntimePolicy(
                max_iterations=policy_cfg.get("max_iterations", 25),
                stasis_mode=policy_cfg.get("stasis_mode", False),
                tool_failure_mode=policy_cfg.get("tool_failure_mode", "continue"),
            )
            RuntimeInfoTool.set_context(profile=profile, policy=policy, execution_mode="agi_loop")
        except Exception:
            pass

        running_messages = list(llm_messages)
        max_tool_calls = agi_config.get("max_tool_calls_per_tick", 15)
        total_tool_calls = 0

        log.info("[agi_tick] agent=%s model=%s provider=%s conn_id=%s url=%s",
                 agent, model, conn.get("provider"), conn.get("id"), url)

        # Swap SearXNG → OpenRouter native web search when using OpenRouter
        tool_defs = _prepare_tools_for_connection(tool_defs, conn)

        response_text = ""
        for step in range(max_steps + 1):
            payload = {
                "model": model,
                "messages": running_messages,
                "temperature": profile.get("temperature", 0.7),
            }
            if tool_defs:
                payload["tools"] = tool_defs

            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(url, json=payload, headers=headers)
                if resp.status_code >= 400:
                    body = resp.text[:1000]
                    log.error("[agi_tick] HTTP %d from %s — model=%s body=%s",
                              resp.status_code, url, model, body)
                resp.raise_for_status()
                data = resp.json()

            choice = data.get("choices", [{}])[0]
            msg = choice.get("message", {})
            finish = choice.get("finish_reason", "stop")
            usage = data.get("usage", {})

            # Cost metering
            try:
                from src.observability.metering import meter_from_raw_usage, log_cost_event
                metering = meter_from_raw_usage(
                    usage, provider=conn.get("provider", "openai"), model=model,
                )
                total_cost += metering.cost.total_cost
                _tool_source = "platform" if conn.get("platform_hosted") else "user"
                log_cost_event(metering, agent=agent, chat_id=chat_id, source=_tool_source)
            except Exception:
                pass

            # Tool calls
            tool_calls = msg.get("tool_calls")
            if (finish == "tool_calls" or tool_calls) and step < max_steps:
                running_messages.append(msg)
                from src.tools.registry import execute_tool
                import json as _json

                for tc in (tool_calls or []):
                    if total_tool_calls >= max_tool_calls:
                        log.warning("[agi_tick] %s hit max_tool_calls (%d) — skipping remaining", agent, max_tool_calls)
                        break
                    fn_name = tc.get("function", {}).get("name", "")
                    fn_args_raw = tc.get("function", {}).get("arguments", "{}")
                    tc_id = tc.get("id", "")
                    try:
                        fn_args = _json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
                    except _json.JSONDecodeError:
                        fn_args = {}

                    total_tool_calls += 1
                    log.info("[agi_tick] %s calling %s(%s) [%d/%d]", agent, fn_name, fn_args, total_tool_calls, max_tool_calls)
                    try:
                        result = execute_tool(fn_name, fn_args, agent_name=agent)
                    except PermissionError as exc:
                        result = f"BLOCKED: {exc}"
                    except Exception as exc:
                        result = f"Error: {exc}"

                    tool_call_log.append({
                        "tool": fn_name, "arguments": fn_args,
                        "result": result[:2000],
                    })
                    running_messages.append({
                        "role": "tool", "tool_call_id": tc_id,
                        "content": result,
                    })
                continue

            # Normal response
            response_text = msg.get("content", "") or ""
            break

        # Save memories
        _extract_and_save_memories(agent, response_text)
        response_text = _strip_memory_tags(response_text)

        # Save to AGI loop chat
        chat_data["messages"].append({
            "role": "assistant", "text": response_text, "time": now,
        })
        chat_data["updated"] = now
        _save_chat(chat_data["id"], chat_data)

        return {
            "response": response_text,
            "tool_calls": tool_call_log,
            "cost": total_cost,
            "model": model,
            "task_type": task_type,
        }

    except Exception as exc:
        log.error("[agi_tick] Tick failed: %s", exc)
        return {"error": str(exc), "response": "", "cost": 0.0, "tool_calls": []}


def _build_tick_narrative(
    tick_result: dict,
    agent: str,
    loop_num: int,
    tick_num: int,
    cost: float,
) -> str:
    """Build a short human-readable narrative from a tick result."""
    parts: list[str] = []

    # Error case
    if tick_result.get("error"):
        parts.append(f"Encountered an error: {str(tick_result['error'])[:500]}")
        return " ".join(parts)

    # Tool usage summary
    tools = tick_result.get("tool_calls", [])
    if tools:
        names = list(dict.fromkeys(tc.get("tool", "unknown") for tc in tools))
        if len(names) == 1:
            parts.append(f"Used {names[0]}.")
        else:
            parts.append(f"Used {', '.join(names[:-1])} and {names[-1]}.")

    # Response excerpt — first 4 sentences or first 600 chars
    response = (tick_result.get("response") or "").strip()
    if response:
        sentences = response.replace("\n", " ").split(". ")
        excerpt = ". ".join(sentences[:4])
        if not excerpt.endswith("."):
            excerpt += "."
        if len(excerpt) > 600:
            excerpt = excerpt[:597] + "..."
        parts.append(excerpt)
    else:
        parts.append("No textual response returned.")

    return " ".join(parts)


# ── Tick staleness detection ──────────────────────────────────────

def _compute_tick_fingerprint(tick_result: dict) -> str:
    """Create a normalized fingerprint from a tick's tool calls + response.

    The fingerprint captures *what* the agent did (tool names + arg keys)
    and a trimmed, lowered version of the response.  Two ticks with the
    same fingerprint are considered duplicates.
    """
    import hashlib

    parts: list[str] = []

    # Tool signature: sorted list of "tool_name:sorted_arg_keys"
    for tc in tick_result.get("tool_calls", []):
        name = tc.get("tool", "")
        arg_keys = sorted(tc.get("arguments", {}).keys())
        # Include arg *values* for small args (detect identical calls)
        arg_vals = []
        for k in arg_keys:
            v = str(tc["arguments"][k])[:80]
            arg_vals.append(f"{k}={v}")
        parts.append(f"{name}({','.join(arg_vals)})")

    # Response text — normalize whitespace, lowercase, first 500 chars
    resp = (tick_result.get("response") or "").strip().lower()
    resp = " ".join(resp.split())[:500]
    parts.append(resp)

    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()


async def _agi_loop_runner():
    """Background coroutine that drives the autonomous AGI loop."""
    from src.tools.agi_loop import get_loop_state

    state = get_loop_state()
    config = _load_agi_loop_config()

    agent = config.get("profile", "elysia")
    stimulus_template = config.get("stimulus", "")
    ticks_per_loop = config.get("ticks_per_loop", 15)
    max_loops = config.get("max_loops", 0)  # 0 = infinite
    interval_secs = config.get("interval_hours", 0) * 3600 + config.get("interval_minutes", 30) * 60

    state.running = True
    state.paused = False
    state.session_cost = 0.0
    state.error_streak = 0
    state.started_at = datetime.now(timezone.utc).isoformat()
    state.stopped_at = None
    state.stop_reason = None

    loop_count = 0

    try:
        while state.running:
            loop_count += 1
            state.current_loop = loop_count

            if max_loops > 0 and loop_count > max_loops:
                state.stop_reason = "max_loops_reached"
                break

            for tick in range(1, ticks_per_loop + 1):
                # Pause gate
                if state.paused:
                    log.info("[agi_loop] Paused — waiting for resume")
                    while state.paused and state.running:
                        await asyncio.sleep(1)
                    if not state.running:
                        break

                if not state.running:
                    break

                state.current_tick = tick
                state.total_ticks += 1

                # Build stimulus
                if stimulus_template:
                    stim = stimulus_template
                else:
                    # Gather last tick summary for progress context
                    _prev_narrative = ""
                    if state.journal_entries:
                        _last_journal = state.journal_entries[-1]
                        _prev_narrative = _last_journal.get("narrative", "")

                    _progress_ctx = ""
                    if _prev_narrative:
                        _progress_ctx = (
                            f"\n\nPREVIOUS TICK SUMMARY: {_prev_narrative}\n"
                            "Do NOT repeat the same action. Build on what was done or move to the next task."
                        )

                    _stale_warning = ""
                    if state.stale_streak > 0:
                        _remaining = config.get("stale_streak_limit", 2) - state.stale_streak
                        _stale_warning = (
                            f"\n\n⚠ REPETITION DETECTED: Your last {state.stale_streak} tick(s) were identical. "
                            f"You will be auto-stopped in {_remaining} more stale tick(s). "
                            "You MUST do something DIFFERENT this tick or call agi_loop(action='request_stop')."
                        )

                    stim = (
                        f"[AGI Loop] Tick {tick}/{ticks_per_loop} — loop {loop_count}."
                        f"{_progress_ctx}{_stale_warning}\n\n"
                        "INSTRUCTIONS:\n"
                        "1. Use the continuation_update tool to READ your continuation notes first "
                        "(mode='replace_section') to see where you left off.\n"
                        "2. Review memory for pending tasks or ideas.\n"
                        "3. Take the most impactful autonomous action available. Use your tools.\n"
                        "4. After acting, use continuation_update (mode='append') to log what you did "
                        "and what to do next tick.\n"
                        "5. If you have anything to report to the operator, use the inbox tool "
                        "(action='send') to message them.\n"
                        "6. Report what you did in your response.\n\n"
                        "IMPORTANT: If you have completed all available tasks, are stuck repeating "
                        "the same actions, or determine that continued execution is unproductive, "
                        "use the agi_loop tool with action='request_stop' and provide a reason. "
                        "Do NOT keep looping if there is nothing meaningful left to do."
                    )

                # Execute
                tick_result = await _execute_agi_tick(agent, stim, config)

                tick_cost = tick_result.get("cost", 0.0)
                state.session_cost += tick_cost
                state.total_cost += tick_cost

                if tick_result.get("error"):
                    state.error_streak += 1
                    state.last_error = tick_result["error"]
                else:
                    state.error_streak = 0

                state.log_tick({
                    "loop": loop_count,
                    "tick": tick,
                    "time": datetime.now(timezone.utc).isoformat(),
                    "agent": agent,
                    "stimulus": stim[:2000],
                    "response": tick_result.get("response", "")[:5000],
                    "tool_calls": tick_result.get("tool_calls", []),
                    "cost": tick_cost,
                    "error": tick_result.get("error"),
                    "model": tick_result.get("model", ""),
                    "task_type": tick_result.get("task_type", ""),
                })

                # ── Journal narrative ─────────────────────────────
                _journal_narrative = _build_tick_narrative(
                    tick_result, agent, loop_count, tick, tick_cost,
                )
                state.log_journal({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "loop": loop_count,
                    "tick": tick,
                    "agent": agent,
                    "model": tick_result.get("model", ""),
                    "task_type": tick_result.get("task_type", ""),
                    "narrative": _journal_narrative,
                    "full_response": tick_result.get("response", "")[:5000],
                    "full_error": str(tick_result.get("error") or "")[:2000],
                    "tool_calls": tick_result.get("tool_calls", []),
                    "tools_used": [tc.get("tool", "") for tc in tick_result.get("tool_calls", [])],
                    "cost": round(tick_cost, 6),
                    "had_error": bool(tick_result.get("error")),
                })

                # ── Staleness detection ───────────────────────────
                if not tick_result.get("error"):
                    _fp = _compute_tick_fingerprint(tick_result)
                    _was_stale = state.record_fingerprint(_fp)
                    stale_limit = config.get("stale_streak_limit", 2)
                    if _was_stale:
                        log.warning("[agi_loop] Stale tick detected (streak %d/%d)",
                                    state.stale_streak, stale_limit)
                    if stale_limit and state.stale_streak >= stale_limit:
                        state.running = False
                        state.stop_reason = "stale_streak_exceeded"
                        log.warning("[agi_loop] Stale streak hit %d — auto-stopping to prevent repetition",
                                    state.stale_streak)
                        break

                # Budget checks
                if config.get("auto_pause_on_budget"):
                    if config.get("per_tick_cap") and tick_cost > config["per_tick_cap"]:
                        state.paused = True
                        state.stop_reason = "per_tick_budget_exceeded"
                        log.warning("[agi_loop] Per-tick budget exceeded ($%.4f > $%.4f)",
                                    tick_cost, config["per_tick_cap"])
                        break
                    if config.get("per_session_cap") and state.session_cost > config["per_session_cap"]:
                        state.running = False
                        state.stop_reason = "session_budget_exceeded"
                        log.warning("[agi_loop] Session budget exceeded ($%.4f > $%.4f)",
                                    state.session_cost, config["per_session_cap"])
                        break

                # Error streak check
                streak_limit = config.get("auto_pause_on_error_streak", 5)
                if streak_limit and state.error_streak >= streak_limit:
                    state.running = False
                    state.stop_reason = "error_streak_exceeded"
                    log.warning("[agi_loop] Error streak hit %d — stopping", state.error_streak)
                    break

            # Wait between loops
            if state.running and not state.paused and interval_secs > 0:
                log.info("[agi_loop] Waiting %d seconds before next loop", interval_secs)
                for _ in range(int(interval_secs)):
                    if not state.running:
                        break
                    await asyncio.sleep(1)

    except asyncio.CancelledError:
        state.stop_reason = "cancelled"
    except Exception as exc:
        state.stop_reason = f"exception: {exc}"
        log.error("[agi_loop] Runner crashed: %s", exc)
    finally:
        state.running = False
        state.stopped_at = datetime.now(timezone.utc).isoformat()
        log.info("[agi_loop] Stopped — reason: %s, total ticks: %d, cost: $%.4f",
                 state.stop_reason, state.total_ticks, state.total_cost)


@app.post("/api/agi-loop/start")
async def api_agi_loop_start():
    from src.tools.agi_loop import get_loop_state
    state = get_loop_state()
    if state.running:
        return JSONResponse({"ok": False, "reason": "Already running"}, 409)
    state.reset()
    state.task = asyncio.create_task(_agi_loop_runner())
    return JSONResponse({"ok": True, "message": "AGI loop started"})


@app.post("/api/agi-loop/stop")
async def api_agi_loop_stop():
    from src.tools.agi_loop import get_loop_state
    state = get_loop_state()
    if not state.running:
        return JSONResponse({"ok": False, "reason": "Not running"}, 409)
    state.running = False
    if state.task and not state.task.done():
        state.task.cancel()
    return JSONResponse({"ok": True, "message": "Stop signal sent"})


@app.post("/api/agi-loop/pause")
async def api_agi_loop_pause():
    from src.tools.agi_loop import get_loop_state
    state = get_loop_state()
    if not state.running:
        return JSONResponse({"ok": False, "reason": "Not running"}, 409)
    state.paused = True
    return JSONResponse({"ok": True, "message": "Pause requested"})


@app.post("/api/agi-loop/resume")
async def api_agi_loop_resume():
    from src.tools.agi_loop import get_loop_state
    state = get_loop_state()
    if not state.paused:
        return JSONResponse({"ok": False, "reason": "Not paused"}, 409)
    state.paused = False
    return JSONResponse({"ok": True, "message": "Resumed"})


@app.get("/api/agi-loop/status")
async def api_agi_loop_status():
    from src.tools.agi_loop import get_loop_state
    return JSONResponse(get_loop_state().to_dict())


@app.get("/api/agi-loop/history")
async def api_agi_loop_history(limit: int = Query(50)):
    from src.tools.agi_loop import get_loop_state
    state = get_loop_state()
    # Reload from disk if memory is empty (e.g. after process restart)
    if not state.tick_history:
        state.load_history_from_disk()
    return JSONResponse({
        "ticks": state.tick_history[-limit:],
        "total": len(state.tick_history),
    })


@app.get("/api/agi-loop/journal")
async def api_agi_loop_journal(limit: int = Query(500)):
    """Return recent narrative journal entries."""
    from src.tools.agi_loop import get_loop_state
    state = get_loop_state()
    # Reload from disk if memory is empty (e.g. after process restart)
    if not state.journal_entries:
        state.load_journal_from_disk()
    entries = state.journal_entries[-limit:]
    return JSONResponse({
        "entries": entries,
        "total": len(state.journal_entries),
    })


@app.delete("/api/agi-loop/journal")
async def api_agi_loop_journal_clear():
    """Clear all journal entries."""
    from src.tools.agi_loop import get_loop_state
    state = get_loop_state()
    state.clear_journal()
    return JSONResponse({"ok": True, "message": "Journal cleared"})


# ── Inbox API ────────────────────────────────────────────────────

@app.get("/api/inbox/list")
async def api_inbox_list():
    """Return all inbox entries."""
    from src.tools.inbox import _read_all
    entries = _read_all()
    return JSONResponse({"entries": entries, "total": len(entries)})


@app.post("/api/inbox/send")
async def api_inbox_send(request: Request):
    """Send a message or add a task via the inbox tool."""
    from src.tools.inbox import InboxTool
    body = await request.json()
    result = InboxTool.execute(body)
    ok = not result.startswith("Error")
    return JSONResponse({"ok": ok, "result": result, "error": None if ok else result})


@app.post("/api/inbox/ack")
async def api_inbox_ack(request: Request):
    """Acknowledge an inbox entry by ID."""
    from src.tools.inbox import InboxTool
    body = await request.json()
    task_id = body.get("task_id", "")
    result = InboxTool.execute({"action": "ack", "task_id": task_id})
    ok = not result.startswith("Error")
    return JSONResponse({"ok": ok, "result": result, "error": None if ok else result})


@app.post("/api/inbox/delete")
async def api_inbox_delete(request: Request):
    """Delete an inbox entry by ID."""
    from src.tools.inbox import _read_all, _write_all
    body = await request.json()
    entry_id = body.get("id", "")
    entries = _read_all()
    new_entries = [e for e in entries if e.get("id") != entry_id]
    if len(new_entries) == len(entries):
        return JSONResponse({"ok": False, "error": f"Entry '{entry_id}' not found"})
    _write_all(new_entries)
    return JSONResponse({"ok": True})


# ── Model Router config (delegated to src.routing) ───────────────────────
from src.routing.model_router import (
    load_router_config  as _load_model_router_config,
    save_router_config  as _save_model_router_config,
    CONFIG_DEFAULTS     as _MODEL_ROUTER_DEFAULTS,
    MODEL_ROUTER_FILE,
)



@app.get("/api/model-router/config")
async def api_model_router_config_get():
    return JSONResponse(_load_model_router_config())


@app.post("/api/model-router/config")
async def api_model_router_config_save(request: Request):
    data = await request.json()
    _save_model_router_config(data)
    return JSONResponse({"ok": True, "config": data})


@app.post("/api/model-router/reset")
async def api_model_router_reset():
    _save_model_router_config(_MODEL_ROUTER_DEFAULTS)
    return JSONResponse({"ok": True, "config": _MODEL_ROUTER_DEFAULTS})


# ── Budget Tracker API ───────────────────────────────────────────
@app.get("/api/model-router/budget")
async def api_budget_get():
    """Return current budget/spending summary."""
    try:
        from src.routing.budget_tracker import BudgetTracker
        tracker = BudgetTracker()
        return JSONResponse(tracker.get_summary())
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/model-router/budget/reset-session")
async def api_budget_reset_session():
    """Reset the session-level spending counter."""
    try:
        from src.routing.budget_tracker import BudgetTracker
        tracker = BudgetTracker()
        tracker.reset_session()
        return JSONResponse({"ok": True, "summary": tracker.get_summary()})
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


# ── Model Router presets (save-states) ───────────────────────────
_ROUTER_PRESETS_DIR = _CONFIG_DIR / "router_presets"

def _ensure_router_presets_dir():
    _ROUTER_PRESETS_DIR.mkdir(parents=True, exist_ok=True)

def _list_router_presets() -> list:
    _ensure_router_presets_dir()
    presets = []
    for f in sorted(_ROUTER_PRESETS_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            presets.append({
                "name": data.get("name", f.stem),
                "description": data.get("description", ""),
                "filename": f.stem,
                "created": data.get("created", ""),
            })
        except Exception:
            pass
    return presets

@app.get("/api/model-router/presets")
async def api_model_router_presets_list():
    return JSONResponse({"presets": _list_router_presets()})

@app.post("/api/model-router/presets")
async def api_model_router_presets_save(request: Request):
    body = await request.json()
    name = (body.get("name") or "").strip()
    if not name:
        return JSONResponse({"ok": False, "error": "Name is required"}, status_code=400)
    description = (body.get("description") or "").strip()
    config = body.get("config") or _load_model_router_config()
    _ensure_router_presets_dir()
    import re as _re, datetime as _dt
    safe = _re.sub(r'[^\w\-. ]', '', name).strip().replace(' ', '_')[:80] or "preset"
    payload = {
        "name": name,
        "description": description,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "config": config,
    }
    dest = _ROUTER_PRESETS_DIR / f"{safe}.json"
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return JSONResponse({"ok": True, "filename": safe})

@app.post("/api/model-router/presets/{filename}/load")
async def api_model_router_presets_load(filename: str):
    _ensure_router_presets_dir()
    f = _ROUTER_PRESETS_DIR / f"{filename}.json"
    if not f.exists():
        return JSONResponse({"ok": False, "error": "Preset not found"}, status_code=404)
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        config = data.get("config", {})
        _save_model_router_config(config)
        return JSONResponse({"ok": True, "config": config, "name": data.get("name", filename)})
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

@app.delete("/api/model-router/presets/{filename}")
async def api_model_router_presets_delete(filename: str):
    _ensure_router_presets_dir()
    f = _ROUTER_PRESETS_DIR / f"{filename}.json"
    if f.exists():
        f.unlink()
    return JSONResponse({"ok": True})


# ── About / Wiki page ────────────────────────────────────────────
ABOUT_FILE = _CONFIG_DIR / "about.json"

def _load_about() -> dict:
    return _read_json(ABOUT_FILE, {"text": ""})

def _save_about(data: dict):
    _write_json(ABOUT_FILE, data)

# README key → absolute path for each wiki article
_WIKI_README_MAP = {
    "root":               Path(__file__).resolve().parent.parent.parent / "README.md",
    "orion-ui-standalone": Path(__file__).resolve().parent.parent / "README.md",
    "engine":             Path(__file__).resolve().parent.parent.parent / "engine" / "README.md",
    "web":                Path(__file__).resolve().parent / "README.md",
    "user-data":           Path(__file__).resolve().parent / "USER_DATA.md",
    "data":               Path(__file__).resolve().parent.parent / "data" / "README.md",
    "src":                Path(__file__).resolve().parent.parent / "src" / "README.md",
    "src/memory":         Path(__file__).resolve().parent.parent / "src" / "memory" / "README.md",
    "src/llm_client":     Path(__file__).resolve().parent.parent / "src" / "llm_client" / "README.md",
    "src/tools":          Path(__file__).resolve().parent.parent / "src" / "tools" / "README.md",
    "src/directives":     Path(__file__).resolve().parent.parent / "src" / "directives" / "README.md",
    "src/governance":     Path(__file__).resolve().parent.parent / "src" / "governance" / "README.md",
    "src/policy":         Path(__file__).resolve().parent.parent / "src" / "policy" / "README.md",
    "src/observability":  Path(__file__).resolve().parent.parent / "src" / "observability" / "README.md",
    "src/storage":        Path(__file__).resolve().parent.parent / "src" / "storage" / "README.md",
    "config":             Path(__file__).resolve().parent.parent / "config" / "README.md",
    "profiles":           Path(__file__).resolve().parent.parent / "profiles" / "README.md",
    "prompts":            Path(__file__).resolve().parent.parent / "prompts" / "README.md",
    "directives-user":    Path(__file__).resolve().parent.parent / "directives" / "README.md",
    "notes":              Path(__file__).resolve().parent.parent / "notes" / "README.md",
    "tests":              Path(__file__).resolve().parent.parent / "tests" / "README.md",
    "scripts":            Path(__file__).resolve().parent.parent / "scripts" / "README.md",
    "soul-scripts":        Path(__file__).resolve().parent.parent / "data" / "orion" / "SOUL_SCRIPTS.md",
}

def _load_wiki_articles() -> dict:
    """Load all README.md files into a {key: markdown_text} dict."""
    articles = {}
    for key, path in _WIKI_README_MAP.items():
        try:
            if path.exists():
                articles[key] = path.read_text(encoding="utf-8")
        except Exception:
            pass
    return articles

@app.get("/about", response_class=HTMLResponse)
async def page_about(request: Request):
    about = _load_about()
    wiki_articles = _load_wiki_articles()
    return templates.TemplateResponse(request, "about.html", {
        "page": "about",
        "about_text": about.get("text", ""),
        "wiki_articles": wiki_articles,
    })

class AboutUpdate(BaseModel):
    text: str

@app.post("/api/about")
async def api_about_save(body: AboutUpdate):
    _save_about({"text": body.text})
    return JSONResponse({"ok": True})


# ═══════════════════════════════════════════════════════════════════
#  CHAT API
# ═══════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    agent: str
    stimulus: str
    mode: str = "chat"               # "chat" or "burst"
    burst_ticks: int = 3
    max_steps: int = 10
    connection_id: Optional[str] = None
    model_override: Optional[str] = None
    chat_id: Optional[str] = None
    images: Optional[list] = None    # list of image URLs to include


@app.post("/api/chat/upload")
async def api_chat_upload(file: UploadFile = File(...)):
    """Upload a file for use in chat (images, text docs)."""
    allowed = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".avif",
               ".pdf", ".txt", ".md", ".csv", ".json", ".yaml", ".yml"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed:
        return JSONResponse({"error": f"File type {ext} not allowed"}, 400)
    safe_name = f"chat_{uuid.uuid4().hex[:8]}{ext}"
    dest = _UPLOADS_DIR / safe_name
    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        return JSONResponse({"error": "File too large (max 20MB)"}, 400)
    with open(dest, "wb") as f:
        f.write(content)
    return JSONResponse({"url": f"/uploads/{safe_name}", "name": file.filename, "type": file.content_type})


class FolderCreate(BaseModel):
    name: str

@app.post("/api/chat/send")
async def api_chat_send(req: ChatRequest, request: Request):
    user = getattr(request.state, "user", None)
    if not _can_access_agent(req.agent, _get_user_id(request)):
        return JSONResponse({"error": "Agent not available"}, status_code=403)
    conn = _resolve_connection(req.connection_id, req.agent)
    if not conn:
        return JSONResponse({"error": "No API connection available. Add one in Settings."}, 400)

    chat_data = _load_chat(req.chat_id) if req.chat_id else None
    if req.chat_id and not chat_data:
        return JSONResponse({"error": "Chat not found"}, 404)
    if not chat_data:
        chat_data = _create_new_chat(req.agent)

    now = datetime.now(timezone.utc).isoformat()
    chat_data["messages"].append({"role": "user", "text": req.stimulus, "time": now})
    chat_data["updated"] = now

    # Build prompt with all identity layers (offload to thread — FAISS encode is CPU-bound)
    llm_messages, layers, tool_defs = await asyncio.to_thread(
        _build_chat_messages, req.agent, chat_data["messages"]
    )

    # Resolve model (with model router task-tier mapping)
    profile = _load_profile(req.agent)
    agent_cfg = _get_agent_config(req.agent)

    # ── Model router: classify task → resolve tier/model ──
    routed_model = None
    routed_provider = None
    task_type = "general"
    _direct_conn_id = None
    _router_decision = None
    try:
        from src.routing.model_router import ModelRouter
        router = ModelRouter.from_config()
        if router.enabled and not req.model_override:
            decision = router.route(req.stimulus)
            task_type = decision.task_type
            _router_decision = decision
            if decision.is_direct_model:
                routed_model = decision.model
                _direct_conn_id = decision.direct_conn_id
            elif not decision.fallback:
                routed_model = decision.model
                routed_provider = decision.provider
    except Exception as exc:
        log.warning("[router] Model router failed: %s", exc)

    model = (
        req.model_override
        or routed_model
        or agent_cfg.get("model")
        or profile.get("model", "")
        or (conn["models"][0] if conn.get("models") else "gpt-4o-mini")
    )

    # If the router chose a direct-model connection, switch to it
    if _direct_conn_id and not req.model_override:
        direct_conn = _resolve_connection(_direct_conn_id, req.agent)
        if direct_conn:
            conn = direct_conn
            log.info("[router] Switched connection to direct model '%s' (conn=%s)", model, _direct_conn_id)
    # If the router chose a tier with its own connection, switch to it
    elif _router_decision and _router_decision.connection_id and not req.model_override:
        tier_conn = _resolve_connection(_router_decision.connection_id, req.agent)
        if tier_conn:
            conn = tier_conn
            log.info("[router] Switched connection to tier '%s'", _router_decision.tier_name)

    # ── OpenRouter prefix fix ──────────────────────────────────────
    if conn.get("provider") == "openrouter" and model and "/" not in model:
        model = f"openai/{model}"
        log.info("[send] Auto-prefixed model for OpenRouter: %s", model)

    # Call LLM API (with tool-call loop)
    url = conn["url"].rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if conn.get("api_key"):
        headers["Authorization"] = f"Bearer {conn['api_key']}"

    log.info("[send] agent=%s model=%s conn=%s", req.agent, model, conn.get("id"))

    MAX_TOOL_ROUNDS = 10          # safety cap
    tool_call_log: list[dict] = []  # track every tool invocation for the UI
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    cost_data = {}
    _credits_deducted = 0           # cumulative credits deducted across rounds
    _credits_balance = -1           # user's credit balance after deductions (-1 = not applicable)

    # ── Pre-flight credit check for platform-hosted keys ──
    if conn.get("platform_hosted") and user:
        _preflight_balance = get_user_credits(user["id"])
        if _preflight_balance <= 0:
            return JSONResponse({
                "error": "Insufficient credits. Purchase more in the Store.",
                "credits_balance": 0,
                "redirect": "/store",
            }, status_code=402)

    # ── Inject runtime context for runtime_info tool ──
    try:
        from src.tools.runtime_info import RuntimeInfoTool
        from src.runtime_policy import RuntimePolicy
        policy_cfg = profile.get("policy", {})
        policy = RuntimePolicy(
            max_iterations=policy_cfg.get("max_iterations", 25),
            stasis_mode=policy_cfg.get("stasis_mode", False),
            tool_failure_mode=policy_cfg.get("tool_failure_mode", "continue"),
        )
        RuntimeInfoTool.set_context(profile=profile, policy=policy, execution_mode="interactive")
    except Exception as exc:
        log.warning("[runtime_info] Failed to set context: %s", exc)

    running_messages = list(llm_messages)  # mutable copy for the loop

    # Swap SearXNG → OpenRouter native web search when using OpenRouter
    tool_defs = _prepare_tools_for_connection(tool_defs, conn)

    for _round in range(MAX_TOOL_ROUNDS + 1):
        try:
            payload = {
                "model": model,
                "messages": running_messages,
                "temperature": profile.get("temperature", 0.7),
            }
            if tool_defs:
                payload["tools"] = tool_defs
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            return JSONResponse({"error": f"API {exc.response.status_code}: {exc.response.text[:200]}"}, 502)
        except Exception as exc:
            return JSONResponse({"error": f"Request failed: {exc}"}, 502)

        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        finish = choice.get("finish_reason", "stop")
        usage = data.get("usage", {})

        # Accumulate token usage across rounds
        for k in total_usage:
            total_usage[k] += usage.get(k, 0)

        # ── Cost metering (every round) ──
        provider = conn.get("provider", "openai")
        _cost_source = "platform" if conn.get("platform_hosted") else "user"
        try:
            from src.observability.metering import meter_from_raw_usage, log_cost_event
            metering = meter_from_raw_usage(usage, provider=provider, model=model)
            cost_data = metering.cost.to_dict()
            log_cost_event(metering, agent=req.agent, chat_id=chat_data["id"], source=_cost_source)
        except Exception as exc:
            log.warning("[metering] cost computation failed: %s", exc)

        # ── Credit deduction for platform-hosted LLM keys ──
        # Charge whenever tokens were used — even if the model is missing from
        # pricing.yaml (estimate_llm_credit_cost_safe applies a fallback rate so
        # unpriced platform models are never billed as free).
        if conn.get("platform_hosted") and user and total_usage.get("total_tokens", 0) > 0:
            try:
                credit_cost = estimate_llm_credit_cost_safe(
                    cost_data.get("total_cost", 0), total_usage.get("total_tokens", 0)
                )
                if credit_cost > 0:
                    balance = get_user_credits(user["id"])
                    if balance < credit_cost:
                        # Drain remaining balance so the next pre-flight (<= 0)
                        # blocks further calls — prevents unlimited free calls
                        # from users who can't cover the actual cost.
                        if balance > 0:
                            deduct_user_credits(user["id"], balance, f"llm:{model}:shortfall")
                            _credits_deducted += balance
                        _credits_balance = 0
                        return JSONResponse({
                            "error": "Insufficient credits for LLM usage",
                            "credits_needed": credit_cost,
                            "credits_balance": 0,
                            "redirect": "/store",
                        }, status_code=402)
                    deduct_result = deduct_user_credits(user["id"], credit_cost, f"llm:{model}:{total_usage.get('total_tokens', 0)}tok")
                    # Track cumulative credit info for the response
                    _credits_deducted += credit_cost
                    _credits_balance = deduct_result.get("balance", 0)
            except Exception as exc:
                log.warning("[credits] LLM credit deduction failed: %s", exc)

        # ── If the LLM wants to call tools ──
        tool_calls = msg.get("tool_calls")
        if finish == "tool_calls" or tool_calls:
            # Append the assistant message WITH tool_calls to the running context
            running_messages.append(msg)

            from src.tools.registry import execute_tool
            import json as _json

            for tc in (tool_calls or []):
                fn_name = tc.get("function", {}).get("name", "")
                fn_args_raw = tc.get("function", {}).get("arguments", "{}")
                tc_id = tc.get("id", "")

                # Parse arguments
                try:
                    fn_args = _json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
                except _json.JSONDecodeError:
                    fn_args = {}

                # Execute (authorization is enforced inside execute_tool)
                log.info("[tools] Round %d — %s calling %s(%s)", _round + 1, req.agent, fn_name, fn_args)
                try:
                    _user_id = user["id"] if user else ""
                    result = execute_tool(fn_name, fn_args, agent_name=req.agent, user_id=_user_id)
                except PermissionError as exc:
                    result = f"BLOCKED: {exc}"
                    log.warning("[tools] %s blocked for %s: %s", fn_name, req.agent, exc)
                except Exception as exc:
                    result = f"Error: {exc}"
                    log.error("[tools] %s failed: %s", fn_name, exc)

                # Log for UI
                tool_call_log.append({
                    "round": _round + 1,
                    "tool": fn_name,
                    "arguments": fn_args,
                    "result": result[:500],
                })

                # Append tool result message for the next round
                running_messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": result,
                })

            # Continue the loop — LLM will see the tool results
            continue

        # ── Normal text response — done ──
        break

    raw_response = msg.get("content", "") or ""

    # Extract and save any [MEMORY_SAVE: ...] tags to the vault
    saved_memories = _extract_and_save_memories(req.agent, raw_response)
    # Strip memory tags from the text shown to the user
    response_text = _strip_memory_tags(raw_response)

    # ── Image generation: detect [IMAGE_GEN: ...] tag ──
    generated_image = None
    img_match = re.search(r'\[IMAGE_GEN:\s*(.+?)\]', response_text, re.DOTALL)
    if img_match:
        img_prompt = img_match.group(1).strip()
        response_text = response_text[:img_match.start()] + response_text[img_match.end():]
        response_text = response_text.strip()
        try:
            settings = _load_settings()
            img_cfg = settings.get("image", {})
            provider = img_cfg.get("preferred", "none")
            if provider and provider != "none":
                # Image generation always uses platform keys — charge credits.
                _img_credits = estimate_image_credit_cost(provider) if user else 0
                if user and _img_credits > 0 and get_user_credits(user["id"]) < _img_credits:
                    response_text += (
                        f"\n\n*Image generation skipped — needs {_img_credits} credits. "
                        f"Add more in the Store.*"
                    )
                else:
                    result = await _generate_image(provider, img_prompt, img_cfg, settings)
                    if "error" not in result:
                        generated_image = {**result, "prompt": img_prompt}
                        if user and _img_credits > 0:
                            try:
                                _r = deduct_user_credits(user["id"], _img_credits, f"image:{provider}")
                                _credits_deducted += _img_credits
                                _credits_balance = _r.get("balance", _credits_balance)
                            except Exception as exc:
                                log.warning("[credits] image credit deduction failed: %s", exc)
                    else:
                        log.warning("[image-gen] %s", result["error"])
                        response_text += f"\n\n*Image generation failed: {result['error']}*"
        except Exception as exc:
            log.warning("[image-gen] Image generation failed: %s", exc)
            response_text += f"\n\n*Image generation failed: {exc}*"

    # ── Video generation: detect [VIDEO_GEN: ...] tag, with user-intent fallback ──
    generated_video = None
    vid_prompt = None
    vid_match = re.search(r'\[VIDEO_GEN:\s*(.+?)\]', response_text, re.DOTALL)
    if vid_match:
        vid_prompt = vid_match.group(1).strip()
        response_text = response_text[:vid_match.start()] + response_text[vid_match.end():]
        response_text = response_text.strip()
    else:
        p = (req.stimulus or "").strip()
        p_low = p.lower()
        if p and any(k in p_low for k in ("video", "loop", "looping", "animate", "animation", "live wallpaper", "background")):
            vid_prompt = p

    if vid_prompt:
        try:
            settings = _load_settings()
            vid_cfg = settings.get("video", {})
            vid_provider = vid_cfg.get("preferred", "none")
            if vid_provider and vid_provider != "none":
                # Veo video is billed per second — pre-check balance, then charge.
                _vid_secs = vid_cfg.get("duration_seconds", 8)
                _vid_credits = estimate_video_credit_cost(vid_provider, _vid_secs) if user else 0
                if user and _vid_credits > 0 and get_user_credits(user["id"]) < _vid_credits:
                    generated_video = {
                        "error": f"Not enough credits for video ({_vid_credits} needed).",
                        "prompt": vid_prompt,
                    }
                    response_text += (
                        f"\n\n*Video generation skipped — needs {_vid_credits} credits. "
                        f"Add more in the Store.*"
                    )
                else:
                    result = await _generate_video(
                        vid_provider, vid_prompt, vid_cfg, settings,
                        save_dir=_VIDEO_UPLOADS_DIR,
                    )
                    generated_video = {**result, "prompt": vid_prompt}
                    if "error" in result:
                        log.warning("[video-gen] %s", result["error"])
                        response_text += f"\n\n*Video generation failed: {result['error']}*"
                    elif user and _vid_credits > 0:
                        try:
                            _r = deduct_user_credits(user["id"], _vid_credits, f"video:{vid_provider}:{int(_vid_secs)}s")
                            _credits_deducted += _vid_credits
                            _credits_balance = _r.get("balance", _credits_balance)
                        except Exception as exc:
                            log.warning("[credits] video credit deduction failed: %s", exc)
            else:
                generated_video = {
                    "error": "Video generation is disabled. Enable a Veo model in Settings -> Video Generation.",
                    "prompt": vid_prompt,
                }
                response_text += "\n\n*Video generation is disabled. Enable a Veo model in Settings -> Video Generation.*"
        except Exception as exc:
            log.warning("[video-gen] Video generation failed: %s", exc)
            generated_video = {"error": str(exc), "prompt": vid_prompt}
            response_text += f"\n\n*Video generation failed: {exc}*"

    # Add tool layer to metadata
    _tier_label = getattr(_router_decision, "tier_name", None) if _router_decision else None
    layers["tools"]["calls"] = tool_call_log
    layers["router"] = {
        "task_type": task_type,
        "routed_model": routed_model,
        "routed_provider": routed_provider,
        "tier": _tier_label,
    }

    chat_data["messages"].append({
        "role": "assistant", "text": response_text, "time": now,
        "usage": total_usage,
        "data": {
            "agent": req.agent, "model": model,
            "usage": total_usage, "cost": cost_data,
            "tool_calls": tool_call_log,
            "generated_image": generated_image,
            "generated_video": generated_video,
            "task_type": task_type,
            "router_tier": _tier_label,
        },
        "layers": layers,
    })
    _save_chat(chat_data["id"], chat_data)

    idx = _load_chat_index()
    for c in idx["chats"]:
        if c["id"] == chat_data["id"]:
            c["updated"] = now
            break
    _save_chat_index(idx)

    # Build credit usage info for the response
    _credit_info = {}
    if conn.get("platform_hosted") and _credits_deducted > 0:
        _credit_info = {
            "credits_deducted": _credits_deducted,
            "credits_balance": _credits_balance if _credits_balance >= 0 else (get_user_credits(user["id"]) if user else 0),
            "platform_hosted": True,
            "markup": LLM_MARKUP_MULTIPLIER,
        }
    elif user:
        _credit_info = {
            "credits_deducted": 0,
            "credits_balance": get_user_credits(user["id"]),
            "platform_hosted": bool(conn.get("platform_hosted")),
            "markup": 0,
        }

    return {
        "response": response_text, "chat_id": chat_data["id"],
        "model": model, "usage": total_usage, "cost": cost_data, "layers": layers,
        "saved_memories": saved_memories,
        "tool_calls": tool_call_log,
        "generated_image": generated_image,
        "generated_video": generated_video,
        "task_type": task_type,
        "router_tier": _tier_label,
        "credits": _credit_info,
    }


# ═══════════════════════════════════════════════════════════════════
#  STREAMING CHAT — SSE endpoint with real-time thinking display
# ═══════════════════════════════════════════════════════════════════

class _ThinkTagParser:
    """State machine that detects <think>...</think> tags in a stream.
    Handles tags split across chunks via an internal buffer.
    Returns a list of (event_type, text) tuples per feed() call.
    """
    def __init__(self):
        self.in_think = False
        self._buf = ""

    def feed(self, text: str) -> list[tuple[str, str]]:
        self._buf += text
        events: list[tuple[str, str]] = []
        while self._buf:
            if self.in_think:
                idx = self._buf.find("</think>")
                if idx == -1:
                    # Hold back last 8 chars in case of partial tag
                    safe = self._buf[: max(0, len(self._buf) - 8)]
                    if safe:
                        events.append(("thinking", safe))
                        self._buf = self._buf[len(safe):]
                    break
                if idx > 0:
                    events.append(("thinking", self._buf[:idx]))
                events.append(("thinking_done", ""))
                self.in_think = False
                self._buf = self._buf[idx + 8:]
            else:
                idx = self._buf.find("<think>")
                if idx == -1:
                    safe = self._buf[: max(0, len(self._buf) - 7)]
                    if safe:
                        events.append(("text", safe))
                        self._buf = self._buf[len(safe):]
                    break
                if idx > 0:
                    events.append(("text", self._buf[:idx]))
                events.append(("thinking_start", ""))
                self.in_think = True
                self._buf = self._buf[idx + 7:]
        return events

    def flush(self) -> list[tuple[str, str]]:
        events: list[tuple[str, str]] = []
        if self._buf:
            events.append(("thinking" if self.in_think else "text", self._buf))
            self._buf = ""
        if self.in_think:
            events.append(("thinking_done", ""))
            self.in_think = False
        return events


def _sse(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/chat/stream")
async def api_chat_stream(req: ChatRequest, request: Request):
    """Streaming chat endpoint — returns Server-Sent Events with real-time
    thinking display and incremental text."""
    user = getattr(request.state, "user", None)
    if not _can_access_agent(req.agent, _get_user_id(request)):
        return JSONResponse({"error": "Agent not available"}, status_code=403)
    conn = _resolve_connection(req.connection_id, req.agent)
    if not conn:
        return JSONResponse({"error": "No API connection available. Add one in Settings."}, 400)

    chat_data = _load_chat(req.chat_id) if req.chat_id else None
    if req.chat_id and not chat_data:
        return JSONResponse({"error": "Chat not found"}, 404)
    if not chat_data:
        chat_data = _create_new_chat(req.agent)

    return StreamingResponse(
        _stream_chat_generator(req, request, user, conn, chat_data),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _stream_chat_generator(req: ChatRequest, request: Request, user, conn, chat_data):
    """Async generator that yields SSE events for a streaming chat."""
    now = datetime.now(timezone.utc).isoformat()
    chat_data["messages"].append({"role": "user", "text": req.stimulus, "time": now})
    chat_data["updated"] = now

    llm_messages, layers, tool_defs = await asyncio.to_thread(
        _build_chat_messages, req.agent, chat_data["messages"]
    )

    profile = _load_profile(req.agent)
    agent_cfg = _get_agent_config(req.agent)

    # ── Model router: classify task → resolve tier/model ──
    routed_model = None
    routed_provider = None
    task_type = "general"
    _router_decision = None
    try:
        from src.routing.model_router import ModelRouter
        router = ModelRouter.from_config()
        if router.enabled and not req.model_override:
            decision = router.route(req.stimulus)
            task_type = decision.task_type
            _router_decision = decision
            if decision.is_direct_model:
                routed_model = decision.model
            elif not decision.fallback:
                routed_model = decision.model
                routed_provider = decision.provider
    except Exception as exc:
        log.warning("[router] Model router failed: %s", exc)

    model = (
        req.model_override
        or routed_model
        or agent_cfg.get("model")
        or profile.get("model", "")
        or (conn["models"][0] if conn.get("models") else "gpt-4o-mini")
    )

    # If router chose a specific connection, switch to it
    if _router_decision:
        if getattr(_router_decision, "direct_conn_id", None) and not req.model_override:
            direct_conn = _resolve_connection(_router_decision.direct_conn_id, req.agent)
            if direct_conn:
                conn = direct_conn
        elif getattr(_router_decision, "connection_id", None) and not req.model_override:
            tier_conn = _resolve_connection(_router_decision.connection_id, req.agent)
            if tier_conn:
                conn = tier_conn

    # ── OpenRouter prefix fix ──────────────────────────────────────
    # OpenRouter requires "provider/model" format (e.g. "openai/gpt-5.4").
    # If the resolved model has no slash and the connection is OpenRouter,
    # auto-prefix with "openai/" so bare names like "gpt-5.4" work.
    if conn.get("provider") == "openrouter" and model and "/" not in model:
        model = f"openai/{model}"
        log.info("[stream] Auto-prefixed model for OpenRouter: %s", model)

    url = conn["url"].rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if conn.get("api_key"):
        headers["Authorization"] = f"Bearer {conn['api_key']}"

    log.info("[stream] agent=%s model=%s conn=%s", req.agent, model, conn.get("id"))

    MAX_TOOL_ROUNDS = 10
    tool_call_log: list[dict] = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    cost_data = {}
    _credits_deducted = 0
    _credits_balance = -1

    # ── Pre-flight credit check for platform-hosted keys ──
    # Reject immediately if the user has zero credits to prevent free API usage
    if conn.get("platform_hosted") and user:
        _preflight_balance = get_user_credits(user["id"])
        if _preflight_balance <= 0:
            yield _sse({"type": "error", "message": "Insufficient credits. Purchase more in the Store.", "redirect": "/store"})
            return

    # Inject runtime context
    try:
        from src.tools.runtime_info import RuntimeInfoTool
        from src.runtime_policy import RuntimePolicy
        policy_cfg = profile.get("policy", {})
        policy = RuntimePolicy(
            max_iterations=policy_cfg.get("max_iterations", 25),
            stasis_mode=policy_cfg.get("stasis_mode", False),
            tool_failure_mode=policy_cfg.get("tool_failure_mode", "continue"),
        )
        RuntimeInfoTool.set_context(profile=profile, policy=policy, execution_mode="interactive")
    except Exception:
        pass

    running_messages = list(llm_messages)

    # ── Inject attached images into the last user message (vision) ──
    if req.images:
        for i in range(len(running_messages) - 1, -1, -1):
            if running_messages[i].get("role") == "user":
                orig = running_messages[i].get("content", "")
                content_parts = [{"type": "text", "text": orig}]
                for img_url in req.images:
                    content_parts.append({"type": "image_url", "image_url": {"url": img_url}})
                running_messages[i] = {"role": "user", "content": content_parts}
                break

    full_response = ""   # accumulate final response text (for saving to chat)

    # Swap SearXNG → OpenRouter native web search when using OpenRouter
    tool_defs = _prepare_tools_for_connection(tool_defs, conn)

    for _round in range(MAX_TOOL_ROUNDS + 1):
        payload = {
            "model": model,
            "messages": running_messages,
            "temperature": profile.get("temperature", 0.7),
            "stream": True,
        }
        if tool_defs:
            payload["tools"] = tool_defs

        try:
            async with httpx.AsyncClient(timeout=180) as client:
                async with client.stream("POST", url, json=payload, headers=headers) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        log.error("[stream] API error %s for model=%s: %s", resp.status_code, model, body.decode()[:500])
                        yield _sse({"type": "error", "message": f"API {resp.status_code}: {body.decode()[:300]}"})
                        return

                    think_parser = _ThinkTagParser()
                    round_content = ""
                    tool_calls_acc: dict[int, dict] = {}
                    finish_reason = None

                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        choice = chunk.get("choices", [{}])[0]
                        delta = choice.get("delta", {})
                        finish_reason = choice.get("finish_reason") or finish_reason

                        # ── Accumulate streamed tool call chunks ──
                        if "tool_calls" in delta:
                            for tc_chunk in delta["tool_calls"]:
                                idx = tc_chunk.get("index", 0)
                                if idx not in tool_calls_acc:
                                    tool_calls_acc[idx] = {
                                        "id": tc_chunk.get("id", ""),
                                        "name": tc_chunk.get("function", {}).get("name", ""),
                                        "arguments": "",
                                    }
                                if tc_chunk.get("id"):
                                    tool_calls_acc[idx]["id"] = tc_chunk["id"]
                                fn = tc_chunk.get("function", {})
                                if fn.get("name"):
                                    tool_calls_acc[idx]["name"] = fn["name"]
                                if fn.get("arguments"):
                                    tool_calls_acc[idx]["arguments"] += fn["arguments"]
                            continue

                        # ── Stream text content with think-tag detection ──
                        content = delta.get("content", "")
                        if not content:
                            continue
                        round_content += content

                        for ev_type, ev_text in think_parser.feed(content):
                            yield _sse({"type": ev_type, "content": ev_text})

                    # Flush any remaining buffered text
                    for ev_type, ev_text in think_parser.flush():
                        yield _sse({"type": ev_type, "content": ev_text})

                    # ── Accumulate usage from the final chunk ──
                    # (Some providers include usage in the last chunk)
                    try:
                        usage = chunk.get("usage", {}) or {}
                        for k in total_usage:
                            total_usage[k] += usage.get(k, 0)
                    except Exception:
                        pass

                    full_response += round_content

        except Exception as exc:
            yield _sse({"type": "error", "message": f"Request failed: {exc}"})
            return

        # ── Cost metering ──
        provider = conn.get("provider", "openai")
        _cost_source = "platform" if conn.get("platform_hosted") else "user"
        try:
            from src.observability.metering import meter_from_raw_usage, log_cost_event
            metering = meter_from_raw_usage(total_usage, provider=provider, model=model)
            cost_data = metering.cost.to_dict()
            log_cost_event(metering, agent=req.agent, chat_id=chat_data["id"], source=_cost_source)
        except Exception:
            pass

        # ── Credit deduction for platform-hosted keys ──
        # Charge whenever tokens were used, even for models missing from
        # pricing.yaml (safe estimator applies a fallback so they're never free).
        if conn.get("platform_hosted") and user and total_usage.get("total_tokens", 0) > 0:
            try:
                credit_cost = estimate_llm_credit_cost_safe(
                    cost_data.get("total_cost", 0), total_usage.get("total_tokens", 0)
                )
                if credit_cost > 0:
                    balance = get_user_credits(user["id"])
                    if balance < credit_cost:
                        # Drain remaining balance so the next pre-flight blocks
                        # further calls (prevents unlimited free calls).
                        if balance > 0:
                            deduct_user_credits(user["id"], balance, f"llm:{model}:shortfall")
                            _credits_deducted += balance
                        _credits_balance = 0
                        yield _sse({"type": "error", "message": "Insufficient credits for LLM usage"})
                        return
                    deduct_result = deduct_user_credits(user["id"], credit_cost, f"llm:{model}:{total_usage.get('total_tokens', 0)}tok")
                    _credits_deducted += credit_cost
                    _credits_balance = deduct_result.get("balance", 0)
            except Exception:
                pass

        # ── Handle tool calls ──
        if tool_calls_acc:
            # Build the assistant message with tool_calls for context
            tc_list = []
            for idx in sorted(tool_calls_acc):
                tc = tool_calls_acc[idx]
                tc_list.append({
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                })
            running_messages.append({"role": "assistant", "content": round_content or None, "tool_calls": tc_list})

            from src.tools.registry import execute_tool

            for idx in sorted(tool_calls_acc):
                tc = tool_calls_acc[idx]
                fn_name = tc["name"]
                try:
                    fn_args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except json.JSONDecodeError:
                    fn_args = {}

                yield _sse({"type": "tool_start", "tool": fn_name, "arguments": fn_args})

                try:
                    result = execute_tool(fn_name, fn_args, agent_name=req.agent)
                except PermissionError as exc:
                    result = f"BLOCKED: {exc}"
                except Exception as exc:
                    result = f"Error: {exc}"

                tool_call_log.append({
                    "round": _round + 1,
                    "tool": fn_name,
                    "arguments": fn_args,
                    "result": result[:500] if isinstance(result, str) else str(result)[:500],
                })

                yield _sse({"type": "tool_result", "tool": fn_name, "result": result[:500] if isinstance(result, str) else str(result)[:500]})

                running_messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result if isinstance(result, str) else str(result),
                })
            continue

        # No tool calls — we're done
        break

    # ── Post-stream processing ──
    saved_memories = _extract_and_save_memories(req.agent, full_response)
    response_text = _strip_memory_tags(full_response)

    # Image generation
    generated_image = None
    img_match = re.search(r'\[IMAGE_GEN:\s*(.+?)\]', response_text, re.DOTALL)
    if img_match:
        img_prompt = img_match.group(1).strip()
        response_text = response_text[:img_match.start()] + response_text[img_match.end():]
        response_text = response_text.strip()
        try:
            settings = _load_settings()
            img_cfg = settings.get("image", {})
            img_provider = img_cfg.get("preferred", "none")
            if img_provider and img_provider != "none":
                _img_credits = estimate_image_credit_cost(img_provider) if user else 0
                if user and _img_credits > 0 and get_user_credits(user["id"]) < _img_credits:
                    response_text += (
                        f"\n\n*Image generation skipped — needs {_img_credits} credits. "
                        f"Add more in the Store.*"
                    )
                else:
                    result = await _generate_image(img_provider, img_prompt, img_cfg, settings)
                    if "error" not in result:
                        generated_image = {**result, "prompt": img_prompt}
                        if user and _img_credits > 0:
                            try:
                                _r = deduct_user_credits(user["id"], _img_credits, f"image:{img_provider}")
                                _credits_deducted += _img_credits
                                _credits_balance = _r.get("balance", _credits_balance)
                            except Exception as exc:
                                log.warning("[credits] image credit deduction failed: %s", exc)
        except Exception:
            pass

    # Video generation
    generated_video = None
    vid_prompt = None
    vid_match = re.search(r'\[VIDEO_GEN:\s*(.+?)\]', response_text, re.DOTALL)
    if vid_match:
        vid_prompt = vid_match.group(1).strip()
        response_text = response_text[:vid_match.start()] + response_text[vid_match.end():]
        response_text = response_text.strip()
    else:
        p = (req.stimulus or "").strip()
        p_low = p.lower()
        if p and any(k in p_low for k in ("video", "loop", "looping", "animate", "animation", "live wallpaper", "background")):
            vid_prompt = p

    if vid_prompt:
        try:
            settings = _load_settings()
            vid_cfg = settings.get("video", {})
            vid_provider = vid_cfg.get("preferred", "none")
            if vid_provider and vid_provider != "none":
                # Veo video is billed per second — pre-check balance, then charge.
                _vid_secs = vid_cfg.get("duration_seconds", 8)
                _vid_credits = estimate_video_credit_cost(vid_provider, _vid_secs) if user else 0
                if user and _vid_credits > 0 and get_user_credits(user["id"]) < _vid_credits:
                    generated_video = {
                        "error": f"Not enough credits for video ({_vid_credits} needed).",
                        "prompt": vid_prompt,
                    }
                    response_text += (
                        f"\n\n*Video generation skipped — needs {_vid_credits} credits. "
                        f"Add more in the Store.*"
                    )
                else:
                    result = await _generate_video(
                        vid_provider, vid_prompt, vid_cfg, settings,
                        save_dir=_VIDEO_UPLOADS_DIR,
                    )
                    generated_video = {**result, "prompt": vid_prompt}
                    if "error" in result:
                        log.warning("[video-gen] %s", result["error"])
                        response_text += f"\n\n*Video generation failed: {result['error']}*"
                    elif user and _vid_credits > 0:
                        try:
                            _r = deduct_user_credits(user["id"], _vid_credits, f"video:{vid_provider}:{int(_vid_secs)}s")
                            _credits_deducted += _vid_credits
                            _credits_balance = _r.get("balance", _credits_balance)
                        except Exception as exc:
                            log.warning("[credits] video credit deduction failed: %s", exc)
            else:
                generated_video = {
                    "error": "Video generation is disabled. Enable a Veo model in Settings -> Video Generation.",
                    "prompt": vid_prompt,
                }
                response_text += "\n\n*Video generation is disabled. Enable a Veo model in Settings -> Video Generation.*"
        except Exception as exc:
            log.warning("[video-gen] Streaming video gen failed: %s", exc)
            generated_video = {"error": str(exc), "prompt": vid_prompt}
            response_text += f"\n\n*Video generation failed: {exc}*"

    # Build layers metadata
    layers["tools"]["calls"] = tool_call_log
    _tier_label = getattr(_router_decision, "tier_name", None) if _router_decision else None
    layers["router"] = {
        "task_type": task_type,
        "routed_model": routed_model,
        "routed_provider": routed_provider,
        "tier": _tier_label,
    }

    # Save to chat history
    chat_data["messages"].append({
        "role": "assistant", "text": response_text, "time": now,
        "usage": total_usage,
        "data": {
            "agent": req.agent, "model": model,
            "usage": total_usage, "cost": cost_data,
            "tool_calls": tool_call_log,
            "generated_image": generated_image,
            "generated_video": generated_video,
            "task_type": task_type,
        },
        "layers": layers,
    })
    _save_chat(chat_data["id"], chat_data)

    idx_data = _load_chat_index()
    for c in idx_data["chats"]:
        if c["id"] == chat_data["id"]:
            c["updated"] = now
            break
    _save_chat_index(idx_data)

    # Build credit info
    _credit_info = {}
    if conn.get("platform_hosted") and _credits_deducted > 0:
        _credit_info = {
            "credits_deducted": _credits_deducted,
            "credits_balance": _credits_balance if _credits_balance >= 0 else (get_user_credits(user["id"]) if user else 0),
            "platform_hosted": True, "markup": LLM_MARKUP_MULTIPLIER,
        }
    elif user:
        _credit_info = {
            "credits_deducted": 0,
            "credits_balance": get_user_credits(user["id"]),
            "platform_hosted": bool(conn.get("platform_hosted")), "markup": 0,
        }

    # ── Final "done" event with all metadata ──
    yield _sse({
        "type": "done",
        "response": response_text,
        "chat_id": chat_data["id"],
        "model": model,
        "usage": total_usage,
        "cost": cost_data,
        "layers": layers,
        "saved_memories": saved_memories,
        "tool_calls": tool_call_log,
        "generated_image": generated_image,
        "generated_video": generated_video,
        "task_type": task_type,
        "credits": _credit_info,
    })


@app.get("/api/chat/history")
async def api_chat_history(request: Request):
    uid = _get_user_id(request)
    return _load_chat_index(user_id=uid)

@app.post("/api/chat/new")
async def api_chat_new(request: Request):
    body = await request.json()
    uid = _get_user_id(request)
    agents = _list_agents()
    agent = body.get("agent", agents[0] if agents else "agent")
    return _create_new_chat(agent, user_id=uid)

@app.get("/api/chat/{chat_id}")
async def api_chat_get(chat_id: str, request: Request):
    uid = _get_user_id(request)
    data = _load_chat(chat_id, user_id=uid)
    return data if data else JSONResponse({"error": "Not found"}, 404)

@app.delete("/api/chat/{chat_id}")
async def api_chat_delete(chat_id: str, request: Request):
    uid = _get_user_id(request)
    if uid and uid != "__local__":
        path = user_chats_dir(uid) / f"{chat_id}.json"
    else:
        path = _CHATS_DIR / f"{chat_id}.json"
    if path.exists():
        path.unlink()
    idx = _load_chat_index(user_id=uid)
    idx["chats"] = [c for c in idx["chats"] if c["id"] != chat_id]
    _save_chat_index(idx, user_id=uid)
    return {"ok": True}

@app.put("/api/chat/{chat_id}")
async def api_chat_update(chat_id: str, request: Request):
    body = await request.json()
    uid = _get_user_id(request)
    data = _load_chat(chat_id, user_id=uid)
    if not data:
        return JSONResponse({"error": "Not found"}, 404)
    for key in ("title", "folder_id"):
        if key in body:
            data[key] = body[key]
    _save_chat(chat_id, data, user_id=uid)
    idx = _load_chat_index(user_id=uid)
    for c in idx["chats"]:
        if c["id"] == chat_id:
            c.update({k: body[k] for k in ("title", "folder_id") if k in body})
            break
    _save_chat_index(idx, user_id=uid)
    return {"ok": True}

@app.post("/api/chat/{chat_id}/title")
async def api_chat_auto_title(chat_id: str, request: Request):
    uid = _get_user_id(request)
    data = _load_chat(chat_id, user_id=uid)
    if not data or len(data.get("messages", [])) < 2:
        return {"title": None}

    agent = data.get("agent", "")
    conn = _resolve_connection(None, agent)
    if not conn:
        return {"title": None}

    snippet = "\n".join(f"{m['role']}: {m['text'][:200]}" for m in data["messages"][:4])
    profile = _load_profile(agent, user_id=uid)
    model = _get_agent_config(agent, user_id=uid).get("model") or profile.get("model", "") or "gpt-4o-mini"

    url = conn["url"].rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if conn.get("api_key"):
        headers["Authorization"] = f"Bearer {conn['api_key']}"

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "Generate a short (3-6 word) title for this conversation. Reply with ONLY the title."},
                    {"role": "user", "content": snippet},
                ],
                "temperature": 0.5, "max_tokens": 20,
            }, headers=headers)
            resp.raise_for_status()
            title = resp.json()["choices"][0]["message"]["content"].strip().strip('"')
    except Exception:
        return {"title": None}

    data["title"] = title
    _save_chat(chat_id, data)
    idx = _load_chat_index()
    for c in idx["chats"]:
        if c["id"] == chat_id:
            c["title"] = title
            break
    _save_chat_index(idx)
    return {"title": title}


# ═══════════════════════════════════════════════════════════════════
#  PROFILES API
# ═══════════════════════════════════════════════════════════════════

# ── User profile must come BEFORE the {name} wildcard routes ─────
@app.put("/api/profiles/user")
async def api_profile_user(request: Request):
    """Update user profile (name, avatar, color, crop/position)."""
    body = await request.json()
    uid = _get_user_id(request)
    settings = _load_settings(user_id=uid)
    user_profile = settings.setdefault("user_profile", {})
    for key in ("name", "color"):
        if key in body:
            user_profile[key] = body[key]
    # Handle image — save base64 as file, store URL path
    if "image" in body:
        img_data = body["image"]
        if not img_data:
            # Remove image — also delete old file
            old_img = user_profile.pop("image", None)
            if old_img and old_img.startswith("/uploads/"):
                old_path = _UPLOADS_DIR / os.path.basename(old_img)
                if old_path.exists():
                    old_path.unlink(missing_ok=True)
        elif img_data.startswith("data:image"):
            # base64 data-URL → save as file
            try:
                header, b64 = img_data.split(",", 1)
                ext = ".png"
                if "jpeg" in header or "jpg" in header:
                    ext = ".jpg"
                elif "webp" in header:
                    ext = ".webp"
                raw = _b64.b64decode(b64)
                fname = f"user_avatar_{uuid.uuid4().hex[:8]}{ext}"
                dest = _UPLOADS_DIR / fname
                # Remove previous file
                old_img = user_profile.get("image", "")
                if old_img.startswith("/uploads/user_avatar_"):
                    old_path = _UPLOADS_DIR / os.path.basename(old_img)
                    if old_path.exists():
                        old_path.unlink(missing_ok=True)
                with open(dest, "wb") as f:
                    f.write(raw)
                user_profile["image"] = f"/uploads/{fname}"
            except Exception:
                pass  # silently ignore malformed data
        else:
            # Already a URL path — keep as-is
            user_profile["image"] = img_data
    # Photo crop / position fields
    for key in ("photo_zoom", "photo_x", "photo_y"):
        if key in body:
            user_profile[key] = body[key]
    _save_settings(settings, user_id=uid)
    return {"ok": True, "image": user_profile.get("image", "")}

# ── Trash routes must come BEFORE {name} wildcard ────────────────
@app.get("/api/profiles/trash")
async def api_trash_list():
    """List all trashed agents."""
    idx = _load_trash_index()
    now = datetime.now(timezone.utc)
    for entry in idx:
        try:
            expires = datetime.fromisoformat(entry["expires_at"])
            entry["days_remaining"] = max(0, (expires - now).days)
        except (KeyError, ValueError):
            entry["days_remaining"] = 0
    return {"items": idx}


@app.post("/api/profiles/trash/{trash_id}/restore")
async def api_trash_restore(trash_id: str):
    """Restore a trashed agent."""
    name = _restore_agent(trash_id)
    if not name:
        return JSONResponse({"error": "Cannot restore — agent may already exist or trash entry not found"}, 400)
    return {"ok": True, "name": name}


@app.delete("/api/profiles/trash/{trash_id}")
async def api_trash_permanently_delete(trash_id: str):
    """Permanently delete a trashed agent."""
    _permanently_delete_from_trash(trash_id)
    return {"ok": True}

@app.get("/api/profiles/{name}")
async def api_profile_get(name: str, request: Request):
    uid = _get_user_id(request)
    profile = _load_profile(name, user_id=uid)
    if not profile:
        return JSONResponse({"error": "Not found"}, 404)
    return {"name": name, "profile": profile, "config": _get_agent_config(name, user_id=uid),
            "system_prompt": _load_system_prompt(name, user_id=uid),
            "soul_script": _load_soul_script(name, user_id=uid)}

@app.put("/api/profiles/{name}")
async def api_profile_update(name: str, request: Request):
    body = await request.json()
    uid = _get_user_id(request)
    if "system_prompt" in body:
        _save_system_prompt(name, body["system_prompt"], user_id=uid)
    profile = _load_profile(name, user_id=uid)
    for key in ("model", "temperature"):
        if key in body:
            profile[key] = body[key]
    _save_profile(name, profile, user_id=uid)
    if "config" in body:
        cfg = _get_agent_config(name, user_id=uid)
        cfg.update(body["config"])
        _save_agent_config(name, cfg, user_id=uid)
    return {"ok": True}

@app.post("/api/profiles")
async def api_profile_create(request: Request):
    body = await request.json()
    uid = _get_user_id(request)
    name = body.get("name", "").strip().lower().replace(" ", "_")
    if not name:
        return JSONResponse({"error": "Name required"}, 400)
    # Check both global and per-user profiles
    if (_PROFILES_DIR / f"{name}.yaml").exists():
        return JSONResponse({"error": "Already exists"}, 400)
    if uid and uid != "__local__":
        if (user_profiles_dir(uid) / f"{name}.yaml").exists():
            return JSONResponse({"error": "Already exists"}, 400)
    _save_profile(name, {"name": name, "model": body.get("model", ""), "temperature": 0.7,
                         "system_prompt": f"{name}.system.md"}, user_id=uid)
    _save_system_prompt(name, body.get("system_prompt", f"You are {name}."), user_id=uid)
    return {"ok": True, "name": name}

@app.delete("/api/profiles/{name}")
async def api_profile_delete(name: str, request: Request):
    """Soft-delete: move agent to trash (30-day retention)."""
    uid = _get_user_id(request)
    # Check per-user profile first, then global
    has_profile = False
    if uid and uid != "__local__":
        has_profile = (user_profiles_dir(uid) / f"{name}.yaml").exists()
    if not has_profile:
        has_profile = (_PROFILES_DIR / f"{name}.yaml").exists()
    if not has_profile:
        return JSONResponse({"error": "Agent not found"}, 404)
    _trash_agent(name)
    return {"ok": True}

@app.put("/api/profiles/{name}/config")
async def api_profile_config(name: str, request: Request):
    """Update individual config fields for an agent (display_name, description, model, etc.)."""
    body = await request.json()
    uid = _get_user_id(request)
    cfg = _get_agent_config(name, user_id=uid)
    for key in ("display_name", "description", "model", "allowed_tools",
                "voice_id", "edge_voice", "inworld_voice", "tts_paid_provider", "tts_free_provider",
                "identity_faiss_profile", "memory_vault_profile"):
        if key in body:
            cfg[key] = body[key]
    # Also persist model + voice + allowed_tools + memory profiles to profile yaml for compatibility
    profile = _load_profile(name, user_id=uid)
    if "model" in body:
        profile["model"] = body["model"]
    if "voice_id" in body:
        profile["voice_id"] = body["voice_id"]
    if "edge_voice" in body:
        profile["edge_voice"] = body["edge_voice"]
    if "inworld_voice" in body:
        profile["inworld_voice"] = body["inworld_voice"]
    if "allowed_tools" in body:
        profile["allowed_tools"] = body["allowed_tools"]
    if "identity_faiss_profile" in body:
        profile["identity_faiss_profile"] = body["identity_faiss_profile"]
    if "memory_vault_profile" in body:
        profile["memory_vault_profile"] = body["memory_vault_profile"]
    _save_profile(name, profile, user_id=uid)
    # Also persist system_prompt_text if sent
    if "system_prompt_text" in body:
        _save_system_prompt(name, body["system_prompt_text"], user_id=uid)
    # Also persist soul_script_text if sent
    if "soul_script_text" in body:
        _save_soul_script(name, body["soul_script_text"], user_id=uid)
    _save_agent_config(name, cfg, user_id=uid)
    return {"ok": True}

@app.put("/api/profiles/{name}/avatar")
async def api_profile_avatar(name: str, request: Request):
    """Update avatar image, colour, or photo crop/position for an agent."""
    body = await request.json()
    uid = _get_user_id(request)
    settings = _load_settings(user_id=uid)
    avatars = settings.setdefault("agent_avatars", {})
    entry = avatars.setdefault(name, {})
    if uid and uid != "__local__":
        uploads_dir = user_uploads_dir(uid)
    else:
        uploads_dir = _UPLOADS_DIR
    if "color" in body:
        entry["color"] = body["color"]
    if "image" in body:
        img_data = body["image"]
        if not img_data:
            old_img = entry.pop("image", None)
            if old_img and old_img.startswith("/uploads/"):
                old_path = uploads_dir / os.path.basename(old_img)
                if old_path.exists():
                    old_path.unlink(missing_ok=True)
        elif img_data.startswith("data:image"):
            try:
                header, b64 = img_data.split(",", 1)
                ext = ".png"
                if "jpeg" in header or "jpg" in header:
                    ext = ".jpg"
                elif "webp" in header:
                    ext = ".webp"
                raw = _b64.b64decode(b64)
                fname = f"avatar_{name}_{uuid.uuid4().hex[:8]}{ext}"
                dest = uploads_dir / fname
                old_img = entry.get("image", "")
                if old_img.startswith("/uploads/avatar_"):
                    old_path = uploads_dir / os.path.basename(old_img)
                    if old_path.exists():
                        old_path.unlink(missing_ok=True)
                with open(dest, "wb") as f:
                    f.write(raw)
                entry["image"] = f"/uploads/{fname}"
            except Exception:
                pass
        else:
            entry["image"] = img_data
    # Photo crop / position fields
    for key in ("photo_zoom", "photo_x", "photo_y"):
        if key in body:
            entry[key] = body[key]
    avatars[name] = entry
    _save_settings(settings, user_id=uid)
    return {"ok": True, "image": entry.get("image", "")}

@app.post("/api/profiles/create")
async def api_profile_create_v2(request: Request):
    """Create a new agent (v2 — supports description and model)."""
    body = await request.json()
    uid = _get_user_id(request)
    name = body.get("name", "").strip().lower().replace(" ", "_")
    if not name:
        return JSONResponse({"error": "Name required"}, 400)
    if (_PROFILES_DIR / f"{name}.yaml").exists():
        return JSONResponse({"error": "Already exists"}, 400)
    if uid and uid != "__local__":
        if (user_profiles_dir(uid) / f"{name}.yaml").exists():
            return JSONResponse({"error": "Already exists"}, 400)
    model = body.get("model", "")
    desc = body.get("description", "")
    _save_profile(name, {"name": name, "model": model, "temperature": 0.7,
                         "system_prompt": f"{name}.system.md"}, user_id=uid)
    _save_system_prompt(name, f"You are {name}.", user_id=uid)
    if desc or model:
        cfg = _get_agent_config(name, user_id=uid)
        if desc:
            cfg["description"] = desc
        if model:
            cfg["model"] = model
        _save_agent_config(name, cfg, user_id=uid)
    return {"ok": True, "name": name}

@app.put("/api/profiles/{name}/knowledge")
async def api_profile_knowledge(name: str, request: Request):
    body = await request.json()
    uid = _get_user_id(request)
    cfg = _get_agent_config(name, user_id=uid)
    essential = set(cfg.get("essential_notes", []))
    # Preserve essential notes — they cannot be detached
    incoming = body.get("attached_notes", [])
    incoming_modes = body.get("note_modes", {})
    for eid in essential:
        if eid not in incoming:
            incoming.append(eid)
        if eid not in incoming_modes:
            incoming_modes[eid] = cfg.get("note_modes", {}).get(eid, "directive")
    cfg["attached_notes"] = incoming
    cfg["note_modes"] = incoming_modes
    _save_agent_config(name, cfg, user_id=uid)
    try:
        from src.storage.note_collector import invalidate_notes_faiss
        invalidate_notes_faiss()
        _rebuild_notes_faiss()
    except Exception as exc:
        log.warning("[knowledge] FAISS rebuild skipped: %s", exc)
    return {"ok": True}

def _rebuild_notes_faiss():
    """Rebuild NotesFAISS index from all directive-mode notes across agents.

    Uses semantic chunking (split on ### headers) with overlapping fallback
    for long headerless content.  Each chunk carries ``document_id`` so the
    search filter in NotesFAISS matches correctly.

    Chunk size and overlap are read from the identity FAISS profile
    (config/identity_profile.json) so they respond to UI changes.
    """
    from src.memory.notes_faiss import NotesFAISS
    from src.storage.user_notes_loader import strip_html
    from src.memory.profile_resolver import get_indexing_policy

    # Read from identity profile — falls back to safe defaults
    _idx = get_indexing_policy()
    # Profile stores token counts; approximate to chars (≈4 chars/token)
    CHUNK_TARGET = int(_idx.get("chunk_size_tokens", 400) * 1.5)
    CHUNK_OVERLAP = int(_idx.get("chunk_overlap_tokens", 80) * 1.5)

    def _chunk_text(text: str, doc_id: str, title: str) -> list[dict]:
        """Split text into overlapping chunks, preferring ### boundaries."""
        import re
        sections: list[tuple[str, str]] = []  # (section_title, body)
        parts = re.split(r'(?m)^###\s+', text)
        if len(parts) > 1:
            # First part is content before any header
            if parts[0].strip():
                sections.append((title, parts[0].strip()))
            for part in parts[1:]:
                lines = part.split('\n', 1)
                sec_title = lines[0].strip()
                sec_body = lines[1].strip() if len(lines) > 1 else ''
                if sec_body:
                    sections.append((sec_title, f'### {sec_title}\n{sec_body}'))
        else:
            sections.append((title, text))

        out = []
        for sec_title, body in sections:
            if len(body) <= CHUNK_TARGET + 100:
                out.append({
                    "text": body,
                    "metadata": {"document_id": doc_id, "document_title": title,
                                 "section_path": sec_title},
                })
            else:
                # Sliding window with overlap
                step = max(CHUNK_TARGET - CHUNK_OVERLAP, 200)
                for i in range(0, len(body), step):
                    chunk = body[i:i + CHUNK_TARGET]
                    if len(chunk) < 80 and out:
                        break  # skip tiny trailing scraps
                    out.append({
                        "text": chunk,
                        "metadata": {"document_id": doc_id, "document_title": title,
                                     "section_path": sec_title},
                    })
        return out

    chunks: list[dict] = []
    seen_note_ids: set[str] = set()
    for agent, cfg in _load_settings().get("agent_configs", {}).items():
        modes = cfg.get("note_modes", {})
        for nid in cfg.get("attached_notes", []):
            if modes.get(nid) == "directive" and nid not in seen_note_ids:
                seen_note_ids.add(nid)
                note = _load_note(nid)
                if note and not note.get("trashed"):
                    text = strip_html(note.get("content_html", ""))
                    if text:
                        chunks.extend(_chunk_text(text, nid, note.get("title", "Untitled")))

    # ── Also index soul script directive files for every agent ──
    soul_script_count = 0
    for agent_name in _list_agents():
        ss_path = _DIRECTIVES_DIR / f"{agent_name}.md"
        if ss_path.exists():
            ss_text = ss_path.read_text(encoding="utf-8").strip()
            if ss_text:
                doc_id = f"__soul_script__{agent_name}"
                ss_chunks = _chunk_text(ss_text, doc_id, f"Soul Script — {agent_name}")
                chunks.extend(ss_chunks)
                soul_script_count += 1

    faiss_dir = str(_FAISS_DIR)
    if chunks:
        nf = NotesFAISS(faiss_dir)
        nf.build_index(chunks)
        log.info("[knowledge] NotesFAISS rebuilt — %d chunks from %d notes + %d soul scripts",
                 len(chunks), len(seen_note_ids), soul_script_count)
    else:
        log.info("[knowledge] No directive-mode notes or soul scripts found — NotesFAISS empty")


# ═══════════════════════════════════════════════════════════════════
#  VAULT API
# ═══════════════════════════════════════════════════════════════════

@app.post("/api/vault/add")
async def api_vault_add(request: Request):
    """Manually add a memory to the vault."""
    body = await request.json()
    text = (body.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "Memory text is required"}, 400)
    uid = _get_user_id(request)
    fm = _get_faiss_memory(user_id=uid)
    if fm:
        try:
            mem = fm.add(
                text=text,
                scope=body.get("scope", "shared"),
                category=body.get("category", "other"),
                source=body.get("source", "manual"),
                tags=body.get("tags", []),
            )
            return {"status": "saved", "id": mem.id, "text": text[:120]}
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, 500)
    # VaultStore fallback
    vs = _get_vault_store(user_id=uid)
    if not vs:
        return JSONResponse({"error": "Vault not available"}, 500)
    try:
        mem = vs.create_memory(
            text=text,
            scope=body.get("scope", "shared"),
            category=body.get("category", "other"),
            source=body.get("source", "manual"),
            tags=body.get("tags", []),
        )
        return {"status": "saved", "id": mem.id, "text": text[:120]}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, 500)

@app.post("/api/vault/batch_add")
async def api_vault_batch_add(request: Request):
    """Batch-add multiple memories to the vault in one call."""
    body = await request.json()
    items = body.get("memories")
    if not items or not isinstance(items, list):
        return JSONResponse({"error": "memories array is required"}, 400)
    uid = _get_user_id(request)
    fm = _get_faiss_memory(user_id=uid)
    if fm:
        try:
            created = fm.batch_add(items)
            return {
                "status": "saved",
                "count": len(created),
                "ids": [m.id for m in created],
            }
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, 500)
    # VaultStore fallback
    vs = _get_vault_store(user_id=uid)
    if not vs:
        return JSONResponse({"error": "Vault not available"}, 500)
    try:
        created = vs.batch_create_many(items)
        return {
            "status": "saved",
            "count": len(created),
            "ids": [m.id for m in created],
        }
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, 500)

@app.get("/api/vault/stats")
async def api_vault_stats(request: Request):
    uid = _get_user_id(request)
    fm = _get_faiss_memory(user_id=uid)
    if fm:
        return fm.stats()
    vs = _get_vault_store(user_id=uid)
    if vs:
        active = vs.read_active()
        all_raw = vs.read_all()
        return {"active_count": len(active), "raw_lines": len(all_raw),
                "compactable_lines": len(all_raw) - len(active)}
    return {"error": "Vault not available"}

@app.post("/api/vault/delete")
async def api_vault_delete(request: Request):
    body = await request.json()
    uid = _get_user_id(request)
    fm = _get_faiss_memory(user_id=uid)
    if fm:
        deleted = [mid for mid in body.get("ids", []) if fm.delete(mid)]
        return {"deleted": deleted}
    vs = _get_vault_store(user_id=uid)
    if not vs:
        return {"error": "Vault not available"}
    result = vs.bulk_delete(body.get("ids", []))
    return {"deleted": result["deleted"]}

@app.get("/api/vault/compact")
async def api_vault_compact(request: Request):
    uid = _get_user_id(request)
    fm = _get_faiss_memory(user_id=uid)
    if fm:
        before = fm.stats().get("raw_lines", 0)
        fm.rebuild_index()
        return {"before_lines": before, "after_lines": fm.stats().get("raw_lines", 0)}
    vs = _get_vault_store(user_id=uid)
    if not vs:
        return {"error": "Vault not available"}
    result = vs.compact()
    return {"before_lines": result.get("raw_before", 0), "after_lines": result.get("raw_after", 0)}


# ═══════════════════════════════════════════════════════════════════
#  KNOWLEDGE FOLDERS API
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/knowledge/folders")
async def api_knowledge_folders_list(request: Request):
    uid = _get_user_id(request)
    return JSONResponse(_load_folders(user_id=uid))

@app.post("/api/knowledge/folders")
async def api_knowledge_folders_create(request: Request):
    body = await request.json()
    name = (body.get("name") or "").strip()
    if not name:
        return JSONResponse({"error": "Folder name is required"}, 400)
    uid = _get_user_id(request)
    folders = _load_folders(user_id=uid)
    folder_id = str(uuid.uuid4())[:8]
    folder = {
        "id": folder_id,
        "name": name,
        "emoji": body.get("emoji", "📁"),
        "pinned": False,
        "color": body.get("color", "#6366f1"),
    }
    folders.append(folder)
    _save_folders(folders, user_id=uid)
    return JSONResponse(folder)

@app.put("/api/knowledge/folders/{folder_id}")
async def api_knowledge_folders_update(folder_id: str, request: Request):
    body = await request.json()
    uid = _get_user_id(request)
    folders = _load_folders(user_id=uid)
    for f in folders:
        if f["id"] == folder_id:
            for key in ("name", "emoji", "color"):
                if key in body:
                    f[key] = body[key]
            _save_folders(folders, user_id=uid)
            return JSONResponse(f)
    return JSONResponse({"error": "Folder not found"}, 404)

@app.delete("/api/knowledge/folders/{folder_id}")
async def api_knowledge_folders_delete(folder_id: str, request: Request):
    uid = _get_user_id(request)
    folders = _load_folders(user_id=uid)
    target = next((f for f in folders if f["id"] == folder_id), None)
    if not target:
        return JSONResponse({"error": "Folder not found"}, 404)
    if target.get("pinned"):
        return JSONResponse({"error": "Cannot delete a pinned folder"}, 403)
    # Move notes in this folder to Uncategorized
    idx = _load_notes_index(user_id=uid)
    for entry in idx:
        if entry.get("section") == folder_id:
            entry["section"] = "Uncategorized"
            note = _load_note(entry["id"], user_id=uid)
            if note:
                note["section"] = "Uncategorized"
                _save_note(entry["id"], note, user_id=uid)
    _save_notes_index(idx, user_id=uid)
    folders = [f for f in folders if f["id"] != folder_id]
    _save_folders(folders, user_id=uid)
    return JSONResponse({"ok": True})


# ═══════════════════════════════════════════════════════════════════
#  KNOWLEDGE API
# ═══════════════════════════════════════════════════════════════════

@app.post("/api/knowledge")
async def api_knowledge_create(request: Request):
    body = await request.json()
    uid = _get_user_id(request)
    note_id = str(uuid.uuid4())[:8]
    now = datetime.now(timezone.utc).isoformat()
    note = {
        "id": note_id, "title": body.get("title", "Untitled"),
        "emoji": body.get("emoji", "📄"),
        "content_html": body.get("content_html", ""),
        "preview": body.get("preview", ""),
        "section": body.get("section", "Uncategorized"),
        "created": now, "updated": now,
    }
    _save_note(note_id, note, user_id=uid)
    idx = _load_notes_index(user_id=uid)
    idx.append({k: note[k] for k in ("id", "title", "emoji", "preview", "section", "created", "updated")})
    _save_notes_index(idx, user_id=uid)
    return note

@app.put("/api/knowledge/{note_id}")
async def api_knowledge_update(note_id: str, request: Request):
    body = await request.json()
    uid = _get_user_id(request)
    note = _load_note(note_id, user_id=uid)
    if not note:
        return JSONResponse({"error": "Not found"}, 404)
    for key in ("title", "emoji", "content_html", "preview", "section"):
        if key in body:
            note[key] = body[key]
    note["updated"] = datetime.now(timezone.utc).isoformat()
    _save_note(note_id, note, user_id=uid)
    idx = _load_notes_index(user_id=uid)
    for entry in idx:
        if entry["id"] == note_id:
            for key in ("title", "emoji", "preview", "section", "updated"):
                if key in note:
                    entry[key] = note[key]
            break
    _save_notes_index(idx, user_id=uid)
    return note

@app.delete("/api/knowledge/{note_id}")
async def api_knowledge_delete(note_id: str, request: Request):
    uid = _get_user_id(request)
    now = datetime.now(timezone.utc).isoformat()
    note = _load_note(note_id, user_id=uid)
    if note:
        note["trashed"] = now
        _save_note(note_id, note, user_id=uid)
    idx = _load_notes_index(user_id=uid)
    for entry in idx:
        if entry["id"] == note_id:
            entry["trashed"] = now
            break
    _save_notes_index(idx, user_id=uid)
    return {"ok": True}

@app.get("/api/knowledge/{note_id}")
async def api_knowledge_get(note_id: str, request: Request):
    uid = _get_user_id(request)
    note = _load_note(note_id, user_id=uid)
    return note if note else JSONResponse({"error": "Not found"}, 404)


# ═══════════════════════════════════════════════════════════════════
#  ADMIN — Platform API Key Management
# ═══════════════════════════════════════════════════════════════════

# Admin access: only the emails listed here can access /admin/*
# Set via env var (comma-separated) or falls back to this default.
ADMIN_EMAILS = set(
    e.strip().lower()
    for e in os.environ.get("ADMIN_EMAILS", "dr.trent.hunter@gmail.com").split(",")
    if e.strip()
)

# Admin access by Supabase user id (comma-separated env override). Lets a specific
# account be admin even when its OAuth email isn't in ADMIN_EMAILS.
ADMIN_USER_IDS = set(
    u.strip()
    for u in os.environ.get("ADMIN_USER_IDS", "a2cddfbd-dabb-4d50-b515-b600a08ce8e8").split(",")
    if u.strip()
)

def _user_is_admin(user: dict | None) -> bool:
    """Return True if the given user dict belongs to an admin (email or id)."""
    if not user:
        return False
    if user.get("id") in ADMIN_USER_IDS:
        return True
    return user.get("email", "").lower() in ADMIN_EMAILS


def _check_admin(request: Request) -> bool:
    """Verify the logged-in user is an admin (by OAuth email or user id)."""
    return _user_is_admin(getattr(request.state, "user", None))


def _get_platform_connections() -> list[dict]:
    """Return only platform-hosted connections."""
    store = _load_connections()
    return [c for c in store.get("connections", []) if c.get("platform_hosted")]


@app.get("/admin/keys", response_class=HTMLResponse)
async def page_admin_keys(request: Request):
    """Admin page for managing platform-hosted API keys."""
    if not _check_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    platform_conns = _get_platform_connections()
    # Expose env-var connections for Edge-TTS so the UI shows them
    _existing_providers = {c.get("provider") for c in platform_conns}
    for env_key, provider, label in [
        ("TTS_URL", "edge-tts", "OpenedAI Speech (TTS)"),
    ]:
        if provider not in _existing_providers:
            env_val = os.environ.get(env_key, "")
            if env_val:
                platform_conns.append({
                    "id": f"env_{provider}", "provider": provider,
                    "url": env_val, "api_key": "", "enabled": True,
                    "platform_hosted": True, "name": f"Platform — {label}",
                })
    return templates.TemplateResponse(request, "admin_keys.html", {
        "page": "admin",
        "platform_connections": platform_conns,
    })


@app.post("/api/admin/platform-keys")
async def api_admin_save_platform_key(request: Request):
    """Create or update a platform-hosted connection."""
    if not _check_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=403)

    body = await request.json()
    provider = body.get("provider", "").strip()
    api_key = body.get("api_key", "").strip()
    url = body.get("url", "").strip()
    name = body.get("name", f"Platform — {provider}")
    models = body.get("models", [])

    if not provider:
        return JSONResponse({"error": "provider is required"}, 400)
    _KEYLESS_PROVIDERS = {"edge-tts"}
    if provider not in _KEYLESS_PROVIDERS and provider != "ollama" and not api_key:
        return JSONResponse({"error": "api_key is required"}, 400)

    if provider == "ollama":
        url = _normalize_ollama_url(url or PLATFORM_OLLAMA_URL)

    store = _load_connections()
    # Upsert: find existing platform connection for this provider
    existing = None
    for c in store["connections"]:
        if c.get("provider") == provider and (c.get("platform_hosted") or str(c.get("id", "")).startswith("platform_")):
            existing = c
            break

    if existing:
        if provider == "ollama":
            existing["api_key"] = ""
            existing["type"] = "ollama"
            existing["url"] = _normalize_ollama_url(url or existing.get("url") or PLATFORM_OLLAMA_URL)
        else:
            existing["api_key"] = api_key
            existing["url"] = url or existing.get("url", "")
            existing["type"] = existing.get("type", "external")
        existing["name"] = name
        existing["models"] = models or existing.get("models", [])
        existing["enabled"] = True
        existing["platform_hosted"] = True
        conn = existing
    else:
        conn = {
            "id": f"platform_{provider}",
            "name": name,
            "type": "ollama" if provider == "ollama" else "external",
            "provider": provider,
            "url": url,
            "api_key": "" if provider == "ollama" else api_key,
            "models": models,
            "enabled": True,
            "platform_hosted": True,
        }
        if provider == "ollama":
            conn["url"] = _normalize_ollama_url(url)
        store["connections"].append(conn)

    if provider == "ollama":
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(f"{conn['url'].rstrip('/')}/api/tags")
                resp.raise_for_status()
                conn["models"] = sorted(m["name"] for m in resp.json().get("models", []))
        except Exception:
            pass

    _save_connections(store)
    log.info("[admin] Saved platform key for provider=%s", provider)
    return JSONResponse({"ok": True, "connection": conn})


@app.post("/api/admin/platform-keys/test")
async def api_admin_test_platform_key(request: Request):
    """Test a platform API key by calling the /v1/models endpoint."""
    if not _check_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=403)

    body = await request.json()
    provider = body.get("provider", "").strip()
    url = body.get("url", "").strip().rstrip("/")
    api_key = body.get("api_key", "").strip()

    _KEYLESS_PROVIDERS = {"edge-tts"}
    if provider not in _KEYLESS_PROVIDERS and provider != "ollama" and not api_key:
        return JSONResponse({"ok": False, "error": "No API key provided"})

    # ElevenLabs uses a different endpoint
    if provider == "elevenlabs":
        test_url = f"{url}/v1/voices"
        headers = {"xi-api-key": api_key}
    elif provider == "ollama":
        test_url = f"{_normalize_ollama_url(url)}/api/tags"
        headers = {}
    elif provider == "edge-tts":
        # Self-hosted service — no API key needed
        test_url = f"{url}/v1/models"
        headers = {}
    elif provider == "google_gemini":
        # Google Gemini OpenAI-compatible endpoint
        test_url = f"{url}/models"
        headers = {"Authorization": f"Bearer {api_key}"}
    else:
        # Standard OpenAI-compatible /v1/models
        test_url = f"{url}/v1/models"
        headers = {"Authorization": f"Bearer {api_key}"}

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(test_url, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        if provider == "elevenlabs":
            count = len(data.get("voices", []))
        elif provider == "ollama":
            count = len(data.get("models", []))
        else:
            count = len(data.get("data", data.get("models", [])))
        return JSONResponse({"ok": True, "models_count": count})
    except httpx.HTTPStatusError as e:
        detail = str(e)
        try:
            detail = e.response.text[:200]
        except Exception:
            pass
        return JSONResponse({"ok": False, "error": f"HTTP {e.response.status_code}: {detail}"})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)})


@app.delete("/api/admin/platform-keys/{provider}")
async def api_admin_remove_platform_key(provider: str, request: Request):
    """Remove a platform-hosted connection."""
    if not _check_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=403)

    store = _load_connections()
    store["connections"] = [
        c for c in store["connections"]
        if not (c.get("platform_hosted") and c.get("provider") == provider)
    ]
    _save_connections(store)
    log.info("[admin] Removed platform key for provider=%s", provider)
    return JSONResponse({"ok": True})


@app.post("/api/admin/sync-openrouter-pricing")
async def api_admin_sync_openrouter_pricing(request: Request):
    """Manually trigger OpenRouter pricing sync. Admin only."""
    if not _check_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    result = await _sync_openrouter_pricing(force=True)
    return JSONResponse(result)


# ═══════════════════════════════════════════════════════════════════
#  ADMIN — USER ACCOUNT MANAGEMENT
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/admin/users")
async def api_admin_list_users(request: Request):
    """List all known users and their billing/activity summary. Admin only."""
    if not _check_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    return JSONResponse({"users": list_all_users()})


@app.delete("/api/admin/users/{user_id}")
async def api_admin_wipe_user(user_id: str, request: Request):
    """Wipe all billing/state data for a specific user (by UUID). Admin only."""
    if not _check_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    result = wipe_user_data(user_id)
    # Also remove per-user data directory
    import shutil
    from web.user_data import user_root, _validate_user_id
    try:
        uid = _validate_user_id(user_id)
        uroot = user_root(uid)
        if uroot.exists():
            shutil.rmtree(str(uroot), ignore_errors=True)
            result["user_data_dir_removed"] = True
    except Exception as exc:
        result["user_data_dir_error"] = str(exc)
    # Evict cached vault/faiss instances
    _user_vault_stores.pop(user_id, None)
    _user_faiss_memories.pop(user_id, None)
    return JSONResponse(result)


@app.post("/api/admin/users/wipe-by-email")
async def api_admin_wipe_user_by_email(request: Request):
    """Wipe all billing/state data for a user by email. Admin only."""
    if not _check_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    body = await request.json()
    email = body.get("email", "").strip()
    if not email:
        return JSONResponse({"error": "email is required"}, status_code=400)
    result = wipe_user_by_email(email)
    return JSONResponse(result)


@app.post("/api/admin/users/purge-inactive")
async def api_admin_purge_inactive(request: Request):
    """Purge all users inactive for more than N days. Admin only.

    Body: {"days": 90}  (optional, defaults to INACTIVE_ACCOUNT_DAYS)
    """
    if not _check_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    days = body.get("days", INACTIVE_ACCOUNT_DAYS)
    result = purge_inactive_users(days=days)
    return JSONResponse(result)


# ═══════════════════════════════════════════════════════════════════
#  ADMIN — VOICE ALLOWLIST
# ═══════════════════════════════════════════════════════════════════

@app.get("/admin/voices", response_class=HTMLResponse)
async def page_admin_voices(request: Request):
    """Admin page for managing which ElevenLabs voices users can see."""
    if not _check_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    settings = _load_settings()
    return templates.TemplateResponse(request, "admin_voices.html", {
        "page": "admin",
        "allowed_voices": settings.get("allowed_voices", []),
        "premium_voices": settings.get("premium_voices", []),
    })


@app.get("/api/admin/voices/all")
async def api_admin_voices_all(request: Request):
    """Fetch ALL ElevenLabs voices (admin only, unfiltered)."""
    if not _check_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    conn_data = _load_connections()
    el_conn = None
    for c in conn_data.get("connections", []):
        if c.get("provider") == "elevenlabs" and c.get("enabled", True):
            el_conn = c
            break
    if not el_conn or not el_conn.get("api_key"):
        return JSONResponse({"voices": [], "error": "No ElevenLabs connection configured"})
    url = f"{el_conn['url'].rstrip('/')}/v1/voices"
    headers = {"xi-api-key": el_conn["api_key"]}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        voices = [
            {"voice_id": v["voice_id"], "name": v["name"],
             "category": v.get("category", ""),
             "labels": v.get("labels", {})}
            for v in data.get("voices", [])
        ]
        return JSONResponse({"voices": voices})
    except Exception as e:
        return JSONResponse({"voices": [], "error": str(e)})


@app.put("/api/admin/voices/allowed")
async def api_admin_save_allowed_voices(request: Request):
    """Save the admin-curated allowlist of voice IDs."""
    if not _check_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    body = await request.json()
    allowed = body.get("allowed_voices", [])
    if not isinstance(allowed, list):
        return JSONResponse({"error": "allowed_voices must be a list"}, status_code=400)
    premium = body.get("premium_voices", [])
    if not isinstance(premium, list):
        return JSONResponse({"error": "premium_voices must be a list"}, status_code=400)
    settings = _load_settings()
    settings["allowed_voices"] = allowed
    settings["premium_voices"] = premium
    _save_settings(settings)
    return JSONResponse({"status": "ok", "count": len(allowed), "premium_count": len(premium)})


# ═══════════════════════════════════════════════════════════════════
#  CONNECTIONS API
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/connections")
async def api_connections_list():
    return _load_connections().get("connections", [])

@app.post("/api/connections")
async def api_connections_create(request: Request):
    body = await request.json()
    store = _load_connections()
    conn = {
        "id": str(uuid.uuid4())[:8],
        "name": body.get("name", "Untitled"),
        "type": body.get("type", "external"),
        "provider": body.get("provider", "openai"),
        "url": body.get("url", ""),
        "api_key": body.get("api_key", ""),
        "models": body.get("models", []),
        "enabled": body.get("enabled", True),
    }
    if conn.get("provider") == "ollama":
        conn["url"] = _normalize_ollama_url(conn.get("url"))
    store["connections"].append(conn)
    _save_connections(store)
    return conn

@app.put("/api/connections/{conn_id}")
async def api_connections_update(conn_id: str, request: Request):
    body = await request.json()
    store = _load_connections()
    for c in store["connections"]:
        if c["id"] == conn_id:
            for key in ("name", "type", "provider", "url", "api_key", "models", "enabled"):
                if key in body:
                    c[key] = body[key]
            if c.get("provider") == "ollama":
                c["url"] = _normalize_ollama_url(c.get("url"))
            break
    _save_connections(store)
    return {"ok": True}

@app.delete("/api/connections/{conn_id}")
async def api_connections_delete(conn_id: str):
    store = _load_connections()
    store["connections"] = [c for c in store["connections"] if c["id"] != conn_id]
    _save_connections(store)
    return {"ok": True}

@app.get("/api/connections/{conn_id}/models")
async def api_connections_fetch_models(conn_id: str):
    store = _load_connections()
    conn = next((c for c in store["connections"] if c["id"] == conn_id), None)
    if not conn:
        return JSONResponse({"error": "Not found"}, 404)

    provider = conn.get("provider", "openai")
    base_url = _normalize_ollama_url(conn.get("url")) if provider == "ollama" else conn["url"].rstrip("/")
    headers = {"Authorization": f"Bearer {conn['api_key']}"} if conn.get("api_key") else {}

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            if provider == "ollama":
                resp = await client.get(f"{base_url}/api/tags", headers=headers)
                resp.raise_for_status()
                models = sorted(m["name"] for m in resp.json().get("models", []))
            else:
                resp = await client.get(f"{base_url}/models", headers=headers)
                resp.raise_for_status()
                models = sorted(m["id"] for m in resp.json().get("data", []))
    except Exception as exc:
        return {"error": str(exc)}

    for c in store["connections"]:
        if c["id"] == conn_id:
            c["models"] = models
            break
    _save_connections(store)
    return {"models": models}

@app.post("/api/connections/probe-models")
async def api_connections_probe_models(request: Request):
    """Fetch available models from a connection without it being saved first.
    Useful when adding a new connection — avoids browser CORS restrictions."""
    body = await request.json()
    provider = body.get("provider", "openai")
    base_url = (body.get("url") or "").rstrip("/")
    if provider == "ollama":
        base_url = _normalize_ollama_url(base_url)
    api_key = body.get("api_key", "")
    if not base_url:
        return JSONResponse({"error": "URL is required"}, 400)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            if provider == "ollama":
                resp = await client.get(f"{base_url}/api/tags", headers=headers)
                resp.raise_for_status()
                models = sorted(m["name"] for m in resp.json().get("models", []))
            else:
                resp = await client.get(f"{base_url}/models", headers=headers)
                resp.raise_for_status()
                models = sorted(m["id"] for m in resp.json().get("data", []))
    except Exception as exc:
        return {"error": str(exc)}
    return {"models": models}


@app.get("/api/connections/all-models")
async def api_connections_all_models():
    """Return a map of connection_id → sorted model list for every enabled connection.
    Used by the model-router panel to populate model dropdowns.
    Live-fetches models from all platform-hosted providers, not just Ollama."""
    store = _load_connections()
    updated = False
    async with httpx.AsyncClient(timeout=15) as client:
        for conn in store.get("connections", []):
            if not conn.get("enabled"):
                continue
            provider = conn.get("provider", "openai")
            base_url = (conn.get("url") or "").rstrip("/")

            # Live-fetch models for platform-hosted connections (all providers)
            if conn.get("platform_hosted") or provider == "ollama":
                try:
                    if provider == "ollama":
                        base_url = _normalize_ollama_url(base_url)
                        conn["url"] = base_url
                        resp = await client.get(f"{base_url}/api/tags")
                        resp.raise_for_status()
                        conn["models"] = sorted(m["name"] for m in resp.json().get("models", []))
                        updated = True
                    elif provider == "elevenlabs":
                        # Skip voice providers for model list
                        pass
                    elif provider == "google_gemini":
                        headers = {"Authorization": f"Bearer {conn.get('api_key', '')}"} if conn.get("api_key") else {}
                        resp = await client.get(f"{base_url}/models", headers=headers)
                        resp.raise_for_status()
                        fetched = sorted(m["id"] for m in resp.json().get("data", resp.json().get("models", [])))
                        if fetched:
                            conn["models"] = fetched
                            updated = True
                    else:
                        # Standard OpenAI-compatible /v1/models
                        headers = {"Authorization": f"Bearer {conn.get('api_key', '')}"} if conn.get("api_key") else {}
                        fetch_url = f"{base_url}/models" if base_url.endswith("/v1") else f"{base_url}/v1/models"
                        resp = await client.get(fetch_url, headers=headers)
                        resp.raise_for_status()
                        fetched = sorted(m["id"] for m in resp.json().get("data", []))
                        if fetched:
                            conn["models"] = fetched
                            updated = True
                except Exception:
                    pass  # Keep existing static models if live fetch fails

    if updated:
        _save_connections(store)
    result: dict[str, list[str]] = {}
    for conn in store.get("connections", []):
        if not conn.get("enabled"):
            continue
        result[conn["id"]] = sorted(conn.get("models") or [])
    return result


# Curated "likely effectiveness" tiers for the free-trial model pin (lower =
# pinned higher). DeepSeek is pinned ahead of everything (handled in the sort
# key). The rest balance capability against credit burn: strong-yet-cheap
# workhorses first, credit-hungry flagships last.
def _trial_effectiveness_rank(name: str) -> int:
    """Rank a model by likely usefulness to a trial user (lower = better)."""
    n = name.lower()
    # 4 — credit-hungry flagships: drain the free balance fastest.
    if ("opus" in n or "gpt-5.5" in n or "gpt-5.4-pro" in n
            or "gpt-5.2-pro" in n or "gpt-5-pro" in n or "gpt-4.5" in n):
        return 4
    # 1 — strong + inexpensive workhorses: best value for a trial.
    # ("-mini" so the "mini" inside "gemini" is not matched here.)
    if "haiku" in n or "flash" in n or "-mini" in n:
        return 1
    # 3 — light / lower-capability (cheap but limited).
    if "-nano" in n or "gemma" in n or "-lite" in n:
        return 3
    # 2 — capable mid-tier.
    if ("sonnet" in n or "gemini" in n or "grok" in n or "llama" in n
            or "qwen" in n or "mistral" in n or "gpt-5" in n):
        return 2
    # Unknown — mid-tier by default.
    return 2


def _trial_model_sort_key(m: dict) -> tuple:
    """Sort key for the free-trial pin: DeepSeek first, then by likely
    effectiveness (capability per credit), with cost then name as tiebreakers."""
    name = str(m.get("model", "")).lower()
    est = m.get("est_credits_per_1k_tok", 0) or 0
    # Unpriced models (est == 0) sink within their tier rather than look free.
    cost_rank = est if est > 0 else 10 ** 9
    return (0 if "deepseek" in name else 1, _trial_effectiveness_rank(name), cost_rank, name)


def _trial_provider_sort_key(p: dict) -> tuple:
    """Sort key that floats DeepSeek-bearing / cheapest providers first (free-trial)."""
    models = p.get("models", []) or []
    has_deepseek = any("deepseek" in str(m.get("model", "")).lower() for m in models)
    priced = [m.get("est_credits_per_1k_tok", 0) for m in models if (m.get("est_credits_per_1k_tok", 0) or 0) > 0]
    min_cost = min(priced) if priced else 10 ** 9
    return (0 if has_deepseek else 1, min_cost)


@app.get("/api/platform/models")
async def api_platform_models(request: Request):
    """Public endpoint: Return available platform models for the chat dropdown.
    Groups models by provider with pricing hints and credit cost estimates.
    No admin access required — all authenticated users can see platform models.

    While a user is still spending their free welcome credits (has never bought a
    credit pack), cheap models — DeepSeek first, then by likely effectiveness (capability per credit) — are
    pinned to the top so the free balance stretches further. Every model stays
    selectable; only the ordering changes."""
    user = getattr(request.state, "user", None)
    credits = get_user_credits(user["id"]) if user else 0
    on_trial = bool(user) and not user_has_purchased_credits(user["id"])
    store = _load_connections()
    providers = []
    for conn in store.get("connections", []):
        if not conn.get("enabled") or not conn.get("platform_hosted"):
            continue
        if conn.get("provider") in ("elevenlabs", "edge-tts"):
            continue
        models_with_pricing = []
        prov = conn.get("provider", "openai")
        for m in (conn.get("models") or []):
            # Use metering get_price which handles OpenRouter provider/model passthrough
            from src.observability.metering import get_price as _metering_get_price
            input_rate, _cached, output_rate, _train = _metering_get_price(prov, m)
            # Estimate credits for a typical 1K token exchange (500 in, 500 out)
            est_usd = (input_rate * 500 / 1_000_000) + (output_rate * 500 / 1_000_000)
            est_credits = estimate_llm_credit_cost(est_usd) if est_usd > 0 else 0
            models_with_pricing.append({
                "model": m,
                "input_per_1m": input_rate,
                "output_per_1m": output_rate,
                "est_credits_per_1k_tok": est_credits,
            })
        # On the free trial balance, pin DeepSeek + cheapest models to the top.
        if on_trial:
            models_with_pricing.sort(key=_trial_model_sort_key)
        providers.append({
            "connection_id": conn["id"],
            "provider": prov,
            "name": conn.get("name", prov),
            "models": models_with_pricing,
        })
    # On trial, float providers that carry DeepSeek / the cheapest models first.
    if on_trial:
        providers.sort(key=_trial_provider_sort_key)
    return JSONResponse({
        "providers": providers,
        "credits_balance": credits,
        "trial": on_trial,
        "markup": LLM_MARKUP_MULTIPLIER,
    })


@app.post("/api/connections/refresh-all-models")
async def api_connections_refresh_all_models():
    """Live-fetch models from every enabled connection and persist them.
    Returns the updated connection_id → model-list map."""
    store = _load_connections()
    result: dict[str, list[str]] = {}
    async with httpx.AsyncClient(timeout=15) as client:
        for conn in store.get("connections", []):
            if not conn.get("enabled"):
                continue
            provider = conn.get("provider", "openai")
            base_url = (conn.get("url") or "").rstrip("/")
            if provider == "ollama":
                base_url = _normalize_ollama_url(base_url)
                conn["url"] = base_url
            headers = {"Authorization": f"Bearer {conn['api_key']}"} if conn.get("api_key") else {}
            try:
                if provider == "ollama":
                    resp = await client.get(f"{base_url}/api/tags", headers=headers)
                    resp.raise_for_status()
                    models = sorted(m["name"] for m in resp.json().get("models", []))
                else:
                    resp = await client.get(f"{base_url}/models", headers=headers)
                    resp.raise_for_status()
                    models = sorted(m["id"] for m in resp.json().get("data", []))
            except Exception:
                models = sorted(conn.get("models") or [])
            conn["models"] = models
            result[conn["id"]] = models
    _save_connections(store)
    return result


# ═══════════════════════════════════════════════════════════════════
#  PRICING API
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/pricing")
async def api_pricing_get():
    """Return the full pricing registry."""
    return _load_pricing()

@app.put("/api/pricing")
async def api_pricing_update(request: Request):
    """Replace the entire pricing registry. Admin only."""
    if not _check_admin(request):
        return JSONResponse({"error": "Admin access required"}, 403)
    body = await request.json()
    _save_pricing(body)
    try:
        from src.observability.metering import reset_pricing_cache
        reset_pricing_cache()
    except Exception:
        pass
    return {"ok": True}

@app.put("/api/pricing/{provider}/{model:path}")
async def api_pricing_set_model(provider: str, model: str, request: Request):
    """Update pricing for a single provider/model. Admin only."""
    if not _check_admin(request):
        return JSONResponse({"error": "Admin access required"}, 403)
    body = await request.json()
    pricing = _load_pricing()
    if provider not in pricing:
        pricing[provider] = {}
    entry = pricing[provider].get(model, {})
    for key in ("input_per_1m", "cached_input_per_1m", "output_per_1m", "training_per_1m"):
        if key in body:
            entry[key] = float(body[key])
    pricing[provider][model] = entry
    _save_pricing(pricing)
    try:
        from src.observability.metering import reset_pricing_cache
        reset_pricing_cache()
    except Exception:
        pass
    return {"ok": True, "pricing": entry}

@app.delete("/api/pricing/{provider}/{model:path}")
async def api_pricing_delete_model(provider: str, model: str, request: Request):
    """Remove pricing for a single model. Admin only."""
    if not _check_admin(request):
        return JSONResponse({"error": "Admin access required"}, 403)
    pricing = _load_pricing()
    if provider in pricing:
        pricing[provider].pop(model, None)
        if not pricing[provider]:
            del pricing[provider]
    _save_pricing(pricing)
    try:
        from src.observability.metering import reset_pricing_cache
        reset_pricing_cache()
    except Exception:
        pass
    return {"ok": True}

@app.get("/api/pricing/models")
async def api_pricing_all_models():
    """Return all models from all enabled connections + current pricing."""
    store = _load_connections()
    pricing = _load_pricing()
    result = []
    seen = set()
    for conn in store.get("connections", []):
        if not conn.get("enabled"):
            continue
        provider = conn.get("provider", "openai")
        conn_name = conn.get("name", provider)
        for m in conn.get("models", []):
            key = f"{provider}:{m}"
            if key in seen:
                continue
            seen.add(key)
            # Look up current pricing
            prov_prices = pricing.get(provider, {})
            mp = prov_prices.get(m)
            if not mp:
                for pk, pv in prov_prices.items():
                    if pk.startswith("_"):
                        continue
                    if isinstance(pv, dict) and m.startswith(pk):
                        mp = pv
                        break
            if not mp:
                mp = prov_prices.get("_default", {})
            result.append({
                "provider": provider,
                "connection": conn_name,
                "model": m,
                "input_per_1m": mp.get("input_per_1m", 0.0),
                "cached_input_per_1m": mp.get("cached_input_per_1m", 0.0),
                "output_per_1m": mp.get("output_per_1m", 0.0),
                "training_per_1m": mp.get("training_per_1m", 0.0),
            })
    return {"models": result}

@app.get("/api/pricing/cost-summary")
async def api_pricing_cost_summary(agent: str = "", period: str = "all", source: str = ""):
    """Return aggregated cost stats."""
    try:
        from src.observability.metering import read_cost_log, aggregate_costs
        from datetime import timedelta
        now_utc = datetime.now(timezone.utc)
        kwargs = {}
        if agent:
            kwargs["agent"] = agent
        if source:
            kwargs["source"] = source
        if period == "today":
            kwargs["since"] = now_utc.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        elif period == "week":
            kwargs["since"] = (now_utc - timedelta(days=7)).isoformat()
        elif period == "month":
            kwargs["since"] = (now_utc - timedelta(days=30)).isoformat()
        events = read_cost_log(**kwargs, limit=100000)
        return aggregate_costs(events)
    except Exception as exc:
        return {"error": str(exc)}

@app.get("/api/pricing/cost-log")
async def api_pricing_cost_log(
    agent: str = "", since: str = "", until: str = "",
    source: str = "", limit: int = 100,
):
    """Return recent cost log entries with optional date-range and source filtering."""
    try:
        from src.observability.metering import read_cost_log
        kwargs = {"limit": limit}
        if agent:
            kwargs["agent"] = agent
        if since:
            kwargs["since"] = since
        if until:
            kwargs["until"] = until
        if source:
            kwargs["source"] = source
        return {"events": read_cost_log(**kwargs)}
    except Exception as exc:
        return {"error": str(exc)}


# ═══════════════════════════════════════════════════════════════════
#  TTS API (ElevenLabs + Edge-TTS / Piper / XTTS)
# ═══════════════════════════════════════════════════════════════════

@app.post("/api/tts/speak")
async def api_tts_speak(request: Request):
    """Proxy text-to-speech via ElevenLabs.  Returns audio/mpeg stream."""
    body = await request.json()
    text = body.get("text", "").strip()
    voice_id = body.get("voice_id", "21m00Tcm4TlvDq8ikWAM")
    model_id = body.get("model_id", "eleven_multilingual_v2")

    if not text:
        return JSONResponse({"error": "No text provided"}, 400)

    # ── Ownership check: user must have purchased voice_tts ──
    user = getattr(request.state, "user", None)
    if user and not user_owns_item(user["id"], "voice_tts"):
        return JSONResponse({"error": "Voice TTS not unlocked. Purchase it in the Store first.",
                             "redirect": "/store"}, 403)

    conn_data = _load_connections()
    el_conn = None
    for c in conn_data.get("connections", []):
        if c.get("provider") == "elevenlabs" and c.get("enabled", True):
            el_conn = c
            break
    if not el_conn or not el_conn.get("api_key"):
        return JSONResponse({"error": "No ElevenLabs connection configured. Add one in Settings → Connections."}, 400)

    # ── Check if voice is premium ──
    is_premium = voice_id in set(_load_settings().get("premium_voices", []))

    # ── Pre-flight credit check (always metered at 2× cost) ──
    char_count_est = len(text)
    credit_cost = 0
    if user:
        credit_cost = estimate_tts_credit_cost(char_count_est, provider="elevenlabs", premium=is_premium)
        if credit_cost > 0:
            balance = get_user_credits(user["id"])
            if balance < credit_cost:
                return JSONResponse({
                    "error": "Insufficient credits for TTS usage",
                    "credits_needed": credit_cost,
                    "credits_balance": balance,
                    "redirect": "/store",
                }, status_code=402)
    elif not user:
        return JSONResponse({"error": "Login required for TTS.", "redirect": "/login"}, 401)

    url = f"{el_conn['url'].rstrip('/')}/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": el_conn["api_key"],
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": text, "model_id": model_id,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
        char_count = int(resp.headers.get("x-character-count", len(text)))

        # ── Deduct credits (always metered at 2× cost) ──
        if user and credit_cost > 0:
            try:
                # Recalculate with actual char count from response
                actual_cost = estimate_tts_credit_cost(char_count, provider="elevenlabs", premium=is_premium)
                label = f"tts:elevenlabs{'_premium' if is_premium else ''}:{char_count}chars"
                deduct_user_credits(user["id"], actual_cost, label)
            except Exception as exc:
                log.warning("[credits] TTS credit deduction failed: %s", exc)

        return Response(content=resp.content, media_type="audio/mpeg",
                        headers={"X-TTS-Characters": str(char_count), "X-TTS-Model": model_id,
                                 "X-TTS-Premium": "1" if is_premium else "0"})
    except httpx.HTTPStatusError as e:
        detail = str(e)
        try: detail = e.response.json().get("detail", {}).get("message", str(e))
        except Exception: pass
        return JSONResponse({"error": f"ElevenLabs error: {detail}"}, 502)
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


@app.get("/api/tts/voices")
async def api_tts_voices():
    """Fetch available voices from ElevenLabs, filtered by admin allowlist."""
    conn_data = _load_connections()
    el_conn = None
    for c in conn_data.get("connections", []):
        if c.get("provider") == "elevenlabs" and c.get("enabled", True):
            el_conn = c
            break
    if not el_conn or not el_conn.get("api_key"):
        return JSONResponse({"voices": []})
    url = f"{el_conn['url'].rstrip('/')}/v1/voices"
    headers = {"xi-api-key": el_conn["api_key"]}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        voices = [{"voice_id": v["voice_id"], "name": v["name"], "category": v.get("category", "")}
                  for v in data.get("voices", [])]
        # Apply admin allowlist filter
        settings = _load_settings()
        allowed = settings.get("allowed_voices", [])
        premium_set = set(settings.get("premium_voices", []))
        if allowed:
            allowed_set = set(allowed)
            voices = [v for v in voices if v["voice_id"] in allowed_set]
        for v in voices:
            v["premium"] = v["voice_id"] in premium_set
        return JSONResponse({"voices": voices})
    except Exception as e:
        return JSONResponse({"voices": [], "error": str(e)})


def _get_inworld_api_key():
    """Return the Inworld TTS API key (env var → settings fallback), or None."""
    key = os.environ.get("INWORLD_API_KEY", "").strip()
    if key:
        return key
    settings = _load_settings()
    return settings.get("api_keys", {}).get("inworld", "") or None


@app.post("/api/tts/inworld/speak")
async def api_tts_inworld_speak(request: Request):
    """Synthesise speech via Inworld TTS API (user provides own API key)."""
    body = await request.json()
    text = body.get("text", "").strip()
    voice_id = body.get("voice_id", "Ashley")
    model_id = body.get("model_id", "inworld-tts-1.5-mini")
    if not text:
        return JSONResponse({"error": "No text provided"}, 400)

    # ── Ownership check: user must have purchased voice_tts ──
    user = getattr(request.state, "user", None)
    if user and not user_owns_item(user["id"], "voice_tts"):
        return JSONResponse({"error": "Voice TTS not unlocked. Purchase it in the Store first.",
                             "redirect": "/store"}, 403)

    api_key = _get_inworld_api_key()
    if not api_key:
        return JSONResponse({"error": "No Inworld API key configured. Add it in Settings → API Keys."}, 400)

    char_count = len(text)
    url = "https://api.inworld.ai/tts/v1/voice"
    headers = {
        "Authorization": f"Basic {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"text": text, "voiceId": voice_id, "modelId": model_id}
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
        data = resp.json()
        audio_bytes = _b64.b64decode(data.get("audioContent", ""))
        return Response(content=audio_bytes, media_type="audio/mpeg",
                        headers={"X-TTS-Characters": str(char_count), "X-TTS-Provider": "inworld"})
    except httpx.HTTPStatusError as e:
        detail = str(e)
        try: detail = e.response.text[:300]
        except Exception: pass
        log.error("[tts] Inworld TTS HTTP error: %s", detail)
        return JSONResponse({"error": f"Inworld TTS error: {detail}"}, 502)
    except Exception as e:
        log.error("[tts] Inworld TTS exception: %s", e)
        return JSONResponse({"error": str(e)}, 500)


@app.get("/api/tts/inworld/voices")
async def api_tts_inworld_voices():
    """Fetch available voices from Inworld TTS API."""
    api_key = _get_inworld_api_key()
    if not api_key:
        return JSONResponse({"voices": [], "error": "No Inworld API key configured"})
    url = "https://api.inworld.ai/tts/v1/voices?filter=language%3Den"
    headers = {"Authorization": f"Basic {api_key}"}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
        data = resp.json()
        voices = []
        for v in data.get("voices", []):
            voices.append({
                "voice_id": v.get("voiceId", ""),
                "name": v.get("displayName", v.get("voiceId", "")),
                "description": v.get("description", ""),
                "tags": v.get("tags", []),
            })
        return JSONResponse({"voices": voices})
    except Exception as e:
        log.error("[tts] Inworld voices fetch error: %s", e)
        return JSONResponse({"voices": [], "error": str(e)})


# ═══════════════════════════════════════════════════════════════════
#  STT API (ElevenLabs)
# ═══════════════════════════════════════════════════════════════════

def _get_elevenlabs_conn():
    """Return the first enabled ElevenLabs connection or None."""
    for c in _load_connections().get("connections", []):
        if c.get("provider") == "elevenlabs" and c.get("enabled", True):
            return c
    return None


@app.post("/api/stt/elevenlabs")
async def api_stt_elevenlabs(request: Request):
    """Transcribe uploaded audio via ElevenLabs Speech-to-Text API."""
    # ── Ownership check: user must have purchased voice_stt ──
    user = getattr(request.state, "user", None)
    if user and not user_owns_item(user["id"], "voice_stt"):
        return JSONResponse({"error": "Voice STT not unlocked. Purchase it in the Store first.",
                             "redirect": "/store"}, 403)

    conn = _get_elevenlabs_conn()
    if not conn or not conn.get("api_key"):
        return JSONResponse({"error": "No ElevenLabs connection configured. Add one in Settings → API Keys."}, 400)

    form = await request.form()
    audio_file = form.get("file")
    language = form.get("language", "en")
    if not audio_file:
        return JSONResponse({"error": "No audio file provided"}, 400)
    audio_bytes = await audio_file.read()
    filename = getattr(audio_file, 'filename', 'audio.webm') or 'audio.webm'

    # ── Pre-flight credit check (always metered at 2× cost) ──
    estimated_seconds = max(len(audio_bytes) / (16 * 1024), 1.0)
    credit_cost = 0
    if user:
        credit_cost = estimate_stt_credit_cost(estimated_seconds, provider="elevenlabs")
        if credit_cost > 0:
            balance = get_user_credits(user["id"])
            if balance < credit_cost:
                return JSONResponse({
                    "error": "Insufficient credits for STT usage",
                    "credits_needed": credit_cost,
                    "credits_balance": balance,
                    "redirect": "/store",
                }, status_code=402)
    elif not user:
        return JSONResponse({"error": "Login required for STT.", "redirect": "/login"}, 401)

    url = f"{conn['url'].rstrip('/')}/v1/speech-to-text"
    headers = {"xi-api-key": conn["api_key"]}
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                url,
                headers=headers,
                files={"file": (filename, audio_bytes, "audio/webm")},
                data={"model_id": "scribe_v1", "language_code": language},
            )
            resp.raise_for_status()
            data = resp.json()

        text = data.get("text", "").strip()

        # ── Deduct credits (always metered at 2× cost) ──
        if user and credit_cost > 0:
            try:
                deduct_user_credits(user["id"], credit_cost, f"stt:elevenlabs:{int(estimated_seconds)}s")
            except Exception as exc:
                log.warning("[credits] STT credit deduction failed: %s", exc)

        # Estimate USD cost for frontend session tracker (matches the LLM markup multiplier)
        stt_usd = (estimated_seconds / 60) * 0.006 * LLM_MARKUP_MULTIPLIER
        return JSONResponse({"text": text, "provider": "elevenlabs",
                             "stt_cost": round(stt_usd, 6), "audio_seconds": round(estimated_seconds, 1)})
    except httpx.HTTPStatusError as e:
        detail = str(e)
        try: detail = e.response.json().get("detail", {}).get("message", str(e))
        except Exception: pass
        return JSONResponse({"error": f"ElevenLabs STT error: {detail}"}, 502)
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


# ═══════════════════════════════════════════════════════════════════
#  BURST MODE API
# ═══════════════════════════════════════════════════════════════════

@app.post("/api/chat/run")
async def api_chat_run(req: ChatRequest, request: Request):
    """Launch a burst session and return the session ID."""
    user = getattr(request.state, "user", None)
    if not _can_access_agent(req.agent, _get_user_id(request)):
        return JSONResponse({"error": "Agent not available"}, status_code=403)

    # Resolve connection to check platform_hosted status
    conn = _resolve_connection(req.connection_id, req.agent)
    is_platform = conn.get("platform_hosted", False) if conn else False

    # ── Pre-flight credit check for platform-hosted keys ──
    if is_platform:
        if not user:
            return JSONResponse({"error": "Login required for platform-hosted AGI loop."}, 401)
        balance = get_user_credits(user["id"])
        if balance <= 0:
            return JSONResponse({
                "error": "Insufficient credits. Purchase more in the Store.",
                "credits_balance": 0,
                "redirect": "/store",
            }, status_code=402)

    session_id = str(uuid.uuid4())[:8]
    chat_id = req.chat_id
    if not chat_id:
        chat = _create_new_chat(req.agent, req.mode, {
            "burst_ticks": req.burst_ticks, "max_steps": req.max_steps,
            "connection_id": req.connection_id,
        })
        chat_id = chat["id"]

    if req.stimulus:
        chat_data = _load_chat(chat_id)
        if chat_data:
            chat_data["messages"].append({"role": "user", "text": req.stimulus,
                                          "time": datetime.now(timezone.utc).isoformat()})
            chat_data["updated"] = datetime.now(timezone.utc).isoformat()
            _save_chat(chat_id, chat_data)
            _update_chat_index_entry(chat_id, {"updated": chat_data["updated"]})

    cmd = [
        sys.executable, "-m", "src.run_burst",
        "--profile", req.agent,
        "--burst-ticks", str(req.burst_ticks),
        "--max-steps", str(req.max_steps),
        "--stimulus", req.stimulus,
    ]
    env = os.environ.copy()
    # Tell the engine subprocess what cost source to tag events with
    env["ORION_COST_SOURCE"] = "platform" if is_platform else "user"
    if req.connection_id:
        for c in _load_connections().get("connections", []):
            if c["id"] == req.connection_id:
                if c.get("url"): env["OPENAI_BASE_URL"] = c["url"]
                if c.get("api_key"): env["OPENAI_API_KEY"] = c["api_key"]
                break
    if req.model_override:
        env["AGENT_MODEL_OVERRIDE"] = req.model_override

    started_ts = datetime.now(timezone.utc).isoformat()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, cwd=str(_PROJECT_ROOT), env=env)
        q: queue.Queue = queue.Queue()
        def _reader(p, qu, cid):
            try:
                for line in iter(p.stdout.readline, ""):
                    stripped = line.rstrip("\n")
                    qu.put(stripped)
                    _append_agent_line_to_chat(cid, stripped)
            except Exception: pass
            finally: qu.put(None)
        t = threading.Thread(target=_reader, args=(proc, q, chat_id), daemon=True)
        t.start()
        _chat_sessions[session_id] = {
            "process": proc, "queue": q, "chat_id": chat_id,
            "agent": req.agent, "mode": req.mode, "stimulus": req.stimulus,
            "started": started_ts,
            "output_lines": [], "status": "running",
            "platform_hosted": is_platform,
            "user_id": user["id"] if user else None,
            "credits_deducted": False,
        }
        return JSONResponse({"session_id": session_id, "chat_id": chat_id, "status": "started"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


@app.get("/api/chat/status/{session_id}")
async def api_chat_status(session_id: str):
    """Poll for output from a running chat session."""
    session = _chat_sessions.get(session_id)
    if not session:
        return JSONResponse({"error": "Session not found"}, 404)
    proc = session["process"]
    q = session["queue"]
    new_lines = []
    while True:
        try: line = q.get_nowait()
        except queue.Empty: break
        if line is None: break
        new_lines.append(line)
        session["output_lines"].append(line)

    # ── Rolling credit enforcement for platform-hosted burst sessions ──
    # On every poll while the process is still running, check accumulated
    # cost against the user's remaining balance and kill if exceeded.
    _burst_killed = False
    if (
        proc.poll() is None
        and session.get("platform_hosted")
        and session.get("user_id")
    ):
        try:
            from src.observability.metering import read_cost_log
            events = read_cost_log(
                since=session["started"],
                source="platform",
                limit=100000,
            )
            chat_events = [e for e in events if e.get("chat_id") == session["chat_id"]]
            total_usd = sum(e.get("cost", {}).get("total_cost", 0.0) for e in chat_events)
            total_tok = sum(e.get("usage", {}).get("total_tokens", 0) for e in chat_events)
            credit_cost = estimate_llm_credit_cost_safe(total_usd, total_tok)
            if credit_cost > 0:
                balance = get_user_credits(session["user_id"])
                if credit_cost >= balance:
                    # Kill the subprocess — user can't afford more calls
                    proc.terminate()
                    session["status"] = "stopped"
                    _burst_killed = True
                    log.warning(
                        "[credits] Burst killed — credits exhausted. "
                        "cost=%d credits, balance=%d, session=%s",
                        credit_cost, balance, session_id,
                    )
                    # Immediately deduct what was accumulated
                    session["credits_deducted"] = True
                    deduct_user_credits(
                        session["user_id"],
                        min(credit_cost, balance),
                        f"agi_burst:{session.get('agent', '')}:{len(chat_events)}calls:killed",
                    )
        except Exception as exc:
            log.warning("[credits] Rolling burst credit check failed: %s", exc)

    retcode = proc.poll()
    if retcode is not None and not _burst_killed:
        session["status"] = "completed" if retcode == 0 else "error"

        # ── Post-burst credit deduction for platform-hosted keys ──
        if (
            session.get("platform_hosted")
            and session.get("user_id")
            and not session.get("credits_deducted")
        ):
            session["credits_deducted"] = True
            try:
                from src.observability.metering import read_cost_log
                events = read_cost_log(
                    since=session["started"],
                    source="platform",
                    limit=100000,
                )
                # Only charge for events belonging to this chat
                chat_events = [e for e in events if e.get("chat_id") == session["chat_id"]]
                total_usd = sum(e.get("cost", {}).get("total_cost", 0.0) for e in chat_events)
                total_tok = sum(e.get("usage", {}).get("total_tokens", 0) for e in chat_events)
                credit_cost = estimate_llm_credit_cost_safe(total_usd, total_tok)
                if credit_cost > 0:
                    deduct_user_credits(
                        session["user_id"],
                        credit_cost,
                        f"agi_burst:{session.get('agent', '')}:{len(chat_events)}calls",
                    )
                    log.info("[credits] Burst deduction: %d credits for session %s", credit_cost, session_id)
            except Exception as exc:
                log.warning("[credits] Burst credit deduction failed: %s", exc)

    return JSONResponse({
        "session_id": session_id, "status": session["status"],
        "new_lines": new_lines, "total_lines": len(session["output_lines"]),
        "agent": session["agent"], "mode": session["mode"],
    })


@app.post("/api/chat/stop/{session_id}")
async def api_chat_stop(session_id: str):
    """Stop a running chat session."""
    session = _chat_sessions.get(session_id)
    if not session:
        return JSONResponse({"error": "Session not found"}, 404)
    proc = session["process"]
    if proc.poll() is None:
        proc.terminate()
        session["status"] = "stopped"
    return JSONResponse({"status": session["status"]})


# ═══════════════════════════════════════════════════════════════════
#  OLLAMA MODELS
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/ollama/models")
async def api_ollama_models(url: str = Query("http://orionforge-engine-ollama.flycast:11434")):
    """Fetch locally available Ollama models."""
    models = []
    try:
        api_url = url.rstrip("/") + "/api/tags"
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(api_url)
            resp.raise_for_status()
            data = resp.json()
        for m in data.get("models", []):
            name = m.get("name", "")
            size_bytes = m.get("size", 0)
            size_gb = round(size_bytes / (1024 ** 3), 1) if size_bytes else None
            details = m.get("details", {})
            models.append({
                "name": name,
                "size": f"{size_gb} GB" if size_gb else "unknown",
                "family": details.get("family", ""),
                "parameter_size": details.get("parameter_size", ""),
                "quantization": details.get("quantization_level", ""),
                "format": details.get("format", ""),
                "modified_at": m.get("modified_at", ""),
            })
        return JSONResponse({"models": models, "source": "api", "status": "connected"})
    except Exception:
        pass
    try:
        ollama_exe = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe")
        if not os.path.exists(ollama_exe):
            ollama_exe = "ollama"
        result = subprocess.run([ollama_exe, "list"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            for line in lines[1:]:
                parts = line.split()
                if parts:
                    name = parts[0]
                    size_str = ""
                    for i, p in enumerate(parts):
                        if p.upper().endswith("GB") or p.upper().endswith("MB"):
                            size_str = p; break
                    models.append({"name": name, "size": size_str or "unknown", "family": "",
                                   "parameter_size": "", "quantization": "", "format": "", "modified_at": ""})
            return JSONResponse({"models": models, "source": "cli", "status": "connected"})
    except Exception:
        pass
    return JSONResponse({"models": [], "source": "none", "status": "disconnected",
                         "error": "Could not connect to Ollama."}, 503)


# ═══════════════════════════════════════════════════════════════════
#  CHAT BACKGROUND UPLOAD
# ═══════════════════════════════════════════════════════════════════

@app.post("/api/settings/chat-background")
async def api_upload_chat_background(request: Request, file: UploadFile = File(...)):
    """Upload a background image for the chat page."""
    uid = _get_user_id(request)
    allowed = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".avif"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed:
        return JSONResponse({"error": f"File type {ext} not allowed"}, 400)
    filename = f"chat_bg{ext}"
    if uid and uid != "__local__":
        dest_dir = user_uploads_dir(uid)
    else:
        dest_dir = _UPLOADS_DIR
    dest = dest_dir / filename
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)
    settings = _load_settings(user_id=uid)
    settings["chat_background"] = f"/uploads/{filename}"
    _save_settings(settings, user_id=uid)
    return JSONResponse({"url": settings["chat_background"], "status": "ok"})


@app.delete("/api/settings/chat-background")
async def api_delete_chat_background(request: Request):
    """Remove the chat background image."""
    uid = _get_user_id(request)
    settings = _load_settings(user_id=uid)
    old_bg = settings.get("chat_background")
    if old_bg:
        if uid and uid != "__local__":
            old_path = user_uploads_dir(uid) / os.path.basename(old_bg)
        else:
            old_path = _UPLOADS_DIR / os.path.basename(old_bg)
        if old_path.exists():
            old_path.unlink()
    settings["chat_background"] = None
    _save_settings(settings, user_id=uid)
    return JSONResponse({"status": "ok"})


@app.put("/api/settings/chat-defaults")
async def api_save_chat_defaults(request: Request):
    """Persist the user's last-selected agent, connection, and model."""
    body = await request.json()
    uid = _get_user_id(request)
    settings = _load_settings(user_id=uid)
    defaults = settings.get("chat_defaults", {})
    for key in ("last_agent", "last_connection", "last_model"):
        if key in body:
            defaults[key] = body[key]
    settings["chat_defaults"] = defaults
    _save_settings(settings, user_id=uid)
    return JSONResponse({"status": "ok"})


@app.put("/api/settings/default-agent")
async def api_save_default_agent(request: Request):
    """Save the user's preferred default agent for new chats."""
    body = await request.json()
    agent = body.get("default_agent", "")
    allowed = _list_unlocked_agents(request)
    if agent and agent not in allowed:
        return JSONResponse({"error": "Agent not available"}, status_code=400)
    uid = _get_user_id(request)
    settings = _load_settings(user_id=uid)
    settings["default_agent"] = agent
    _save_settings(settings, user_id=uid)
    return JSONResponse({"status": "ok"})


@app.put("/api/settings/timezone")
async def api_save_timezone(request: Request):
    """Save timezone preferences."""
    body = await request.json()
    uid = _get_user_id(request)
    settings = _load_settings(user_id=uid)
    settings["timezone"] = body.get("timezone", "auto")
    settings["timezone_name"] = body.get("timezone_name", None)
    settings["timezone_offset_hours"] = body.get("timezone_offset_hours", None)
    _save_settings(settings, user_id=uid)
    return JSONResponse({"status": "ok"})


@app.put("/api/settings/pinned-models")
async def api_save_pinned_models(request: Request):
    """Save the user's pinned/favorite model list for the chat dropdown."""
    body = await request.json()
    pinned = body.get("pinned_models", [])
    if not isinstance(pinned, list):
        return JSONResponse({"error": "pinned_models must be a list"}, 400)
    uid = _get_user_id(request)
    settings = _load_settings(user_id=uid)
    settings["pinned_models"] = pinned[:50]  # cap at 50 favorites
    _save_settings(settings, user_id=uid)
    return JSONResponse({"status": "ok", "pinned_models": settings["pinned_models"]})


# ═══════════════════════════════════════════════════════════════════
#  API KEYS
# ═══════════════════════════════════════════════════════════════════

@app.put("/api/settings/api-keys")
async def api_save_api_keys(request: Request):
    """Save per-user API keys for LLM providers (OpenAI, Anthropic, DeepSeek, Google Gemini, OpenRouter, Ollama URL)."""
    body = await request.json()
    user_id = _get_user_id(request)
    ollama_url = _normalize_ollama_url(body.get("ollama_url", PLATFORM_OLLAMA_URL))
    new_keys = {
        "openai":        body.get("openai", ""),
        "anthropic":     body.get("anthropic", ""),
        "deepseek":      body.get("deepseek", ""),
        "google_gemini": body.get("google_gemini", ""),
        "openrouter":    body.get("openrouter", ""),
        "ollama_url":    ollama_url,
        "inworld":       body.get("inworld", ""),
    }
    # Merge: empty values in the request preserve existing keys
    existing = load_user_keys(user_id, "api_keys")
    if not existing:
        existing = _load_settings().get("api_keys", {})
    for k, v in new_keys.items():
        if not v and k != "ollama_url":
            new_keys[k] = existing.get(k, "")
    # Save to per-user encrypted store
    save_user_keys(user_id, "api_keys", new_keys)
    # Also save to per-user settings for backward compat
    settings = _load_settings(user_id=user_id)
    settings["api_keys"] = new_keys
    _save_settings(settings, user_id=user_id)
    return JSONResponse({"status": "ok"})


@app.get("/api/settings/api-keys")
async def api_get_api_keys(request: Request):
    """Return saved API keys (masked for security)."""
    user_id = _get_user_id(request)
    # Try per-user store first, fall back to global settings
    keys = load_user_keys(user_id, "api_keys")
    if not keys:
        settings = _load_settings()
        keys = settings.get("api_keys", {})
    masked = {}
    for k, v in keys.items():
        if k == "ollama_url":
            masked[k] = v
        elif v and len(v) > 8:
            masked[k] = v[:4] + "•" * (len(v) - 8) + v[-4:]
        elif v:
            masked[k] = "•" * len(v)
        else:
            masked[k] = ""
    # Also return a "has_key" map so the UI can show ✓ indicators
    has_keys = {k: bool(v and k != "ollama_url") for k, v in keys.items()}
    masked["_configured"] = has_keys
    return JSONResponse(masked)


# User models catalog: user-provider-key → (display name, pricing.yaml section, openrouter prefix)
# Models are derived dynamically from pricing.yaml so every model the platform
# knows about automatically appears in the User Models dropdown.
_USER_MODEL_CATALOG: dict[str, tuple[str, str, str | None]] = {
    "openai":        ("OpenAI",         "openai",   None),
    "anthropic":     ("Anthropic",      "anthropic", None),
    "deepseek":      ("DeepSeek",       "deepseek", None),
    "google_gemini": ("Google Gemini",  "google",   None),
    "xai":           ("xAI (Grok)",     "xai",      None),
    "mistral":       ("Mistral",        "mistral",  None),
    "openrouter":    ("OpenRouter",     "openrouter", "aggregate"),
}

# OpenRouter "vendor/model" prefixes per pricing.yaml section
_OPENROUTER_VENDOR_PREFIX: dict[str, str] = {
    "openai":    "openai",
    "anthropic": "anthropic",
    "deepseek":  "deepseek",
    "google":    "google",
    "xai":       "x-ai",
    "mistral":   "mistralai",
}

# Embedding / non-chat models to exclude from the catalog
_NON_CHAT_MODEL_SUBSTRINGS = ("embedding", "embed", "whisper", "tts", "moderation", "dall-e", "image")

# Provider → (base URL, API key env var) for building dynamic connections
_USER_PROVIDER_URLS: dict[str, str] = {
    "openai":        "https://api.openai.com/v1",
    "anthropic":     "https://api.anthropic.com/v1",
    "deepseek":      "https://api.deepseek.com/v1",
    "google_gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "xai":           "https://api.x.ai/v1",
    "mistral":       "https://api.mistral.ai/v1",
    "openrouter":    "https://openrouter.ai/api/v1",
}


def _is_chat_model(name: str) -> bool:
    low = name.lower()
    return not any(s in low for s in _NON_CHAT_MODEL_SUBSTRINGS)


def _models_for_provider(pricing: dict, section: str) -> list[str]:
    """Return chat-capable model ids from a pricing.yaml provider section."""
    sect = pricing.get(section) or {}
    if not isinstance(sect, dict):
        return []
    out = [m for m in sect.keys() if isinstance(m, str) and not m.startswith("_") and _is_chat_model(m)]
    out.sort()
    return out


def _openrouter_models(pricing: dict) -> list[str]:
    """Build the OpenRouter catalog as 'vendor/model' ids from every known section."""
    out: list[str] = []
    for section, vendor in _OPENROUTER_VENDOR_PREFIX.items():
        for m in _models_for_provider(pricing, section):
            out.append(f"{vendor}/{m}")
    out.sort()
    return out


@app.get("/api/user/models")
async def api_user_models(request: Request):
    """Return available LLM models based on which API keys the user has configured.

    Reads the per-user encrypted key vault first (production / multi-user),
    then falls back to settings.json (local dev / legacy).
    """
    user_id = _get_user_id(request)
    keys = load_user_keys(user_id, "api_keys") or {}
    if not keys:
        keys = _load_settings(user_id=user_id).get("api_keys", {})
    pricing = _load_pricing()
    providers = []
    for provider_key, (display_name, section, special) in _USER_MODEL_CATALOG.items():
        api_val = keys.get(provider_key, "")
        if not (api_val and len(api_val) >= 4):
            continue
        models = _openrouter_models(pricing) if special == "aggregate" else _models_for_provider(pricing, section)
        if not models:
            continue
        providers.append({
            "provider": provider_key,
            "name": display_name,
            "models": models,
        })
    return JSONResponse({"providers": providers})


# ═══════════════════════════════════════════════════════════════════
#  IMAGE GENERATION SETTINGS & API
# ═══════════════════════════════════════════════════════════════════

@app.put("/api/settings/image")
async def api_save_image_settings(request: Request):
    """Save image-generation provider keys and preferred model."""
    body = await request.json()
    user_id = _get_user_id(request)
    from web.key_vault import _SECRET_FIELDS as _SF
    image_secret_fields = _SF.get("image", set())
    image_keys = {
        "preferred":          body.get("preferred", "none"),
        "openai_api_key":     body.get("openai_api_key", ""),
        "use_platform_openai": body.get("use_platform_openai", False),
        "google_api_key":     body.get("google_api_key", ""),
        "use_platform_google": body.get("use_platform_google", False),
        "stability_api_key":  body.get("stability_api_key", ""),
        "use_platform_stability": body.get("use_platform_stability", False),
        "ideogram_api_key":   body.get("ideogram_api_key", ""),
        "replicate_api_key":  body.get("replicate_api_key", ""),
        "fal_api_key":        body.get("fal_api_key", ""),
        "leonardo_api_key":   body.get("leonardo_api_key", ""),
        "midjourney_url":     body.get("midjourney_url", ""),
        "midjourney_api_key": body.get("midjourney_api_key", ""),
    }
    # Merge: empty secret values preserve existing keys
    existing = load_user_keys(user_id, "image")
    if not existing:
        existing = _load_settings().get("image", {})
    for k in image_secret_fields:
        if not image_keys.get(k):
            image_keys[k] = existing.get(k, "")
    save_user_keys(user_id, "image", image_keys)
    settings = _load_settings(user_id=user_id)
    settings["image"] = image_keys
    _save_settings(settings, user_id=user_id)
    return JSONResponse({"status": "ok"})


@app.get("/api/settings/image")
async def api_get_image_settings(request: Request):
    """Return current image generation settings (keys masked)."""
    user_id = _get_user_id(request)
    keys = load_user_keys(user_id, "image")
    if not keys:
        settings = _load_settings()
        keys = settings.get("image", {"preferred": "none"})
    # Mask secret fields for the response
    from web.key_vault import _SECRET_FIELDS as _SF
    masked = dict(keys)
    for field in _SF.get("image", set()):
        raw = masked.get(field, "")
        masked[field] = mask_value(raw) if raw else ""
    return JSONResponse(masked)


@app.post("/api/image/generate")
async def api_image_generate(request: Request):
    """Generate an image using the preferred (or requested) provider."""
    user = getattr(request.state, "user", None)
    body = await request.json()
    prompt = body.get("prompt", "").strip()
    if not prompt:
        return JSONResponse({"error": "No prompt provided"}, 400)

    settings = _load_settings()
    img_cfg = settings.get("image", {})
    provider = body.get("provider") or img_cfg.get("preferred", "none")

    if provider == "none":
        return JSONResponse({"error": "No image provider configured. Set one in Settings → Image Generation."}, 400)

    # ── Credit pre-flight (image generation uses platform keys) ──
    credit_cost = estimate_image_credit_cost(provider) if user else 0
    if user and credit_cost > 0:
        balance = get_user_credits(user["id"])
        if balance < credit_cost:
            return JSONResponse({
                "error": "Insufficient credits for image generation",
                "credits_needed": credit_cost,
                "credits_balance": balance,
                "redirect": "/store",
            }, status_code=402)

    result = await _generate_image(provider, prompt, img_cfg, settings)
    if "error" in result:
        return JSONResponse(result, 502)

    if user and credit_cost > 0:
        try:
            deduct_user_credits(user["id"], credit_cost, f"image:{provider}")
        except Exception as exc:
            log.warning("[credits] image credit deduction failed: %s", exc)
    return JSONResponse(result)


# ═══════════════════════════════════════════════════════════════════
#  VIDEO GENERATION SETTINGS & API
# ═══════════════════════════════════════════════════════════════════

@app.put("/api/settings/video")
async def api_save_video_settings(request: Request):
    """Save video-generation provider keys and preferred model."""
    body = await request.json()
    user_id = _get_user_id(request)
    from web.key_vault import _SECRET_FIELDS as _SF
    video_secret_fields = _SF.get("video", set())
    video_keys = {
        "preferred":         body.get("preferred", "none"),
        "google_api_key":    body.get("google_api_key", ""),
        "aspect_ratio":      body.get("aspect_ratio", "16:9"),
        "duration_seconds":  body.get("duration_seconds", 8),
    }
    # Preserve existing secret values if empty fields are sent
    existing = load_user_keys(user_id, "video")
    if not existing:
        existing = _load_settings().get("video", {})
    for k in video_secret_fields:
        if not video_keys.get(k):
            video_keys[k] = existing.get(k, "")
    save_user_keys(user_id, "video", video_keys)
    settings = _load_settings(user_id=user_id)
    settings["video"] = video_keys
    _save_settings(settings, user_id=user_id)
    return JSONResponse({"status": "ok"})


@app.get("/api/settings/video")
async def api_get_video_settings(request: Request):
    """Return current video generation settings (keys masked)."""
    user_id = _get_user_id(request)
    from web.key_vault import _SECRET_FIELDS as _SF
    keys = load_user_keys(user_id, "video")
    if not keys:
        settings = _load_settings()
        keys = settings.get("video", {"preferred": "none"})
    masked = dict(keys)
    for field in _SF.get("video", set()):
        raw = masked.get(field, "")
        masked[field] = mask_value(raw) if raw else ""
    return JSONResponse(masked)


@app.post("/api/video/generate")
async def api_video_generate(request: Request):
    """Generate a video using the preferred (or requested) Veo provider."""
    user = getattr(request.state, "user", None)
    body = await request.json()
    prompt = body.get("prompt", "").strip()
    if not prompt:
        return JSONResponse({"error": "No prompt provided"}, 400)

    settings = _load_settings()
    vid_cfg = settings.get("video", {})
    provider = body.get("provider") or vid_cfg.get("preferred", "none")

    if provider == "none":
        return JSONResponse(
            {"error": "No video provider configured. Set one in Settings → Video Generation."},
            400,
        )

    # ── Credit pre-flight (Veo video is billed per second — the priciest media op) ──
    _vid_secs = vid_cfg.get("duration_seconds", 8)
    credit_cost = estimate_video_credit_cost(provider, _vid_secs) if user else 0
    if user and credit_cost > 0:
        balance = get_user_credits(user["id"])
        if balance < credit_cost:
            return JSONResponse({
                "error": "Insufficient credits for video generation",
                "credits_needed": credit_cost,
                "credits_balance": balance,
                "redirect": "/store",
            }, status_code=402)

    result = await _generate_video(
        provider, prompt, vid_cfg, settings,
        save_dir=_VIDEO_UPLOADS_DIR,
    )
    if "error" in result:
        return JSONResponse(result, 502)

    if user and credit_cost > 0:
        try:
            deduct_user_credits(user["id"], credit_cost, f"video:{provider}:{int(_vid_secs)}s")
        except Exception as exc:
            log.warning("[credits] video credit deduction failed: %s", exc)
    return JSONResponse(result)


# ═══════════════════════════════════════════════════════════════════
#  VOICE — STT & TTS
# ═══════════════════════════════════════════════════════════════════

@app.put("/api/settings/voice")
async def api_save_voice_settings(request: Request):
    """Save STT and TTS settings (keys, URLs, voice selections)."""
    body = await request.json()
    user_id = _get_user_id(request)
    settings = _load_settings()
    from web.key_vault import _SECRET_FIELDS as _SF
    tts_secret_fields = _SF.get("tts", set())

    stt = body.get("stt", {})
    settings["stt"] = {
        "provider":             stt.get("provider", "none"),
    }

    tts = body.get("tts", {})
    tts_data = {
        "provider":             tts.get("provider", "none"),
        "elevenlabs_api_key":   tts.get("elevenlabs_api_key", ""),
        "use_platform_elevenlabs": tts.get("use_platform_elevenlabs", False),
        "elevenlabs_voice_id":  tts.get("elevenlabs_voice_id", ""),
        "elevenlabs_voice_name":tts.get("elevenlabs_voice_name", ""),
        "openedai_cloud_url":       tts.get("openedai_cloud_url", ""),
        "openedai_cloud_api_key":   tts.get("openedai_cloud_api_key", ""),
        "use_platform_openedai_cloud": tts.get("use_platform_openedai_cloud", False),
        "openedai_cloud_voice":     tts.get("openedai_cloud_voice", ""),
        "openedai_cloud_model":     tts.get("openedai_cloud_model", "tts-1"),
    }
    # Merge: empty secret values preserve existing keys
    existing = load_user_keys(user_id, "tts")
    if not existing:
        existing = settings.get("tts", {})
    for k in tts_secret_fields:
        if not tts_data.get(k):
            tts_data[k] = existing.get(k, "")
    settings["tts"] = tts_data
    save_user_keys(user_id, "tts", tts_data)

    _save_settings(settings)
    return JSONResponse({"status": "ok"})


@app.get("/api/settings/voice")
async def api_get_voice_settings(request: Request):
    """Return current STT / TTS configuration (keys masked)."""
    settings = _load_settings()
    tts = dict(settings.get("tts", {"provider": "none"}))
    # Mask TTS secret fields
    from web.key_vault import _SECRET_FIELDS as _SF
    for field in _SF.get("tts", set()):
        raw = tts.get(field, "")
        tts[field] = mask_value(raw) if raw else ""
    return JSONResponse({
        "stt": settings.get("stt", {"provider": "none"}),
        "tts": tts,
    })


# ── Voice provider proxy endpoints (fetch available voices) ──────

@app.post("/api/settings/voice/elevenlabs/voices")
async def api_elevenlabs_voices_proxy(request: Request):
    """Fetch available ElevenLabs voices using the provided API key."""
    body = await request.json()
    api_key = body.get("api_key", "").strip()
    if not api_key:
        return JSONResponse({"voices": [], "error": "No API key provided"})
    url = "https://api.elevenlabs.io/v1/voices"
    headers = {"xi-api-key": api_key}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        voices = [
            {"voice_id": v["voice_id"], "name": v["name"], "category": v.get("category", "")}
            for v in data.get("voices", [])
        ]
        return JSONResponse({"voices": voices})
    except httpx.HTTPStatusError as e:
        detail = str(e)
        try:
            detail = e.response.json().get("detail", {}).get("message", str(e))
        except Exception:
            pass
        return JSONResponse({"voices": [], "error": f"ElevenLabs: {detail}"})
    except Exception as e:
        return JSONResponse({"voices": [], "error": str(e)})


@app.post("/api/settings/voice/openedai/voices")
async def api_openedai_voices_proxy(request: Request):
    """Fetch available voices from an OpenedAI Speech local server."""
    body = await request.json()
    base_url = (body.get("url", "") or "").rstrip("/")
    api_key = body.get("api_key", "")
    if not base_url:
        return JSONResponse({"voices": [], "error": "No server URL provided"})
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    VOICE_ENGINE_MAP = {
        "tts-1": {"engine": "piper", "label": "Piper (CPU · fast)"},
        "tts-1-hd": {"engine": "xtts", "label": "XTTS v2 (GPU · HD)"},
    }
    default_voices = {
        "tts-1": ["alloy", "echo", "echo-alt", "fable", "onyx", "nova", "shimmer"],
        "tts-1-hd": ["alloy", "alloy-alt", "echo", "fable", "onyx", "nova", "shimmer"],
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{base_url}/v1/models", headers=headers)
            resp.raise_for_status()
            models_data = resp.json()
        voices = []
        seen = set()
        for m in models_data.get("data", []):
            mid = m["id"]
            ei = VOICE_ENGINE_MAP.get(mid, {"engine": "unknown", "label": mid})
            for vn in default_voices.get(mid, []):
                key = f"{vn}|{mid}"
                if key not in seen:
                    seen.add(key)
                    voices.append({
                        "voice_id": vn, "name": vn,
                        "model": mid, "engine": ei["engine"],
                        "engine_label": ei["label"],
                    })
        return JSONResponse({"voices": voices})
    except httpx.HTTPStatusError as e:
        detail = str(e)
        try:
            detail = e.response.text[:300]
        except Exception:
            pass
        return JSONResponse({"voices": [], "error": f"OpenedAI: {detail}"})
    except Exception as e:
        return JSONResponse({"voices": [], "error": str(e)})





# ═══════════════════════════════════════════════════════════════════
#  CHAT EMOJI GENERATION
# ═══════════════════════════════════════════════════════════════════

@app.post("/api/chats/{chat_id}/emoji")
async def api_generate_emoji(chat_id: str):
    """Generate a single emoji that represents the chat topic."""
    chat = _load_chat(chat_id)
    if not chat:
        return JSONResponse({"error": "Not found"}, 404)
    title = chat.get("title", "New Chat")
    if title == "New Chat":
        return JSONResponse({"emoji": "💬"})
    conn_data = _load_connections()
    agent_name = chat.get("agent", "")
    conn_id = conn_data.get("agent_connections", {}).get(agent_name)
    conn = None
    for c in conn_data.get("connections", []):
        if c["id"] == conn_id and c.get("enabled"):
            conn = c; break
    if not conn:
        for c in conn_data.get("connections", []):
            if c.get("enabled") and c.get("api_key"):
                conn = c; break
    if not conn:
        return JSONResponse({"emoji": "💬"})
    try:
        url = conn["url"].rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if conn.get("api_key"):
            headers["Authorization"] = f"Bearer {conn['api_key']}"
        model = (conn.get("models") or ["gpt-4o-mini"])[0]
        cheap = [m for m in (conn.get("models") or []) if "mini" in m or "flash" in m]
        if cheap: model = cheap[0]
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, headers=headers, json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "Given a chat title, return a SINGLE emoji that best represents the topic. Return ONLY one emoji character, nothing else."},
                    {"role": "user", "content": title},
                ],
                "max_tokens": 5, "temperature": 0.3,
            })
            resp.raise_for_status()
            emoji = resp.json()["choices"][0]["message"]["content"].strip()
            if len(emoji) > 4: emoji = "💬"
    except Exception:
        emoji = "💬"
    chat["emoji"] = emoji
    _save_chat(chat_id, chat)
    _update_chat_index_entry(chat_id, {"emoji": emoji})
    return JSONResponse({"emoji": emoji})


# ═══════════════════════════════════════════════════════════════════
#  CHAT FOLDERS & MOVE
# ═══════════════════════════════════════════════════════════════════

@app.post("/api/chats/folders")
async def api_create_folder(folder: FolderCreate):
    """Create a new chat folder."""
    idx = _load_chat_index()
    new_folder = {"id": str(uuid.uuid4())[:8], "name": folder.name, "order": len(idx.get("folders", []))}
    idx.setdefault("folders", []).append(new_folder)
    _save_chat_index(idx)
    return JSONResponse(new_folder)


@app.put("/api/chats/folders/{folder_id}")
async def api_update_folder(folder_id: str, request: Request):
    """Rename a folder."""
    body = await request.json()
    idx = _load_chat_index()
    for f in idx.get("folders", []):
        if f["id"] == folder_id:
            if "name" in body: f["name"] = body["name"]
            _save_chat_index(idx)
            return JSONResponse(f)
    return JSONResponse({"error": "Not found"}, 404)


@app.delete("/api/chats/folders/{folder_id}")
async def api_delete_folder(folder_id: str):
    """Delete a folder (chats inside become un-foldered)."""
    idx = _load_chat_index()
    idx["folders"] = [f for f in idx.get("folders", []) if f["id"] != folder_id]
    for c in idx.get("chats", []):
        if c.get("folder_id") == folder_id:
            c["folder_id"] = None
    _save_chat_index(idx)
    for c in idx.get("chats", []):
        ch = _load_chat(c["id"])
        if ch and ch.get("folder_id") == folder_id:
            ch["folder_id"] = None
            _save_chat(c["id"], ch)
    return JSONResponse({"status": "deleted"})


@app.put("/api/chats/{chat_id}/move")
async def api_move_chat(chat_id: str, request: Request):
    """Move a chat to a folder (or null to un-folder)."""
    body = await request.json()
    folder_id = body.get("folder_id")
    chat = _load_chat(chat_id)
    if not chat:
        return JSONResponse({"error": "Not found"}, 404)
    chat["folder_id"] = folder_id
    _save_chat(chat_id, chat)
    _update_chat_index_entry(chat_id, {"folder_id": folder_id})
    return JSONResponse({"status": "moved", "folder_id": folder_id})


# ═══════════════════════════════════════════════════════════════════
#  HEALTH
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def api_health():
    fm = _get_faiss_memory()
    vs = _get_vault_store()
    return {"status": "ok", "agents": _list_agents(), "vault_loaded": fm is not None, "vault_store": vs is not None}
