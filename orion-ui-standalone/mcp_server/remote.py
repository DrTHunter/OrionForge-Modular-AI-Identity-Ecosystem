"""OrionForge MCP — remote, multi-tenant transport.

This is the Phase-2 counterpart to ``orion_mcp.py`` (the local stdio server).
It exposes the *same* tools and prompts over **streamable HTTP**, but resolves a
**per-user** engine for every request instead of a single process-global one, so
one hosted endpoint can serve every website user with their own isolated data.

Design:
  - The tool bodies are shared with the stdio server via
    ``orion_mcp.register_tools(server, get_engine)``. Here ``get_engine`` returns
    a per-user :class:`OrionEngine` whose ``data_dir`` is the same
    ``data/users/<uid>`` directory the web app uses — so the MCP and the app
    read/write the same Memory Vault.
  - The current user for a request is carried in a :class:`ContextVar`
    (``_USER_CTX``). ``stateless_http=True`` keeps each request self-contained so
    the contextvar set by the host's auth wrapper is visible when the tool runs,
    and there is no cross-worker session affinity to manage on Fly.

This module deliberately does **no** authentication or billing — that lives in
``web/app.py`` (which already imports Supabase auth + the credit system) and is
applied as an ASGI wrapper around the app returned by :func:`build_mcp_app`.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from contextvars import ContextVar
from pathlib import Path
from typing import Tuple

from mcp_server.engine import OrionEngine

# ── Per-request user identity ───────────────────────────────────────
_USER_CTX: ContextVar[str] = ContextVar("orion_mcp_user", default="")

# ── Bounded per-user engine cache ───────────────────────────────────
# Each engine lazily builds that user's FAISS indexes on first use, so we keep
# them warm across requests. Bounded so a flood of users can't grow unbounded.
_MAX_ENGINES = 256
_engines: "OrderedDict[str, OrionEngine]" = OrderedDict()
_engines_lock = threading.Lock()


def _data_users_root() -> Path:
    repo = Path(os.environ.get("ORION_REPO") or Path(__file__).resolve().parent.parent)
    return (repo / "data" / "users").resolve()


def set_current_user(user_id: str) -> None:
    """Set the user whose engine subsequent tool calls in this context resolve to."""
    _USER_CTX.set(user_id or "")


def current_user() -> str:
    return _USER_CTX.get()


def engine_for_user(user_id: str) -> OrionEngine:
    """Return (building + caching on first use) the engine for ``user_id``.

    The engine's data dir is ``{ORION_REPO}/data/users/<uid>`` — identical to the
    web app's per-user layout (see ``web/user_data.py``), so memory is shared.
    """
    uid = (user_id or "").strip()
    if not uid:
        raise ValueError("engine_for_user: empty user_id")
    with _engines_lock:
        eng = _engines.get(uid)
        if eng is not None:
            _engines.move_to_end(uid)  # mark most-recently-used
            return eng
        if len(_engines) >= _MAX_ENGINES:
            _engines.popitem(last=False)  # evict least-recently-used
        eng = OrionEngine(data_dir=str(_data_users_root() / uid), user=uid)
        _engines[uid] = eng
        return eng


def _engine_for_current_user() -> OrionEngine:
    uid = _USER_CTX.get()
    if not uid:
        raise RuntimeError(
            "No authenticated user in context — the MCP auth wrapper must call "
            "set_current_user() before dispatching the request."
        )
    return engine_for_user(uid)


def build_mcp_app() -> Tuple[object, object]:
    """Build the streamable-HTTP MCP handler + its FastMCP server.

    Returns ``(asgi_app, mcp_server)``. The caller (``web/app.py``) wraps
    ``asgi_app`` with bearer-token auth, mounts it at ``/mcp``, and runs
    ``mcp_server.session_manager.run()`` inside its lifespan.

    We delegate straight to the session manager's ASGI handler rather than
    returning FastMCP's full Starlette app — that way every request under the
    ``/mcp`` mount (including the exact ``/mcp`` path with no trailing slash) is
    handled, with no inner route matching or 307 trailing-slash redirects that
    some MCP HTTP clients won't follow.
    """
    from mcp.server.fastmcp import FastMCP
    from mcp_server.orion_mcp import register_tools

    mcp = FastMCP(
        "orionforge",
        stateless_http=True,
        # JSON responses (not SSE): the host app wraps this in Starlette
        # BaseHTTPMiddleware (auth/CSRF), which buffers and would break an SSE
        # stream. Tool calls are request/response, so plain JSON is the right fit.
        json_response=True,
    )
    register_tools(mcp, _engine_for_current_user)
    # Create the session manager (lazily built by streamable_http_app()).
    mcp.streamable_http_app()
    session_manager = mcp.session_manager

    async def asgi_app(scope, receive, send):
        await session_manager.handle_request(scope, receive, send)

    return asgi_app, mcp
