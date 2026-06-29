"""Hosted MCP connector (Phase 2) tests.

Run from project root:
    python -m tests.test_mcp_remote

Covers:
  - Personal token store (web/mcp_tokens.py): mint->verify roundtrip, rotation
    (one active token per user), revoke, status shape, and hash-at-rest (the raw
    token is never written to disk).
  - Per-user engine resolution (mcp_server/remote.py): two users get distinct
    data dirs under data/users/<uid>; same user is cached; vault path matches the
    web app's per-user layout.
  - The /mcp bearer-auth + credit-gate ASGI wrapper (web/app.py): missing/invalid
    token -> 401, no credits -> 402, valid+funded -> passes through with the right
    user bound into the engine context.
"""

import asyncio
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PASS = 0
FAIL = 0


def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label}  {detail}")


# ═══════════════════════════════════════════════════════════════════
#  1. Personal token store
# ═══════════════════════════════════════════════════════════════════

def test_token_store():
    print("\n=== MCP Remote — Token Store ===")
    import web.mcp_tokens as mt

    tmp = Path(tempfile.mkdtemp())
    orig = mt._TOKENS_FILE
    mt._TOKENS_FILE = tmp / "mcp_tokens.json"

    uid_a = "user-token-alice"
    uid_b = "user-token-bob"

    try:
        # Mint + verify roundtrip
        raw_a = mt.mint_token(uid_a)
        check("mint returns prefixed token", raw_a.startswith("oforge_"))
        check("verify resolves to minter", mt.verify_token(raw_a) == uid_a)

        # Second user is independent
        raw_b = mt.mint_token(uid_b)
        check("verify B resolves to B", mt.verify_token(raw_b) == uid_b)
        check("A and B tokens differ", raw_a != raw_b)

        # Rotation: minting again for A replaces the old token
        raw_a2 = mt.mint_token(uid_a)
        check("rotated token is new", raw_a2 != raw_a)
        check("old token no longer valid", mt.verify_token(raw_a) is None)
        check("new token valid", mt.verify_token(raw_a2) == uid_a)
        check("rotation didn't touch B", mt.verify_token(raw_b) == uid_b)

        # Bad / empty tokens
        check("garbage token rejected", mt.verify_token("oforge_nope") is None)
        check("empty token rejected", mt.verify_token("") is None)
        check("None token rejected", mt.verify_token(None) is None)

        # Hash-at-rest: raw token must NOT appear anywhere in the registry file
        raw_disk = mt._TOKENS_FILE.read_text(encoding="utf-8")
        check("raw token A not on disk", raw_a2 not in raw_disk)
        check("raw token B not on disk", raw_b not in raw_disk)
        registry = json.loads(raw_disk)
        check("registry keyed by hash (64 hex)", all(len(k) == 64 for k in registry))

        # Status is non-secret
        st = mt.get_token_status(uid_a)
        check("status exists", st["exists"] is True)
        check("status has created ts", bool(st.get("created")))
        check("status prefix is short", 0 < len(st.get("prefix", "")) <= len("oforge_") + 6)
        check("status never leaks raw", st.get("prefix", "") in raw_a2)

        # Revoke
        check("revoke removes token", mt.revoke_token(uid_a) is True)
        check("revoked token invalid", mt.verify_token(raw_a2) is None)
        check("status after revoke is empty", mt.get_token_status(uid_a) == {"exists": False})
        check("revoke again is no-op", mt.revoke_token(uid_a) is False)
        check("revoke didn't touch B", mt.verify_token(raw_b) == uid_b)

    finally:
        mt._TOKENS_FILE = orig
        shutil.rmtree(str(tmp), ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════
#  2. Per-user engine resolution
# ═══════════════════════════════════════════════════════════════════

def test_engine_isolation():
    print("\n=== MCP Remote — Per-User Engine Resolution ===")
    import mcp_server.remote as remote

    tmp = Path(tempfile.mkdtemp())
    # Point the per-user data root at a temp repo and reset the cache.
    prev_repo = os.environ.get("ORION_REPO")
    os.environ["ORION_REPO"] = str(tmp)
    with remote._engines_lock:
        remote._engines.clear()

    try:
        uid_a = "user-eng-alice"
        uid_b = "user-eng-bob"

        eng_a = remote.engine_for_user(uid_a)
        eng_b = remote.engine_for_user(uid_b)

        check("different users -> different engines", eng_a is not eng_b)
        check("same user -> cached engine", remote.engine_for_user(uid_a) is eng_a)

        # Data dir matches the web app's per-user layout: data/users/<uid>
        expect_a = (tmp / "data" / "users" / uid_a).resolve()
        check("engine A data_dir scoped to user", eng_a.data_dir == expect_a)
        check("engine A vault under user dir",
              eng_a.vault_path == expect_a / "memory" / "vault.jsonl")
        check("engine B data_dir differs from A", eng_b.data_dir != eng_a.data_dir)
        check("engine user label set", eng_a.user == uid_a)

        # Empty user id is rejected
        try:
            remote.engine_for_user("")
            check("empty uid rejected", False, "should have raised")
        except ValueError:
            check("empty uid rejected", True)

        # contextvar resolver
        remote.set_current_user(uid_b)
        check("current_user reflects contextvar", remote.current_user() == uid_b)
        check("_engine_for_current_user uses ctx", remote._engine_for_current_user() is eng_b)

    finally:
        if prev_repo is None:
            os.environ.pop("ORION_REPO", None)
        else:
            os.environ["ORION_REPO"] = prev_repo
        with remote._engines_lock:
            remote._engines.clear()
        shutil.rmtree(str(tmp), ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════
#  3. /mcp bearer-auth + credit-gate ASGI wrapper
# ═══════════════════════════════════════════════════════════════════

def _drive(asgi, headers):
    """Drive an ASGI http request through `asgi`, returning (status, body, calls)."""
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
    }
    sent = {"status": None, "body": b""}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg):
        if msg["type"] == "http.response.start":
            sent["status"] = msg["status"]
        elif msg["type"] == "http.response.body":
            sent["body"] += msg.get("body", b"")

    asyncio.run(asgi(scope, receive, send))
    return sent["status"], sent["body"]


def test_auth_wrapper():
    print("\n=== MCP Remote — Auth + Credit Gate ===")
    import web.app as app
    import mcp_server.remote as remote

    # Record whether the inner MCP app was reached and with which user bound.
    reached = {"called": False, "user": None}

    async def fake_inner(scope, receive, send):
        reached["called"] = True
        reached["user"] = remote.current_user()
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": b'{"ok":true}'})

    wrapper = app._MCPAuthASGI(fake_inner)

    # Save originals we monkeypatch
    orig_verify = app.mcp_tokens.verify_token
    orig_credits = app.get_user_credits
    orig_purchased = app.user_has_purchased_credits
    orig_ensure = app.ensure_user_dirs
    orig_seed_v = app.seed_user_vault
    orig_seed_c = app.seed_user_chats

    # Stub out side-effecting dir creation
    app.ensure_user_dirs = lambda uid: None
    app.seed_user_vault = lambda uid: None
    app.seed_user_chats = lambda uid: None

    FUNDED = "user-funded"
    BROKE = "user-broke"

    app.mcp_tokens.verify_token = lambda tok: {
        "good-funded": FUNDED, "good-broke": BROKE
    }.get(tok)
    app.get_user_credits = lambda uid: 50 if uid == FUNDED else 0
    app.user_has_purchased_credits = lambda uid: False

    try:
        # No Authorization header -> 401, inner never reached
        reached["called"] = False
        status, body = _drive(wrapper, {})
        check("missing token -> 401", status == 401)
        check("401 inner not reached", reached["called"] is False)

        # Bad token -> 401
        reached["called"] = False
        status, _ = _drive(wrapper, {"authorization": "Bearer wrong"})
        check("invalid token -> 401", status == 401)
        check("401 sets WWW-Authenticate path (inner skipped)", reached["called"] is False)

        # Valid token but no credits -> 402
        reached["called"] = False
        status, body = _drive(wrapper, {"authorization": "Bearer good-broke"})
        check("funded check: broke user -> 402", status == 402)
        check("402 inner not reached", reached["called"] is False)

        # Valid + funded -> passes through, user bound into engine context
        reached["called"] = False
        status, body = _drive(wrapper, {"authorization": "Bearer good-funded"})
        check("funded user -> 200 pass-through", status == 200)
        check("inner reached", reached["called"] is True)
        check("correct user bound in ctx", reached["user"] == FUNDED)

        # Case-insensitive scheme
        reached["called"] = False
        status, _ = _drive(wrapper, {"authorization": "bearer good-funded"})
        check("lowercase 'bearer' accepted", status == 200 and reached["called"])

    finally:
        app.mcp_tokens.verify_token = orig_verify
        app.get_user_credits = orig_credits
        app.user_has_purchased_credits = orig_purchased
        app.ensure_user_dirs = orig_ensure
        app.seed_user_vault = orig_seed_v
        app.seed_user_chats = orig_seed_c


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Keep the heavy engine warm-up off during tests.
    os.environ.setdefault("ORION_MCP_WARM", "0")

    test_token_store()
    test_engine_isolation()
    test_auth_wrapper()

    print(f"\n{'=' * 60}")
    print(f"  MCP REMOTE (PHASE 2) RESULTS")
    print(f"{'=' * 60}")
    print(f"  Passed: {PASS}")
    print(f"  Failed: {FAIL}")
    print(f"  Total:  {PASS + FAIL}")
    if FAIL:
        print(f"\n  ✗ {FAIL} CHECK(S) FAILED")
        sys.exit(1)
    else:
        print(f"\n  ✓ ALL {PASS} CHECKS PASSED")
