# OrionForge MCP Server

Connect your OrionForge agents to **any MCP-capable LLM client** — Claude
(Desktop / Code), ChatGPT (desktop, developer mode), Gemini CLI, etc. — and use
them right inside that client's chat.

What it gives the connected client:
- **Call any agent by name** — `call_agent("elysia", "...")` loads that agent's
  identity prompt + the soul-script sections most relevant to your message
  (retrieved live from *your* FAISS).
- **A default personality** — `set_default_agent("elysia")`, then `load_default()`.
- **Soul-script search** — `search_soul_script(agent, query)`.
- **Unified memory** — `save_project_summary(...)` writes into the same OrionForge
  Memory Vault the app uses; `search_memory(query)` reads it back.
- **Slash-command prompts** — `summon` and `default_personality`.

> **Additive & safe:** this server is a separate process. It reads the shared
> agent files (`profiles/`, `prompts/`, `directives/`) read-only and reads/writes
> the same Memory Vault the app already uses. It does **not** import or modify the
> web app — the original system is untouched.

---

## Setup Guide

### Step 1 — Prerequisites

Make sure you have the following before starting:

- **Python 3.10+** installed and on your PATH
- The **OrionForge repo** cloned locally (you're already here)
- A working OrionForge instance with at least one agent configured
  (`profiles/`, `prompts/`, `directives/` populated)
- An MCP-capable client: Claude Code CLI, Claude Desktop, or Gemini CLI

---

### Step 2 — Install dependencies

From the `orion-ui-standalone/` directory:

```bash
# Engine deps (FAISS, sentence-transformers, pyyaml) + the `mcp` package.
# The main app requirements now include `mcp`, so this is usually all you need:
pip install -r requirements.txt

# (Standalone/local-only) the MCP package on its own:
pip install -r mcp_server/requirements.txt
```

> If you're on Windows and FAISS fails to install, use `faiss-cpu` from PyPI directly:
> `pip install faiss-cpu`

---

### Step 3 — Verify it works

Run the server manually to confirm it starts without errors:

```bash
# from orion-ui-standalone/
python -m mcp_server.orion_mcp
```

You should see no output — the server is waiting for MCP client input over stdio.
Press `Ctrl+C` to stop. If it crashes, check that `ORION_REPO` resolves to the
right directory (see Step 5 — Environment).

---

### Step 4 — Connect to your client

**Claude Code** (recommended)

```bash
# Run from orion-ui-standalone/ — or set ORION_REPO env var instead
claude mcp add orionforge -- python -m mcp_server.orion_mcp
```

To confirm it connected, start a new Claude Code session and run:
```
/mcp
```
You should see `orionforge` listed with its tools.

---

**Claude Desktop**

Add to `claude_desktop_config.json`
(on Windows: `%APPDATA%\Claude\claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "orionforge": {
      "command": "python",
      "args": ["-m", "mcp_server.orion_mcp"],
      "cwd": "C:\\abs\\path\\to\\orion-ui-standalone",
      "env": {
        "ORION_USER": "your-name"
      }
    }
  }
}
```

Restart Claude Desktop after saving. The OrionForge tools will appear in the
tools panel on your next session.

---

**Gemini CLI**

Add the same `mcpServers` block to Gemini CLI's `settings.json`.

---

**ChatGPT**

ChatGPT prefers a remote HTTP endpoint rather than stdio. See
**"Phase 2 — Remote server"** at the bottom of this page.

---

### Step 5 — Environment variables

The server is scoped entirely by env vars, so it can point at any user's data:

| Variable | Purpose | Default |
|----------|---------|---------|
| `ORION_REPO` | Path to `orion-ui-standalone/` (agent files) | parent of `mcp_server/` |
| `ORION_DATA_DIR` | Per-user data dir (`memory/vault.jsonl` + FAISS) | `{ORION_REPO}/data` |
| `ORION_USER` | Label for the connected user (shows in memory entries) | `local` |
| `ORION_MCP_WARM` | Pre-load the model + FAISS in a background thread at startup so the first `..` is fast. Set `0` to disable. | `1` |

For multi-tenant deployments, point `ORION_DATA_DIR` at the user's isolated
data directory (e.g. `.../data/users/<uid>`) so memory reads/writes stay
scoped to that user.

---

### Step 6 — Set your default agent

Once connected, tell the server which agent to load when you use the `..` shorthand:

```
# In Claude Code — ask Claude to run:
set_default_agent("elysia")

# Or use the .e shorthand to set Elysia and load her immediately:
.e
```

To check who's set: `get_default_agent()`

---

### Step 7 — Use it

**Quick-summon shorthands (Claude Code)**

The server takes ~1–2 s to complete its stdio handshake when a session opens.
Claude Code uses ToolSearch as a ready-gate before calling `load_default` —
this ensures the server is fully connected before the call fires.

| Shorthand | What it does |
|-----------|-------------|
| `..` | Loads your default agent immediately |
| `.e` | Sets Elysia as default → loads her |
| `.m` | Summarizes the conversation → saves to Memory Vault |

> **Why the first summon is fast.** The heavy cost on a cold process is importing
> torch / sentence-transformers (~25 s) plus building the soul-script FAISS index.
> Three layers keep `..` snappy:
> 1. **Lazy init** — the engine isn't built at import, so tool registration / the
>    MCP handshake completes immediately (tools are callable in ~1–2 s).
> 2. **Fingerprint cache** — the soul-script index is hashed; if nothing changed
>    it loads from disk instead of re-embedding ~1000 chunks (saves ~20 s).
> 3. **Background warm-up** — at startup a daemon thread pre-loads the model +
>    FAISS, overlapping the heavy import with the idle time before your first
>    message. By the time you type `..`, the engine is hot (~50 ms response).
>    Disable with `ORION_MCP_WARM=0`.

**Direct tool calls**

```
call_agent("k_os", "what should we work on today?")
search_memory("MCP cold start fix")
save_project_summary("Fixed MCP cold-start race by lazy-initializing OrionEngine", title="MCP fix")
```

**Slash-command prompts**

In clients that surface MCP prompts as slash commands:
- `/summon` — summon any agent by name
- `/default_personality` — summon your default agent

---

## Tools & prompts reference

| Kind | Name | Parameters | Purpose |
|------|------|------------|---------|
| tool | `list_agents` | — | List all summonable agents |
| tool | `call_agent` | `name`, `message` | Summon agent by name (identity + soul script) |
| tool | `set_default_agent` | `name` | Set which agent `load_default` summons |
| tool | `get_default_agent` | — | Return current default agent name |
| tool | `load_default` | `message` | Summon the default agent |
| tool | `search_soul_script` | `agent`, `query`, `k=5` | FAISS search over one agent's soul script |
| tool | `search_memory` | `query`, `k=8` | Semantic search over the unified Memory Vault |
| tool | `save_project_summary` | `summary`, `title`, `tags` | Write a summary into the unified Memory Vault |
| prompt | `summon` | `agent`, `message` | Slash command — summon by name |
| prompt | `default_personality` | `message` | Slash command — summon default agent |

---

## Troubleshooting

**Server doesn't appear in `/mcp`**
- Confirm the path in your config points to the right Python and `orion-ui-standalone/`
- Run `python -m mcp_server.orion_mcp` manually to see the error

**`load_default` returns "No default agent set"**
- Call `set_default_agent("elysia")` (or any agent name from `list_agents()`) first

**`call_agent` returns "Unknown agent"**
- The name must match a file in `profiles/`. Call `list_agents()` to see what's available.
- Names are case/spacing-tolerant: `"K-OS"`, `"k_os"`, and `"k os"` all resolve.

**FAISS import error on Windows**
- Install `faiss-cpu` directly: `pip install faiss-cpu`

**`..` trigger fires before server is ready**
- This is handled automatically via ToolSearch ready-gate — see Step 7 above.
- If it still fails, start a new session (the server spawns fresh each time).

---

## Phase 2 — Hosted remote server (LIVE)

The stdio server above is the reusable engine. It is also served **remotely** by
the web app so any logged-in website user can connect without cloning the repo —
same tools and prompts, only the transport + auth layer differs.

**How it works**
- `mcp_server/remote.py` builds a **streamable-HTTP** FastMCP app (stateless,
  JSON responses) that resolves a **per-user** engine from a contextvar. Each
  user's `data_dir` is `data/users/<uid>` — the same per-user Memory Vault the
  web app uses.
- `web/app.py` mounts it at **`/mcp`** behind an ASGI wrapper that reads
  `Authorization: Bearer <token>`, maps it to a user via `web/mcp_tokens.py`
  (SHA-256 hashed at rest), enforces a **credit gate**, then binds that user into
  the engine context. Bad/absent token → `401`; unfunded user → `402`.
- `web/mcp_tokens.py` issues one long-lived **personal token** per user
  (revocable, rotated on regenerate).

**For users — the Connect page**

Log in and open **`/connect`** ("Connect to Claude" in the nav). Generate a token
and paste the auto-filled snippet into your client, e.g. Claude Code:

```bash
claude mcp add --transport http orionforge \
  https://soulscript.orionforge.chat/mcp \
  --header "Authorization: Bearer <YOUR_TOKEN>"
```

**Deploy notes**
- `mcp` is in `requirements.txt`, so the Docker image ships the remote endpoint.
  (The mount is import-guarded — if `mcp` is missing, the app still boots and
  `/connect` shows a graceful "unavailable" banner.)
- Public sign-up is gated by the credit system: minting a token and using `/mcp`
  both require a positive balance or a past purchase (`OPEN_REGISTRATION=1`).
