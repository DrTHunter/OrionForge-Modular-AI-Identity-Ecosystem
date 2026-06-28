# Orion Forge — VS Code Bridge (MCP)

Use your Orion Forge agents **inside VS Code**. The VS Code Bridge is a tiny MCP
server (`scripts/orion_vscode_bridge.py`) that exposes a single tool, `orion_chat`,
to VS Code's Copilot **agent mode** (or any MCP-capable VS Code extension). It
forwards your prompt — plus optional context — to a running Orion Forge instance
and returns the full response: the reply text, which agent / model / router tier
handled it, token usage, cost, tool calls, and any saved memories.

> **How it differs from the `orionforge` MCP server.** The `orionforge` server
> (`mcp_server/`) reads your agent files and FAISS indexes *locally*. The VS Code
> Bridge instead talks to the **live web app over HTTP** (`POST /api/chat/send`),
> so every call runs the full pipeline: 6-layer identity injection, your Memory
> Vault, model routing, tools, and per-user credit billing. It's a standalone
> script, fully isolated from the core runtime — enable, disable, or delete it
> without touching anything else.

```
VS Code (Copilot agent / MCP client)
        │  stdio (JSON-RPC)
        ▼
scripts/orion_vscode_bridge.py   ──HTTP──►   {base_url}/api/chat/send
        ▲                                          │
        └────────────  orion_chat result  ◄────────┘
```

---

## The tool: `orion_chat`

| Input | Required | Description |
|-------|----------|-------------|
| `prompt` | ✅ | Task or question to send to Orion Forge. |
| `agent` | — | Persona: `orion_cannon` (default), `elysia_cannon`, or `k_os`. |
| `context` | — | Code summary / chat summary prepended before the prompt. |
| `model_override` | — | Force a model. Aliases: `deepseek-reasoner`, `gpt-5.2`, `claude-sonnet-latest`, `claude-opus-latest`. |
| `chat_id` | — | Reuse an existing Orion chat thread. |
| `mode` | — | `chat` (default) or `burst` (autonomous multi-step). |
| `burst_ticks` / `max_steps` | — | Burst loop bounds. |

---

## Prerequisites

- **VS Code** with MCP support — GitHub Copilot **agent mode**, or an MCP-capable
  extension (Continue, Cline, etc.).
- **Python 3.10+** and the `requests` package: `pip install requests`.
- **An Orion Forge instance to point at** — the hosted platform
  (`https://orionforge-engine.fly.dev`) or a local app (`http://127.0.0.1:8989`).
- **Credentials** for that instance if it requires auth (see *Per-user setup*).

---

## Step 1 — Install the dependency

```bash
pip install requests
```

The bridge itself is a single file already in the repo:
`orion-ui-standalone/scripts/orion_vscode_bridge.py`.

## Step 2 — Register it in VS Code

Create a **workspace** config at `.vscode/mcp.json` (or run **MCP: Add Server**
from the Command Palette for a user-level install):

```json
{
  "servers": {
    "orion-forge": {
      "type": "stdio",
      "command": "C:/path/to/python.exe",
      "args": ["C:/path/to/orion-ui-standalone/scripts/orion_vscode_bridge.py"],
      "env": {
        "ORION_VSCODE_BRIDGE_BASE_URL": "https://orionforge-engine.fly.dev",
        "ORION_VSCODE_BRIDGE_AUTH_MODE": "bearer",
        "ORION_VSCODE_BRIDGE_BEARER_TOKEN": "${input:orion_token}",
        "ORION_VSCODE_BRIDGE_DEFAULT_AGENT": "orion_cannon",
        "ORION_VSCODE_BRIDGE_DEFAULT_MODEL": "deepseek-reasoner",
        "ORION_VSCODE_BRIDGE_FALLBACK_MODEL": "gpt-5.2"
      }
    }
  },
  "inputs": [
    {
      "id": "orion_token",
      "type": "promptString",
      "description": "Your Orion Forge access token",
      "password": true
    }
  ]
}
```

> Using `${input:orion_token}` with `"password": true` means VS Code **prompts**
> for your token and never writes it to disk in plaintext.

## Step 3 — Use it

Open the **Copilot Chat** view, switch to **Agent** mode, and enable the
`orion-forge` tools. Now just chat — VS Code routes your prompts through
`orion_chat` into Orion Forge. To force a persona or model, the agent passes
`agent` / `model_override`; hand Orion a code summary via `context`.

---

## Per-user setup (important)

The bridge always acts **as the identity it authenticates with**, and Orion Forge
is **multi-tenant** — every user has an isolated data tree (`data/users/{uid}/`:
chats, memory vault, credits). So **"per user" means each person authenticates as
themselves**: their VS Code chats then use *their own* memory, *their own* agents,
and bill *their own* credits.

Choose the auth mode with `ORION_VSCODE_BRIDGE_AUTH_MODE`:

| Mode | Best for | What you set | Notes |
|------|----------|--------------|-------|
| **`bearer`** | **Each user, hosted platform** | their own access token in `ORION_VSCODE_BRIDGE_BEARER_TOKEN` | Acts as that user (their data + credits). **Best per-user option today.** Tokens are short-lived — refresh on expiry. |
| `cookie` | Each user, alt to bearer | `ORION_VSCODE_BRIDGE_COOKIE_VALUE` = their `sb_access_token` | Same per-user effect via the session cookie. |
| `disabled` | Local dev, single user | nothing | Talks to a local app with auth off (`base_url=http://127.0.0.1:8989`). |
| `bridge_key` | Self-host / owner | static key in `ORION_VSCODE_BRIDGE_KEY` matching the server's `ORION_BRIDGE_API_KEY` | Maps to **one** configured account. Single-owner, long-lived. |

### Per-user, hosted (recommended): `bearer`

1. Log in to your Orion Forge account in the browser.
2. Copy your access token — it's the `sb_access_token` session cookie
   (DevTools → Application → Cookies), set when you log in.
3. In `.vscode/mcp.json`, set `ORION_VSCODE_BRIDGE_AUTH_MODE=bearer` and supply the
   token via an `inputs` prompt (`password: true`) so it's never committed.
4. The bridge now runs **as you** — your vault, your chats, your credits.

> Tokens expire. When calls start returning **401**, grab a fresh token. A
> platform-issued *long-lived per-user key* is a planned enhancement — see below.

### Self-host / owner: `bridge_key`

A static key authenticates the bridge as one account without expiring tokens.
Fully opt-in: when no key is set server-side, the mechanism is inert and normal
login is unchanged.

1. Generate a key (in your own terminal):
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(48))"
   ```
2. Set it on the server: `ORION_BRIDGE_API_KEY=<key>` (Fly:
   `fly secrets set ORION_BRIDGE_API_KEY="<key>" -a orionforge-engine && fly deploy`;
   local: env var or `config/auth.json` → `bridge_api_key`). To share your browser
   account's data, also set `ORION_BRIDGE_USER_ID=<your Supabase uid>` (or
   `auth.json` → `bridge_user_id`); otherwise calls land in an isolated
   `__bridge__` space.
3. Point the bridge at the same key: `ORION_VSCODE_BRIDGE_AUTH_MODE=bridge_key`,
   `ORION_VSCODE_BRIDGE_KEY=<same key>`.

**Security:** treat the key like a password (full account access); it's compared
in constant time and only works over the server's HTTPS endpoint. Remove it with
`fly secrets unset ORION_BRIDGE_API_KEY` + redeploy.

### Per-user keys at scale (roadmap)

Today `bridge_key` maps **one** key → **one** user id. For a hosted multi-user
rollout where every user gets their own long-lived key, the platform would issue
**per-user keys** (a key→uid map) plus a *"Connect VS Code"* page that hands each
user their key and a ready-made `.vscode/mcp.json`. Until then, per-user = `bearer`
(or `cookie`) tokens, above.

---

## Configuration reference

The bridge reads `config/vscode_bridge.json`, overridden by `ORION_VSCODE_BRIDGE_*`
environment variables (env wins).

| Env var | Purpose | Default |
|---------|---------|---------|
| `ORION_VSCODE_BRIDGE_BASE_URL` | Orion instance URL | `http://127.0.0.1:8989` |
| `ORION_VSCODE_BRIDGE_AUTH_MODE` | `disabled` / `bearer` / `cookie` / `bridge_key` | `disabled` |
| `ORION_VSCODE_BRIDGE_BEARER_TOKEN` | Supabase JWT (bearer mode) | — |
| `ORION_VSCODE_BRIDGE_COOKIE_VALUE` | Session cookie value (cookie mode) | — |
| `ORION_VSCODE_BRIDGE_KEY` | Static bridge key (bridge_key mode) | — |
| `ORION_VSCODE_BRIDGE_DEFAULT_AGENT` | Persona | `orion_cannon` |
| `ORION_VSCODE_BRIDGE_DEFAULT_MODEL` | First model to try | — |
| `ORION_VSCODE_BRIDGE_FALLBACK_MODEL` | Model on retryable failure | — |
| `ORION_VSCODE_BRIDGE_DEFAULT_MODE` | `chat` / `burst` | `chat` |
| `ORION_VSCODE_BRIDGE_TIMEOUT_SECONDS` | HTTP timeout | `120` |
| `ORION_VSCODE_BRIDGE_CONFIG` | Path to the JSON config | `config/vscode_bridge.json` |

Example `config/vscode_bridge.json`:

```json
{
  "base_url": "https://orionforge-engine.fly.dev",
  "default_agent": "orion_cannon",
  "allowed_agents": ["orion_cannon", "elysia_cannon", "k_os"],
  "default_model_override": "deepseek-reasoner",
  "fallback_model_override": "gpt-5.2",
  "auth": { "mode": "bearer", "bearer_token": "" }
}
```

## Model selection

- `default_model_override` is tried first; `fallback_model_override` is used if the
  first fails with a retryable API error (400/404/408/409/422/429/5xx).
- Per-request: pass `model_override` in the `orion_chat` call.
- Aliases expand to an ordered list of concrete model ids, tried until one works:
  `deepseek-reasoner`, `gpt-5.2`, `claude-sonnet-latest`, `claude-opus-latest`.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `401 Invalid bridge key` / Unauthorized | Token expired or key mismatch — refresh the token or realign the key. |
| Connection refused | Wrong `base_url`, or the local app isn't running (`uvicorn web.app:app --port 8989`). |
| Tool doesn't appear in VS Code | Reload window; check the `command`/`args` paths in `.vscode/mcp.json`; open the MCP server **Output** panel for errors. |
| `ModuleNotFoundError: requests` | `pip install requests`. |
| `agent '...' is not allowed` | Use `orion_cannon`, `elysia_cannon`, or `k_os`. |

## Security

- **Never commit tokens or keys.** Use VS Code `inputs` (`password: true`) or
  environment variables, and keep secrets out of `.vscode/mcp.json` in git.
- The bridge has full access to whatever account it authenticates as — guard the
  token/key accordingly.
