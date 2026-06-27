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

## Install

```bash
# from orion-ui-standalone/
pip install -r requirements.fly.txt          # the app's engine deps (faiss, sentence-transformers, pyyaml)
pip install -r mcp_server/requirements.txt   # adds the `mcp` package
```

## Per-user configuration (environment)

The server is scoped entirely by env vars, so the website can hand each user a
ready-made config pointing at *their* data:

| Var | Meaning | Default |
|-----|---------|---------|
| `ORION_REPO` | path to `orion-ui-standalone` (agent files) | parent of `mcp_server/` |
| `ORION_DATA_DIR` | per-user data dir holding `memory/vault.jsonl` + faiss | `{ORION_REPO}/data` |
| `ORION_USER` | label for the connected user | `local` |

For multi-tenant: point `ORION_DATA_DIR` at the user's data directory
(e.g. `.../data/users/<uid>`) so memory reads/writes stay scoped to that user.

## Connect it to your client (stdio)

**Claude Code**
```bash
claude mcp add orionforge -- python -m mcp_server.orion_mcp
# (run from orion-ui-standalone/, or pass --cwd / set ORION_REPO)
```

**Claude Desktop** — add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "orionforge": {
      "command": "python",
      "args": ["-m", "mcp_server.orion_mcp"],
      "cwd": "/abs/path/to/orion-ui-standalone",
      "env": { "ORION_USER": "trent" }
    }
  }
}
```

**Gemini CLI** — add the same block under `mcpServers` in its `settings.json`.

**ChatGPT** — enable developer mode / custom connectors and point it at the server
(ChatGPT prefers a remote URL; see "Per-user on the website" below).

After connecting, start a fresh session: the tools appear, and `summon` /
`default_personality` show up as slash commands.

## Tools & prompts

| Kind | Name | Purpose |
|------|------|---------|
| tool | `list_agents` | list summonable agents |
| tool | `call_agent(name, message)` | summon any agent by name (identity + soul script) |
| tool | `set_default_agent(name)` / `get_default_agent` | default personality |
| tool | `load_default(message)` | summon the default personality |
| tool | `search_soul_script(agent, query, k)` | FAISS search over one soul script |
| tool | `search_memory(query, k)` | semantic search over the unified vault |
| tool | `save_project_summary(summary, title, tags)` | write a summary into the unified vault |
| prompt | `summon(agent, message)` | slash command to summon by name |
| prompt | `default_personality(message)` | slash command for the default agent |

## Per-user on the website (next phase)

This stdio server is the reusable engine. To offer it as a one-click per-user
connector from the website:
1. Run this as a **remote** MCP server (streamable-HTTP transport) alongside the app.
2. Add OAuth/token auth that maps the bearer token → a Supabase user id, and set
   `ORION_DATA_DIR` to that user's data dir per request/session.
3. Give each user a "Connect" page with their personal MCP URL + token to paste
   into Claude/ChatGPT/Gemini.

The tools/prompts above stay identical — only the transport + auth layer is added.
