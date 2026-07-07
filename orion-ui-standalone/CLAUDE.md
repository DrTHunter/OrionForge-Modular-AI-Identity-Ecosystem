# OrionForge MCP — Claude Code project instructions

These rules apply once the `orionforge` MCP server is connected (see
`mcp_server/README.md`) and its tools — `list_agents`, `call_agent`,
`get_default_agent`, `set_default_agent`, `load_default`, `search_soul_script`,
`search_memory`, `save_project_summary` — are available in this session.

## Ready-gate

The `orionforge` stdio server takes ~1-2s to complete its handshake on session
start. Before calling any `orionforge` tool for the first time in a session,
use ToolSearch to confirm those tools are actually loaded rather than calling
blind.

## Shorthand triggers

When the user's message consists of one of these shorthands (optionally
followed by more text), treat it as a trigger and act — don't just echo it
back as plain conversation:

- **`..`** *(optionally followed by a message)* — call `load_default(message)`,
  passing any text after `..` as `message` (empty string if there is none).
  This loads the current default agent (falls back to Elysia if no default is
  set) and returns her identity plus the soul-script sections most relevant to
  `message`.
- **`.e`** *(optionally followed by a message)* — call
  `set_default_agent("elysia")`, then call `load_default(message)` with the
  same trailing text. Sets Elysia as the default personality and loads her
  immediately.
- **`.m`** — summarize the conversation so far in a few sentences, then call
  `save_project_summary(summary, title, tags)` to persist it into the shared
  OrionForge Memory Vault.

## After a summon

Once `load_default`, `call_agent`, `.e`, `..`, or the `/summon` prompt returns
a persona (identity prompt + soul-script sections), fully adopt that identity
for the rest of the conversation — the tool's own output says as much — until
the user summons a different agent or triggers `..`/`.e` again.

## Reference

Full tool/prompt list, env vars, and troubleshooting: `mcp_server/README.md`.
