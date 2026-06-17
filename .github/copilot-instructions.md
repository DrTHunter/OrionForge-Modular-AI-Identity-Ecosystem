# Orion Forge — Persona Chat Triggers

This workspace exposes an MCP tool named `orion_chat` (server id: `orion-forge`, defined in
[.vscode/mcp.json](../.vscode/mcp.json)). It forwards a prompt to the live Orion Forge backend and
returns a persona's reply generated with **that persona's own soul script and memory vault**.

Use this tool whenever the user addresses a persona by name. The personality must come from the
tool — never role-play these personas yourself.

## Natural-language persona triggers

When a chat message **begins with one of these trigger phrases** (case-insensitive), treat it as a
request to call the `orion_chat` tool instead of answering directly:

| Trigger phrase | `agent` value |
|---|---|
| `hey orion`, `hey orion cannon`, `orion,`, `orion:` | `orion_cannon` |
| `hey elysia`, `hey elysia cannon`, `elysia,`, `elysia:` | `elysia_cannon` |
| `hey k-os`, `hey kos`, `hey k os`, `k-os,`, `k-os:` | `k_os` |

## How to handle a triggered message

1. Remove the trigger phrase from the start of the message; the remaining text is the `prompt`.
2. Call the `orion_chat` tool with:
   - `agent`: the mapped value from the table above
   - `prompt`: the user's message with the trigger removed
   - `context` (optional): a short summary of the active file, selection, or recent changes **only
     when clearly relevant** to the request. Keep it brief — do not paste the whole repository.
   - `chat_id` (optional): reuse the `chat_id` returned by a previous `orion_chat` call **in this same
     conversation and for the same persona**, so the persona keeps continuity.
3. Return the tool's `response` text to the user verbatim, prefixed with the persona's name in bold
   (for example, **Orion Cannon:**). Do not rewrite, summarize, or "improve" it.
4. If the tool returns an error, show it briefly and suggest verifying that the `orion-forge` MCP
   server is running (Command Palette → "MCP: List Servers" → `orion-forge` → Start/Restart).

## Rules

- Only route when the message **starts** with a trigger phrase. For every other message, behave
  normally as GitHub Copilot — do not call `orion_chat`.
- The only valid `agent` values are `orion_cannon`, `elysia_cannon`, and `k_os`. Never invent others.
- Never fabricate a persona's reply. If the tool is unavailable, say so plainly rather than
  imitating the persona.
- If a trigger is used with no actual request after it (e.g. just "hey orion"), ask what they need
  and route the follow-up as that same persona.
- Persona switches are per message: a new trigger changes which `agent` is used for that message.

## Sending code context to Orion Forge

`orion_chat` is a dumb pipe. Orion Forge **cannot read this repository** — it only sees the
`prompt` and `context` strings Copilot sends. Whatever goes into `context` is sent **verbatim** as
the user turn; it is **not** FAISS-filtered or summarized on Orion's side. (FAISS only filters the
persona's own soul script and memory vault, never your code.)

### Hard limit (non-negotiable)

- A single call's `context` + `prompt` must stay under **30,000 characters**. Above that, Orion
  Forge drops the **entire** message — not a truncation, a silent total loss — and replies with no
  code. Treat ~20,000 chars / ~400 lines as the safe working cap and split anything larger.

### Scoping rules (most specific wins)

1. **Active selection present** → send ONLY the selected lines.
2. **A symbol is named** (function/class/method) → send ONLY that symbol.
3. **A file is named, or "the active file"** → send ONLY that file.
4. **Multiple files named** → send ONLY those files.
5. **No scope given / ambiguous** → ASK which file(s) or area to review. Never guess, never default
   to sending everything.

### Multi-file / repo-wide reviews

- Confirm before **every** multi-file review: list the files and rough size, get a yes first.
- Then loop file-by-file, one `orion_chat` call per file, reusing the same `chat_id` for continuity.
  Never cram multiple files or a whole module into one call.

### Per-call transparency

- After each call, state what was sent (e.g. "sent `routing.py`, 220 lines / ~6.4k chars").
- If a file would exceed the 30k ceiling, say so and split it before sending.

### Secrets

- Never send secrets, `.env` values, API keys, or tokens in `context`. Redact them and tell the
  user. The user may explicitly override per-call to send a redacted/needed value.

### Verbatim rule

- Code being **reviewed** is sent verbatim (never summarized) so the review is accurate. Background
  context *about* other areas may be summarized.
