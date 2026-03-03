# src/tools/

Tool implementations for the OrionForge agent runtime. Each tool is a class with two methods:

- `definition()` — returns JSON Schema for the LLM to understand the tool
- `execute(arguments)` — runs the tool and returns a string result

## Available Tools

| Tool | File | Description |
|------|------|-------------|
| `echo` | `echo.py` | Simple echo for testing. Returns the input message. |
| `continuation_update` | `continuation_update.py` | Per-agent status document. `append` adds text, `replace_section` updates a `## Section` by heading. Backed by `data/<agent>/continuation.md`. |
| `memory` | `memory_tool.py` | Read/write durable memories. 13 actions: `add`, `bulk_add`, `search`, `recall`, `update`, `delete`, `bulk_delete`, `list_scopes`, `list_categories`, `stats`, `suggest_categories`, `set_category_mode`, `set_suggested_categories`. Backed by `data/memory/vault.jsonl` with FAISS semantic search. |
| `directives` | `directives_tool.py` | Read-only access to user-authored directives. 5 actions: `search`, `list`, `get`, `manifest`, `changes`. Reads from `directives/*.md` and `directives/manifest.json`. |
| `web_search` | `web_search.py` | Web search via SearXNG meta-search engine + page scraping. Returns search results with snippets and optionally fetches full page content. |
| `email` | `email_tool.py` | Email sending via SMTP. Actions: `send`, `add_account`, `remove_account`, `list_accounts`. Supports multiple accounts with password masking. |
| `inbox` | `inbox.py` | Email inbox tool. 4 actions: `check`, `read`, `search`, `summary`. Reads from `data/shared/inbox.jsonl`. |
| `cost_tracker` | `cost_tracker.py` | Cost tracking tool. Actions: `cost_summary`, `cost_log`, `session` + pricing CRUD actions. Reads from the metering system. |

## Registry

| File | Purpose |
|------|---------|
| `registry.py` | Tool registry — discovers and loads tool classes by name. Handles resolution, dispatch, listing, and error paths. Used by the chat loop to instantiate tools from profile `allowed_tools`. |

## Adding a New Tool

1. Create `src/tools/<name>.py` with a class exposing `definition()` and `execute(arguments)`
2. Add `<name>` to `allowed_tools` in the relevant profile YAMLs
3. The registry auto-discovers the tool class on next chat

## Dynamic Schema

The `memory` tool definition dynamically includes current scopes.  Categories and category mode (open/strict) are pulled from the live memory profile, so the tool schema always reflects the current configuration.
