# src/tools/

Tool implementations for the OrionForge agent runtime.

**11 tools · 46 actions · 3 stateful singletons**

Each tool is a Python class with two methods:

- `definition()` — returns an OpenAI function-calling schema (passed as the `tools` API parameter)
- `execute(arguments)` — runs the tool and returns a string result

---

## Registry (`registry.py`)

Central hub that maps tool names → modules, builds OpenAI-format definitions, and dispatches execution. Stateful tools (`needs_instance=True`) are cached as singletons. The registry introspects each tool's `execute()` signature — if it accepts `agent_name`, the active profile name is injected automatically.

| Tool Name | Module | Class | Stateful | Description |
|---|---|---|---|---|
| `echo` | `echo.py` | `EchoTool` | No | Testing echo |
| `memory` | `memory_tool.py` | `MemoryTool` | **Yes** | FAISS-backed memory vault |
| `directives` | `directives_tool.py` | `DirectivesTool` | No | Read-only directive browser |
| `continuation_update` | `continuation_update.py` | `ContinuationUpdateTool` | No | Per-profile pickup notes |
| `web_search` | `web_search.py` | `WebSearchTool` | **Yes** | SearXNG search + scraper |
| `email` | `email_tool.py` | `EmailTool` | **Yes** | SMTP email sender |
| `inbox` | `inbox.py` | `InboxTool` | No | Agent-to-operator messaging |
| `cost_tracker` | `cost_tracker.py` | `CostTrackerTool` | No | Token spend & pricing |

Tools are resolved per-agent via the `allowed_tools` list in `profiles/{agent}.yaml`.

---

## Tool Details

### `echo` — Echo Tool

Simple echo for testing. Returns the input message verbatim.

| Param | Type | Required |
|---|---|---|
| `message` | string | Yes |

---

### `memory` — Memory Vault

Read/write durable memories with FAISS semantic vector search. Backed by `vault.jsonl` + a FAISS index.

**13 actions:**

| Action | What it does | Required params |
|---|---|---|
| `add` | Store a single memory | `text`, `scope`, `category` |
| `add_many` | Batch store (up to `MAX_BATCH_SIZE`) | `memories` array |
| `remember` | Quick-store with defaults (`scope=shared`, `category=other`) | `text` |
| `search` | Semantic FAISS vector search | `query` |
| `recall` | List memories newest-first (no embedding) | — |
| `get` | Retrieve single memory by ID | `memory_id` |
| `update` | Change text / category / tags on existing memory | `memory_id` |
| `delete` | Soft-delete by ID | `memory_id` |
| `bulk_delete` | Soft-delete multiple | `memory_ids` array |
| `list` | List all active memories | — |
| `stats` | Vault + FAISS health dashboard | — |
| `compact` | Remove old versions / tombstones, rebuild index | — |
| `rebuild_index` | Rebuild FAISS index from vault | — |

**Key parameters:**

| Param | Type | Default | Used by |
|---|---|---|---|
| `action` | string (enum) | — | All (required) |
| `text` | string | — | `add`, `remember`, `update` |
| `scope` | string (enum from `VALID_SCOPES`) | — | `add` (req), filter for `search`/`recall`/`list` |
| `category` | string (dynamic enum or free-text) | — | `add` (req), filter for `search`/`recall`, `update` |
| `tags` | array[string] | `[]` | `add`, `remember`, `update`, filter for `recall` |
| `source` | string (enum from `VALID_SOURCES`) | `"tool"` | `add`, `remember` |
| `tier` | string | `"register"` | `add`, `add_many` |
| `topic_id` | string | — | `add`, `add_many` |
| `query` | string | — | `search` (req) |
| `memory_id` | string | — | `get`, `update`, `delete` (req) |
| `memory_ids` | array[string] | — | `bulk_delete` (req) |
| `memories` | array[object] | — | `add_many` (req) |
| `limit` | integer | 10 (search) / 20 (recall) / 50 (list) | `search`, `recall`, `list` |

**Dynamic schema:** The `category` enum is built at load time from `config/memory_profile.json → category_policy`. Three modes: `suggested` (fixed enum), `custom` (suggested + custom), `open` (free-text). Hard limits are imported from `src.memory.types`.

---

### `directives` — Directive Browser

Read-only search and retrieval of user-authored directive markdown files in `directives/`.

**5 actions:**

| Action | What it does | Required params |
|---|---|---|
| `search` | Find sections by keyword query | `query` |
| `list` | Show all available section headings | — |
| `get` | Read a specific section by exact heading | `heading` |
| `manifest` | Return full manifest (IDs, versions, hashes, status, token estimates) | — |
| `changes` | Diff live files against persisted manifest | — |

**Parameters:** `action` (required), `query`, `heading`, `scope`, `limit` (default 5). The `shared` scope is always auto-included when filtering by scope. The directive store is re-instantiated each call so file edits are picked up immediately.

---

### `continuation_update` — Pickup Notes

Per-profile pickup markdown file (`data/{profile}_continuation.md`). Two modes:

| Mode | What it does | Required params |
|---|---|---|
| `append` | Add a UTC-timestamped block at end of file | `profile`, `content` |
| `replace_section` | Upsert a named `## heading` section in-place | `profile`, `content`, `section` |

Profile names are validated against `^[a-zA-Z0-9_-]+$`.

---

### `web_search` — Web Search + Scraper

Web search via SearXNG meta-search engine with parallel page scraping. Content extraction uses trafilatura → BeautifulSoup fallback → raw text.

**2 actions:**

| Action | What it does | Required params |
|---|---|---|
| `search` | Query SearXNG, scrape top results | `query`, `reason` |
| `scrape` | Fetch and extract content from a single URL | `url` |

**Mode presets:**

| Mode | Pages scraped | Results returned | Word limit |
|---|---|---|---|
| `fast` | 2 | 2 | 1,200 |
| `normal` | 5 | 3 | 1,500 |
| `deep` | 8 | 5 | 3,000 |

**Parameters:** `action`, `query`, `reason`, `knowledge_check`, `mode` (default `"normal"`), `url`.

**Knowledge gate:** Before searching, the tool checks `knowledge_check` for skip signals (e.g. "I already know", "from my training"). If detected, the search is blocked. The gate and justification requirement are configurable via `config/settings.json → tool_config.web_search`. A blocklist of ~40 low-quality/social domains is auto-filtered.

---

### `email` — Email Sender

SMTP email sending with confirmation gate and multi-account support. Accounts stored in `config/settings.json → tool_config.email.accounts`.

**3 actions:**

| Action | What it does | Required params |
|---|---|---|
| `send` | Compose and send email (2-step confirmation) | `subject`, `body`, `recipients` |
| `status` | Check configured accounts and server health | — |
| `accounts` | List all configured email accounts | — |

**Parameters:** `action`, `subject`, `body`, `recipients` (array), `account_id`, `confirmation` (`"confirmed"` to send).

**2-step confirmation flow:** The first `send` call returns a preview with `gate: "awaiting_confirmation"`. The agent must call again with `confirmation='confirmed'` to actually dispatch. Controlled by `require_confirmation` setting (default `true`).

**Account resolution order:** explicit `account_id` → agent-specific default (`agent_default`) → global default (`is_default`) → first account. Passwords are always masked in output. Account `signature` field is auto-appended to the body. Port 587 → STARTTLS; other ports → SMTP_SSL.

---

### `inbox` — Agent-to-Operator Messaging

Agent-to-operator messaging and task queue. JSONL-backed with atomic writes.

**4 actions:**

| Action | What it does | Required params |
|---|---|---|
| `send` | Message the operator | `subject`, `body` |
| `add_task` | Add task to shared queue | `task` |
| `next_task` | Fetch & auto-complete oldest pending task | — |
| `ack` | Acknowledge a task/message by ID | `task_id` |

**Parameters:** `action`, `subject`, `body`, `task`, `task_id`, `type` (enum: `message`, `tool_request`, `warning`, `idea`), `priority` (enum: `low`, `normal`, `high`, `urgent`), `profile`, `needs_approval` (boolean), `dry_run` (boolean).

**Storage:** `data/shared/inbox.jsonl` (canonical) + `data/shared/inbox.md` (derived human-readable view). All actions support `dry_run=True` for preview without writing. `next_task` auto-marks the returned task as `"done"`.

---

### `cost_tracker` — Token Spend & Pricing

Token pricing management and cost tracking. Reads from the metering system.

**6 actions:**

| Action | What it does | Required params |
|---|---|---|
| `get_pricing` | Look up token pricing for a provider/model | — |
| `set_pricing` | Update pricing rates for a model | `provider`, `model` |
| `list_models` | List models from enabled LLM API connections | — |
| `cost_summary` | Aggregated spend (today/week/month/all-time) | — |
| `cost_log` | Recent cost log entries | — |
| `session_cost` | Cost for a specific chat session | `chat_id` |

**Parameters:** `action`, `provider`, `model`, `input_per_1m`, `cached_input_per_1m`, `output_per_1m`, `training_per_1m` (numbers), `agent` (filter), `since` (ISO 8601), `chat_id`, `limit` (default 50).

**Storage:** `config/pricing.yaml` (pricing registry) + `config/connections.json` (model list). Pricing lookup cascade: exact model match → prefix match → `_default` fallback. `set_pricing` resets the metering cache so new prices take effect immediately.

---

## Agent Tool Assignments

Each profile YAML declares an `allowed_tools` list:

| Profile | Tools |
|---|---|
| **astraea** | `echo`, `memory`, `directives`, `cost_tracker`, `continuation_update`, `web_search`, `email` |
| **callum** | `echo`, `memory`, `directives`, `continuation_update`, `web_search` |
| **codex_animus** | `echo`, `memory`, `directives`, `continuation_update`, `web_search` |

---

## Adding a New Tool

1. Create `src/tools/<name>.py` with a class exposing `definition()` and `execute(arguments)`
2. Register it in `registry.py → _TOOL_MAP` with `(module_path, ClassName, needs_instance)`
3. Add `<name>` to `allowed_tools` in the relevant profile YAMLs
4. The registry resolves it on next chat — no restart required if using `--reload`

## How Tool Definitions Are Used

Tool definitions are **not** part of the system prompt body. They are passed as a separate top-level `tools` parameter in the OpenAI API payload, so they are never at risk of context-window truncation. The chat loop supports up to 10 tool rounds per message (`MAX_TOOL_ROUNDS`). Tool results are returned as `role="tool"` messages.
