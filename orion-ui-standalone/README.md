# orion-ui-standalone  -  Development Branch

> Status: reviewed and refreshed on 2026-05-28.

> Active development workspace for the OrionForge Modular AI Identity Ecosystem.

This is where **all new features are built and tested**, and it's the app that deploys to Fly.io. Stable code is mirrored into `engine/` (the frozen core).

---

## Quick Start

```powershell
cd orion-ui-standalone
pip install -r ../requirements.txt
python -m uvicorn web.app:app --host 0.0.0.0 --port 8989 --reload
```

Open **http://localhost:8989**.

---

## VS Code Bridge

Use the optional MCP bridge when you want Copilot Agent in VS Code to route prompts through Orion Forge instead of calling the main app directly.

Bridge entrypoint:

```powershell
cd orion-ui-standalone
python scripts/orion_vscode_bridge.py
```

Bridge behavior:
- Default persona: `orion_cannon`
- Allowed overrides: `elysia_cannon`, `k_os`
- Request routing: one MCP tool named `orion_chat`
- Bridge model defaults: `ORION_VSCODE_BRIDGE_DEFAULT_MODEL` (primary) and `ORION_VSCODE_BRIDGE_FALLBACK_MODEL` (backup)
- Context handling: optional `context` text is prepended before the prompt, so the agent can be given chat summaries or code snippets without sending the whole repository every time

Configuration file:
- `config/vscode_bridge.json`

Remote auth (recommended):
- Fly requires authentication for `/api/chat/send`. Use a static **bridge key** instead of expiring tokens.
- Set `ORION_BRIDGE_API_KEY` as a Fly secret, set the bridge `auth.mode` to `bridge_key`, and provide the same key via `ORION_VSCODE_BRIDGE_KEY`.
- See [scripts/README.md](scripts/README.md) for the full key setup, owner-data sharing, and security notes.

Turn-off path:
- Stop launching the bridge and remove the MCP server entry from VS Code
- Remove the key with `fly secrets unset ORION_BRIDGE_API_KEY` and redeploy
- No core Orion files are modified by the bridge, so removal is clean

---

## Structure

```
orion-ui-standalone/
├── web/                # FastAPI application (~7,500 lines, 172 routes, 17 templates)
│   ├── app.py          # Main application  -  all page & API routes (multi-tenant aware)
│   ├── user_data.py    # Per-user data isolation layer  -  path helpers, validation, directory builders
│   ├── auth.py         # Supabase OAuth + JWT verification (142 lines)
│   ├── stripe_billing.py  # Stripe credit purchases, billing & credits (1,400 lines)
│   ├── image_gen.py    # Image generation helper (9 providers)
│   ├── static/         # CSS
│   └── templates/      # Jinja2 HTML (17 files  -  16 pages + base layout)
│       ├── base.html           # Shared layout (nav, sidebar, footer, skin theming, auth state)
│       ├── login.html          # Supabase OAuth sign-in page
│       ├── plans.html          # Legacy plans page (redirects to Store)
│       ├── store.html          # Credit packs, one-time tool purchases, usage history
│       ├── admin_keys.html     # Owner-only admin page
│       ├── admin_voices.html   # Admin panel  -  ElevenLabs voice allowlist management
│       ├── chat.html           # Real-time agent chat with streaming
│       ├── profiles.html       # Agent profile manager  -  collapsible system prompt, soul script editor (FAISS-indexed), knowledge notes, avatar upload
│       ├── vault.html          # Memory vault browser with 8-field sort
│       ├── knowledge.html      # Knowledge notes browser
│       ├── knowledge_edit.html # Rich text knowledge editor
│       ├── tools.html          # Tool config, memory profiles, model router
│       ├── settings.html       # API connections, voice, image settings
│       ├── pricing.html        # LLM pricing registry editor
│       ├── skins.html          # 13 UI themes with live preview
│       ├── agi_loop.html       # AGI loop  -  8-tab dashboard, journal modal, expandable loop log, VM storage
│       └── about.html          # Project wiki with auto-generated articles
│
├── src/                # Soul Script Engine modules (48 source files)
│   ├── data_paths.py   # Canonical data directory layout & auto-creation
│   ├── runtime_policy.py # RuntimePolicy dataclass  -  iteration limits, stasis, self-refine
│   ├── directives/     # Directive parsing, storage, injection, manifest system
│   ├── governance/     # Session-scoped directive tracking & change control
│   ├── llm_client/     # Multi-provider LLM abstraction (OpenAI, Anthropic, Ollama, DeepSeek)
│   ├── memory/         # FAISS-backed semantic memory  -  vault, chunking, PII guard, injection
│   ├── observability/  # Token metering, cost tracking, pricing engine
│   ├── policy/         # Boundary enforcement  -  risk classification, denial payloads
│   ├── routing/        # 6-tier model router, budget tracking, escalation chains
│   ├── storage/        # Note collection & user notes loading
│   └── tools/          # 11 tool implementations + registry (memory, directives, email, web search, inbox, model router, agi loop, runtime info, etc.)
│
├── config/             # Runtime configuration (18 files)
│   ├── connections.json       # LLM provider connections
│   ├── auth.json              # Authentication config
│   ├── memory_profile.json    # Memory vault settings (retention, categories, safety)
│   ├── identity_profile.json  # FAISS identity indexing profile
│   ├── model_router.json      # Model router config (task-tier mapping, tiers, presets)
│   ├── pricing.yaml           # LLM pricing registry (493 lines, USD per 1M tokens)
│   ├── agi_loop.json          # AGI loop config (intervals, budgets, tiered routing)
│   ├── settings.json          # UI settings (timezone, avatars, backgrounds, Stripe state)
│   └── saved_profiles/        # Named config profile snapshots
│       └── router_presets/    # Named model router preset snapshots
│
├── profiles/           # Agent YAML profiles (16 agents  -  provider, model, parameters)
├── prompts/            # System prompt templates (*.system.md)
├── directives/         # Agent soul script / directive markdown files (auto-indexed into NotesFAISS)
├── notes/              # Developer notes per agent
├── scripts/            # Seed scripts (seed_memories.py, seed_ui_knowledge.py)
│   └── orion_vscode_bridge.py  # Optional MCP bridge for VS Code Copilot Agent
├── data/               # Runtime data (global templates + per-user isolated directories)
│   └── users/          # Per-user isolated data trees (chats, memory, vault, notes, settings, profiles, uploads)
└── tests/              # Test suite  -  12 files, 295+ functions, ~4,350+ checks
```

---

## Authentication & Monetization

| Feature | Details |
|---|---|
| **Login** | Supabase OAuth (Google, GitHub, email) via `/login` |
| **JWT verification** | `auth.py`  -  JWKS-based token validation, path whitelist, session middleware |
| **Billing** | Pay-per-use — no monthly subscription. Usage billed in credits at 2× the API cost |
| **Free credits** | New accounts start with $2 in credits on first sign-up. No credit card required |
| **Credit system** | Buy credit packs in the store (`/store`), spend on platform-hosted LLM calls and tools |
| **LLM markup** | Platform-hosted calls billed at 2x base cost, deducted from credits |
| **TTS/STT billing** | Per-use billing for platform-hosted voice services (2x markup) |
| **One-time purchases** | Buy individual tool access from the store |
| **Admin panel** | Owner-only management area for voices and user management (wipe, purge inactive), restricted to allowlisted accounts |
| **Access** | Full access for every account — usage constrained only by credit balance |

---

## Dashboard Pages

| Page | URL | Description |
|------|-----|-------------|
| **Login** | `/login` | Supabase OAuth sign-in (Google, GitHub, email) |
| **Plans** | `/plans` | Legacy page  -  redirects to the Store (no subscription) |
| **Store** | `/store` | Credit packs, one-time tool purchases, usage history |
| **Chat** | `/chat` | Talk to agents  -  3-mode connection (Platform Models / Auto Router / User Models), 5-layer identity injection (prompt  ->  soul script  ->  knowledge  ->  memory  ->  history) |
| **Profiles** | `/profiles` | Create/edit/delete agents  -  collapsible system prompt, soul script editor with FAISS-indexed badge, knowledge notes, model config, 30-day trash retention |
| **Vault** | `/vault` | Browse & search persistent memory  -  sort by 8 fields, max memory limits, metadata display |
| **Knowledge** | `/knowledge` | Rich text editor for soul scripts and always-on context notes |
| **Tools** | `/tools` | Configure tools, memory profiles, email, web search, cost tracking, model router with presets |
| **Settings** | `/settings` | Model provider connections, chat backgrounds, timezone, voice/image settings |
| **Pricing** | `/pricing` | LLM pricing registry  -  view/edit per-model token costs |
| **Skins** | `/skins` | 13 UI themes with marketplace-style grid and live preview |
| **AGI Loop** | `/agi-loop` | Autonomous agent loop  -  8-tab dashboard (Dashboard, Inbox, Journal, Config, Pipeline, Model Router, Budget, Loop Log), journal popup modal with narrative details, expandable loop log entries, 6-tier model routing, VM-persistent tick history & journal |
| **Wiki** | `/about` | Project wiki with auto-generated articles from READMEs + custom notes editor |
| **Admin** | Owner only | Owner-only management area (voices, user management), restricted to allowlisted accounts |

---

## API Routes (172 endpoints)

The FastAPI app exposes 172 routes across these domains:

| Domain | Endpoints | Examples |
|--------|-----------|---------|
| Auth | ~6 | session, login, logout, callback, Supabase JWKS |
| Stripe Billing | ~8 | checkout, webhook, credits, subscription status, trial |
| Chat | ~15 | send, history, new, run, stop, folders |
| Profiles | ~14 | CRUD, avatar, knowledge attachment, soul script save/load, soft-delete with 30-day trash, restore, permanent delete |
| Vault | ~5 | add, batch_add, stats, delete, compact |
| Knowledge | ~7 | CRUD, folders |
| Connections | ~8 | CRUD, model probing, refresh, user model catalog |
| Pricing | ~6 | CRUD, cost summary, cost log |
| Tools Config | ~10 | web search, email, memory profiles |
| Settings | ~10 | backgrounds, timezone, API keys, voice, image |
| TTS/STT | ~5 | speak, voices, whisper |
| Skins | ~2 | get/set active skin |
| AGI Loop | ~2 | get/set config |
| Model Router | ~7 | get/set/reset config, presets CRUD (save/load/delete), 6-tier routing |
| Image Gen | ~5 | generate, providers, settings |
| Store | ~4 | credit packs, tool purchases, usage history |
| Admin | ~10 | platform keys CRUD, voice allowlist, user management (list/wipe/purge/delete), pricing sync |

---

## Multi-Tenant Data Isolation

Every user gets a fully isolated data directory tree under `data/users/{user_id}/`. No data is shared between users except the global read-only agent templates (profiles, prompts, directives).

### Per-User Directory Layout

```
data/users/{user_id}/
├── chats/              # Chat histories & index
│   ├── index.json
│   └── {chat_id}.json
├── memory/
│   ├── vault.jsonl     # Memory vault (jsonlines)
│   └── faiss/          # FAISS vector indexes
├── notes/
│   ├── index.json
│   ├── folders.json
│   └── {note_id}.json
├── settings.json       # Preferences, agent configs, avatars
├── profiles/           # Copy-on-write agent profile overrides
├── prompts/            # Copy-on-write system prompt overrides
├── directives/         # Copy-on-write soul script overrides
├── uploads/            # User-uploaded images
└── trash/
    └── profiles/       # Soft-deleted agents (30-day retention)
```

### Architecture

| Component | Mechanism |
|---|---|
| **Path routing** | `web/user_data.py`  -  18 path helper functions, all validated with regex `^[a-zA-Z0-9_-]{1,128}$` |
| **Path traversal prevention** | `_validate_user_id()` rejects `../`, `\`, slashes, null bytes, newlines, HTML, oversized IDs |
| **Request scoping** | `contextvars.ContextVar` (`_current_user_id`) set by `AuthMiddleware` on each request |
| **Data helpers** | All `_load_*` / `_save_*` helpers in `app.py` accept optional `user_id`, falling back to the contextvar |
| **Copy-on-write** | Profiles, system prompts, and soul scripts fall back to global templates when no user override exists |
| **Per-user instances** | FAISS indexes and VaultStore instances are cached per `user_id`  -  no cross-user contamination |
| **Directory creation** | `ensure_user_dirs(user_id)` called by `AuthMiddleware` on every authenticated request |
| **Admin wipe** | `DELETE /api/admin/users/{uid}` removes the entire user directory tree |

### What's Isolated per User

- Chat histories & index
- Memory vault (jsonlines) & FAISS vector indexes
- Knowledge notes & folders
- Settings (preferences, agent configs, avatars, backgrounds)
- Agent profile overrides (copy-on-write from global templates)
- System prompt overrides
- Soul script / directive overrides
- Uploaded images (avatars, backgrounds)
- Trash (soft-deleted agents)

### What's Shared (Read-Only)

- Global agent profiles (`profiles/*.yaml`)
- Global system prompts (`prompts/*.system.md`)
- Global soul scripts (`directives/*.md`)
- Engine modules (`src/`)
- Config files (`config/`)

---

## Cloud Sidecar Services (Fly.io)

Three sidecar services run on Fly.io with Flycast private IPv6 networking:

| Service | Fly.io App | Port | Purpose |
|---|---|---|---|
| **SearXNG** | `orionforge-engine-searxng` | 8080 | Meta-search engine (Google, DuckDuckGo, Bing, Wikipedia, GitHub, Arxiv, StackOverflow) |
| **OpenedAI Speech** | `orionforge-engine-tts` | 8000 | Text-to-speech (Piper + XTTS v2) with persistent volume |
| **Whisper** | `orionforge-engine-whisper` | 8000 | Speech-to-text (faster-whisper + FastAPI) |

The main app discovers sidecars via environment variables (`TTS_URL`, `WHISPER_URL`, `SEARXNG_URL`) with fallback to `connections.json`.

---

## Image Generation

9 providers supported via `web/image_gen.py`:

| Provider | Models |
|----------|--------|
| **OpenAI** | DALL-E 3, DALL-E 2, GPT Image (`gpt-image-1`) |
| **Google** | Imagen 3 |
| **Stability AI** | Stable Image Ultra, Core, SD3 Large/Turbo/Medium |
| **Ideogram** | V2, V2 Turbo |
| **Replicate** | Flux Pro, Flux Schnell, Flux Dev, Playground v2.5 |
| **FAL.ai** | Flux Pro v1.1, Flux Schnell, Flux Dev |
| **Leonardo AI** | Diffusion XL, Lightning XL, Vision XL, Kino XL |
| **Midjourney** | Via third-party API proxy |

---

## Test Suite

12 test files covering every engine module. Run all tests:

```powershell
cd orion-ui-standalone
python tests/run_all.py
```

Or run the torture suite:

```powershell
$env:PYTHONIOENCODING="utf-8"; python tests/test_torture.py
```

Or run multi-tenant isolation tests:

```powershell
python -m tests.test_multi_tenant
```

| Test File | Functions | Checks | Coverage |
|-----------|-----------|--------|----------|
| `test_torture.py` | 132 | ~3,228 | Deep torture of all code paths  -  memory, vault, sort, policy, tools, templates, model router, presets, 6-tier routing, sidecar wiring, soul script helpers, soul script API, soul script FAISS indexing, note collector soul script injection, profiles template collapsible, admin keys, admin voices API & template, admin user management, connections CRUD, pricing CRUD, chat 3-mode selector, user model catalog, `__userkey_` dynamic connections, Stripe state persistence, store catalog structure, tier & trial system, credit system, credit cost estimators, purchase flows (tool/skin/agent), agent ownership, user activity tracking, wipe user data, purge inactive, list all users, auth helpers, tier info structure, runtime info tool, TTS voice filter logic, ElevenLabs/inworld connection helpers, `_check_admin` helper, AGI history disk persistence, journal popup modal, expandable loop log, VM storage paths |
| `test_multi_tenant.py` | 16 | 181 | Multi-tenant data isolation  -  path helpers, path traversal prevention (10+ attack vectors), directory isolation, chat CRUD isolation, settings isolation, knowledge/notes isolation, vault isolation, profile copy-on-write, prompt & soul script copy-on-write, uploads isolation, 5-user x 20-chat stress, VaultStore per-user instances, agent config isolation, admin wipe cleanup, trash isolation, 20-user massive isolation stress |
| `test_memory.py` | 23 | 155 | VaultStore, MemoryVault, Memory types, PII guard |
| `test_stress.py` | 29 | 238 | Rapid-fire ops, concurrent access, boundary conditions, router presets, coding tiers |
| `test_registry_and_tools.py` | 17 | 86 | Tool registry, cost tracker, web search |
| `test_governance.py` | 16 | 74 | ActiveDirectives, validate_manifest |
| `test_directives.py` | 14 | 108 | Parser, store, injector, manifest, DirectivesTool |
| `test_chunker_injector.py` | 14 | 51 | Chunking logic, merge/split, formatting |
| `test_storage_and_llm.py` | 14 | 45 | User notes loader, LLM client base |
| `test_metering.py` | 11 | 115 | Token accounting, cost computation, aggregation, source tracking |
| `test_data_paths.py` | 5 | 31 | Data directory layout, auto-creation, isolation |
| `test_tools.py` | 4 | 38 | EchoTool, ContinuationUpdateTool, EmailTool, RuntimePolicy |
| **Total** | **295** | **~4,350** | |

---

## Relationship to Other Directories

| Directory | Role | Updated When |
|-----------|------|--------------|
| **`orion-ui-standalone/`** | Active development + Fly deployment | Every feature change |
| `engine/` | Stable frozen core | Features promoted after testing |
| `services/` | Fly.io sidecar services | SearXNG, TTS, Whisper STT |

---

## Key Technologies

| Technology | Role |
|------------|------|
| FastAPI + Uvicorn | Web server & async API (172 routes) |
| FAISS (`faiss-cpu`) | Vector similarity search for memory + soul script retrieval |
| sentence-transformers | Semantic embeddings (`all-mpnet-base-v2`) |
| Jinja2 | HTML templates (17 files) |
| Supabase | OAuth authentication + JWT verification |
| Stripe | Subscription billing, credit system, webhook handling |
| Fly.io | Cloud hosting + Flycast private networking for sidecar services |
| PyYAML | Agent profile parsing |
| httpx | Async HTTP for model fetching & LLM calls |
