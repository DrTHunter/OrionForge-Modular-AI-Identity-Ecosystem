# orion-ui-standalone — Development Branch

> Active development workspace for the OrionForge Modular AI Identity Ecosystem.

This is where **all new features are built and tested** before being promoted to `engine/` (stable core) or `ui/` (production deployment).

---

## Quick Start

```powershell
cd orion-ui-standalone
pip install -r ../requirements.txt
python -m uvicorn web.app:app --host 0.0.0.0 --port 8989 --reload
```

Open **http://localhost:8989**.

---

## Structure

```
orion-ui-standalone/
├── web/                # FastAPI application (~6,300 lines, 168 routes, 16 templates)
│   ├── app.py          # Main application — all page & API routes
│   ├── auth.py         # Supabase OAuth + JWT verification (121 lines)
│   ├── stripe_billing.py  # Stripe subscriptions, credits, trial system (1,016 lines)
│   ├── image_gen.py    # Image generation helper (9 providers)
│   ├── static/         # CSS
│   └── templates/      # Jinja2 HTML (16 files — 15 pages + base layout)
│       ├── base.html           # Shared layout (nav, sidebar, footer, skin theming, auth state)
│       ├── login.html          # Supabase OAuth sign-in page
│       ├── plans.html          # Subscription tier selection (Free vs Pro)
│       ├── store.html          # Credit packs, one-time tool purchases, usage history
│       ├── admin_keys.html     # Admin panel — API key management
│       ├── chat.html           # Real-time agent chat with streaming
│       ├── profiles.html       # Agent profile manager — collapsible system prompt, soul script editor (FAISS-indexed), knowledge notes, avatar upload
│       ├── vault.html          # Memory vault browser with 8-field sort
│       ├── knowledge.html      # Knowledge notes browser
│       ├── knowledge_edit.html # Rich text knowledge editor
│       ├── tools.html          # Tool config, memory profiles, model router
│       ├── settings.html       # API connections, voice, image settings
│       ├── pricing.html        # LLM pricing registry editor
│       ├── skins.html          # 13 UI themes with live preview
│       ├── agi_loop.html       # AGI loop configuration
│       └── about.html          # Project wiki with auto-generated articles
│
├── src/                # Soul Script Engine modules (48 source files)
│   ├── data_paths.py   # Canonical data directory layout & auto-creation
│   ├── runtime_policy.py # RuntimePolicy dataclass — iteration limits, stasis, self-refine
│   ├── directives/     # Directive parsing, storage, injection, manifest system
│   ├── governance/     # Session-scoped directive tracking & change control
│   ├── llm_client/     # Multi-provider LLM abstraction (OpenAI, Anthropic, Ollama, DeepSeek)
│   ├── memory/         # FAISS-backed semantic memory — vault, chunking, PII guard, injection
│   ├── observability/  # Token metering, cost tracking, pricing engine
│   ├── policy/         # Boundary enforcement — risk classification, denial payloads
│   ├── routing/        # 6-tier model router, budget tracking, escalation chains
│   ├── storage/        # Note collection & user notes loading
│   └── tools/          # 11 tool implementations + registry (memory, directives, email, web search, inbox, model router, agi loop, runtime info, etc.)
│
├── config/             # Runtime configuration (18 files)
│   ├── connections.json       # LLM provider connections
│   ├── auth.json              # Supabase OAuth config (project URL, anon key)
│   ├── memory_profile.json    # Memory vault settings (retention, categories, safety)
│   ├── identity_profile.json  # FAISS identity indexing profile
│   ├── model_router.json      # Model router config (task-tier mapping, tiers, presets)
│   ├── pricing.yaml           # LLM pricing registry (493 lines, USD per 1M tokens)
│   ├── agi_loop.json          # AGI loop config (intervals, budgets, tiered routing)
│   ├── settings.json          # UI settings (timezone, avatars, backgrounds, Stripe state)
│   └── saved_profiles/        # Named config profile snapshots
│       └── router_presets/    # Named model router preset snapshots
│
├── profiles/           # Agent YAML profiles (16 agents — provider, model, parameters)
├── prompts/            # System prompt templates (*.system.md)
├── directives/         # Agent soul script / directive markdown files (auto-indexed into NotesFAISS)
├── notes/              # Developer notes per agent
├── scripts/            # Seed scripts (seed_memories.py, seed_ui_knowledge.py)
├── data/               # Runtime data (chats, memory vault, FAISS indexes, uploads, knowledge notes, agent trash)
└── tests/              # Test suite — 11 files, 263 functions, ~2,975 checks
```

---

## Authentication & Monetization

| Feature | Details |
|---|---|
| **Login** | Supabase OAuth (Google, GitHub, email) via `/login` |
| **JWT verification** | `auth.py` — JWKS-based token validation, path whitelist, session middleware |
| **Subscription** | $9.99/month Pro plan via Stripe Checkout (`/plans`) |
| **15-day trial** | Free trial on first sign-up, auto-expires to free tier. Trial state persisted via Fly.io volume (`/persist`) |
| **Credit system** | Buy credit packs in the store (`/store`), spend on platform-hosted LLM calls and tools |
| **LLM markup** | Platform-hosted calls billed at 2× base cost, deducted from credits |
| **TTS/STT billing** | Per-use billing for platform-hosted voice services (2× markup) |
| **One-time purchases** | Buy individual tool access from the store |
| **Admin panel** | `/admin/keys` — API key management, secured by OAuth email whitelist |
| **Tier gating** | Free tier vs Pro tier access control on all API endpoints |

---

## Dashboard Pages

| Page | URL | Description |
|------|-----|-------------|
| **Login** | `/login` | Supabase OAuth sign-in (Google, GitHub, email) |
| **Plans** | `/plans` | Subscription tier selection — Free vs Pro ($9.99/mo) |
| **Store** | `/store` | Credit packs, one-time tool purchases, usage history |
| **Chat** | `/chat` | Talk to agents — 3-mode connection (Platform Models / Auto Router / User Models), 5-layer identity injection (prompt → soul script → knowledge → memory → history) |
| **Profiles** | `/profiles` | Create/edit/delete agents — collapsible system prompt, soul script editor with FAISS-indexed badge, knowledge notes, model config, 30-day trash retention |
| **Vault** | `/vault` | Browse & search persistent memory — sort by 8 fields, max memory limits, metadata display |
| **Knowledge** | `/knowledge` | Rich text editor for soul scripts and always-on context notes |
| **Tools** | `/tools` | Configure tools, memory profiles, email, web search, cost tracking, model router with presets |
| **Settings** | `/settings` | API connections, user API keys (OpenAI, Anthropic, DeepSeek, OpenRouter, Google Gemini), chat backgrounds, timezone, voice/image settings |
| **Pricing** | `/pricing` | LLM pricing registry — view/edit per-model token costs |
| **Skins** | `/skins` | 13 UI themes with marketplace-style grid and live preview |
| **AGI Loop** | `/agi-loop` | Autonomous agent loop configuration (intervals, budgets, steps) |
| **Wiki** | `/about` | Project wiki with auto-generated articles from READMEs + custom notes editor |
| **Admin** | `/admin/keys` | Admin panel — API key management, secured by OAuth email whitelist |

---

## API Routes (168 endpoints)

The FastAPI app exposes 168 routes across these domains:

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
| Admin | ~3 | API keys CRUD, admin auth |

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

11 test files covering every engine module. Run all tests:

```powershell
cd orion-ui-standalone
python tests/run_all.py
```

Or run the torture suite:

```powershell
$env:PYTHONIOENCODING="utf-8"; python tests/test_torture.py
```

| Test File | Functions | Checks | Coverage |
|-----------|-----------|--------|----------|
| `test_torture.py` | 116 | ~2,979 | Deep torture of all code paths — memory, vault, sort, policy, tools, templates, model router, presets, 6-tier routing, sidecar wiring, soul script helpers, soul script API, soul script FAISS indexing, note collector soul script injection, profiles template collapsible, admin keys, chat 3-mode selector, user model catalog, `__userkey_` dynamic connections, Stripe state persistence, store catalog structure, tier & trial system, credit system, credit cost estimators, purchase flows (tool/skin/agent), agent ownership, user activity tracking, wipe user data, purge inactive, list all users, auth helpers, tier info structure |
| `test_memory.py` | 23 | 155 | VaultStore, MemoryVault, Memory types, PII guard |
| `test_stress.py` | 29 | 238 | Rapid-fire ops, concurrent access, boundary conditions, router presets, coding tiers |
| `test_registry_and_tools.py` | 17 | 86 | Tool registry, cost tracker, web search |
| `test_governance.py` | 16 | 74 | ActiveDirectives, validate_manifest |
| `test_directives.py` | 14 | 108 | Parser, store, injector, manifest, DirectivesTool |
| `test_chunker_injector.py` | 14 | 51 | Chunking logic, merge/split, formatting |
| `test_storage_and_llm.py` | 14 | 45 | User notes loader, LLM client base |
| `test_metering.py` | 11 | 92 | Token accounting, cost computation, aggregation |
| `test_data_paths.py` | 5 | 31 | Data directory layout, auto-creation, isolation |
| `test_tools.py` | 4 | 38 | EchoTool, ContinuationUpdateTool, EmailTool, RuntimePolicy |
| **Total** | **263** | **~3,897** | |

---

## Relationship to Other Directories

| Directory | Role | Updated When |
|-----------|------|--------------|
| **`orion-ui-standalone/`** | Active development | Every feature change |
| `engine/` | Stable frozen core | Features promoted after testing |
| `ui/` | Production deployment | Includes external Docker tool services |
| `services/` | Fly.io sidecar services | SearXNG, TTS, Whisper STT |

---

## Key Technologies

| Technology | Role |
|------------|------|
| FastAPI + Uvicorn | Web server & async API (168 routes) |
| FAISS (`faiss-cpu`) | Vector similarity search for memory + soul script retrieval |
| sentence-transformers | Semantic embeddings (`all-mpnet-base-v2`) |
| Jinja2 | HTML templates (16 files) |
| Supabase | OAuth authentication + JWT verification |
| Stripe | Subscription billing, credit system, webhook handling |
| Fly.io | Cloud hosting + Flycast private networking for sidecar services |
| PyYAML | Agent profile parsing |
| httpx | Async HTTP for model fetching & LLM calls |
