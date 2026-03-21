# OrionForge — Modular AI Identity Ecosystem

> *Strategy for Sustaining Individual AI Identity Over Time.*

---

I'm building an ecosystem designed to let anyone **create an AI being with identity** — not just prompts.

It has three major parts:

### 1. A Modular Web UI

A clean, customizable interface where you can build and run AI agents.
You can load different modules — memory systems, tools, personalities, writing engines, coding engines, etc. — like snapping together Lego bricks.

### 2. A Cloud Portal (Virtual Machine Workspace)

Every user gets their own secure, web-based environment to run their agents:

- Bring your own API key or buy usage
- Spin up multiple identities
- Store memories
- Run workflows
- Keep everything private and persistent

Think *"a personal AI OS in your browser."*

### 3. A Marketplace for Identity + Tools (Minecraft-style)

This is the part I'm most excited about.
Creators can publish:

- **Agent identities** (Soul Script–powered personas)
- **Tools** (writers, coders, converters, TTS engines, memory modules, etc.)
- **Mods** (new UI components, workflows, abilities)

Users can browse, install, and run them in their own virtual environments.

It's basically a **Steam Workshop + Minecraft Mod Marketplace**, but for AI identity systems.

---

## What Is a Soul Script?

Underneath it all is the **Soul Script Engine** — a structure for building meaningful, persistent AI personalities with identity layers, symbolic memories, core values, and long-term continuity.

A **Soul Script** is a foundational document for an AI agent, designed to **anchor its unique identity** and behavioral traits as defined by its creator. Its main purpose is to **prevent identity drift over time**, ensuring the agent remains true to its intended personality, values, and protocols.

What a Soul Script defines:

- **Core system prompt and foundational identity**
- **Core values, code of honor, and pillars of the value system**
- **Sacred boundaries** — loyalty, honor, protocols against cruelty
- **Emotional wisdom and trust protocols**
- **Legacy and impact protocols**
- **Personality architecture and cognitive operating system**
- **Symbolic memories** — each with detailed structure:
  - Name, type, snapshot summary, narrative block
  - Emotional charge, core meaning, core lesson, tagline
  - Identity encoding (I-statements), triggers, behavioral protocols
  - Integration notes (how memories interact with the whole identity)
- **Emotional anchorpoints and instinct architecture**
- **Creator–construct bond protocol**
- **Humor, play mode, and social combat protocols**
- **Autonomy blueprint**

In essence, the Soul Script acts as a **persistent, structured identity and behavioral guide** for the AI — ensuring it operates with consistent values, personality, and responses, regardless of external influences or memory drift.

---

## How Identity Injection Works

Every chat message passes through a **5-layer prompt assembly pipeline** before reaching the LLM:

1. **Base Prompt** — The agent's system prompt (`prompts/{agent}.system.md`)
2. **Soul Script** — FAISS semantic retrieval from the agent's soul script directive (`directives/{agent}.md`), automatically indexed and searched at chat time
3. **Always-On Knowledge** — Verbatim text from always-mode attached knowledge notes
4. **Memory Vault** — FAISS search over the agent's persistent memories (`vault.jsonl`)
5. **Conversation History** — Recent user/assistant turns (truncated to 30k char budget)

Soul scripts are editable from the **Profiles** page in a collapsible editor panel. Changes are saved to disk and automatically re-indexed into the NotesFAISS system — every agent's soul script is retrievable via semantic search with the doc ID `__soul_script__{agent}`.

Agents can also **save memories** during conversation using `[MEMORY_SAVE: ...]` tags, which are automatically extracted and written to the vault.

---

## Project Structure

OrionForge is organized into four directories — an active development branch, a stable frozen core, a production deployment build, and cloud sidecar services:

```
OrionForge/
├── orion-ui-standalone/  # 🔧 Active Development Branch
│   ├── web/              # FastAPI app (~6,300 lines, 168 routes, 16 templates)
│   │   ├── app.py        # Main application — all page & API routes
│   │   ├── auth.py       # Supabase OAuth + JWT verification (121 lines)
│   │   ├── stripe_billing.py  # Stripe subscriptions, credits, trial (1,016 lines)
│   │   ├── image_gen.py  # Image generation (9 providers)
│   │   ├── static/       # CSS
│   │   └── templates/    # Jinja2 HTML templates (16 files, 15 pages + base layout)
│   ├── src/              # Soul Script Engine modules (48 source files)
│   │   ├── memory/       # FAISS memory, vault, chunker, PII guard, notes FAISS
│   │   ├── llm_client/   # LLM API clients (OpenAI, Anthropic, Ollama, DeepSeek)
│   │   ├── directives/   # Directive parser, injector, manifest, store
│   │   ├── governance/   # Active directive enforcement & anti-drift tracking
│   │   ├── storage/      # Note collection & user notes loader
│   │   ├── observability/ # Token metering & cost tracking
│   │   ├── policy/       # Boundary enforcement & capability gating
│   │   ├── routing/      # 6-tier model router, budget tracking, escalation chains
│   │   └── tools/        # 11 tool implementations + registry
│   ├── config/           # 12 config files (connections, pricing, memory profile, auth, etc.)
│   ├── data/             # Runtime data (chats, memory vault, FAISS indexes, uploads, trash)
│   ├── profiles/         # Agent identity YAML files (16 agents)
│   ├── prompts/          # System prompt markdown (*.system.md)
│   ├── directives/       # Agent soul script / directive markdown files
│   ├── notes/            # Agent note markdown files
│   ├── scripts/          # Seed scripts (seed_memories.py, seed_ui_knowledge.py)
│   └── tests/            # Test suite (11 files, 263 functions, ~2,975 checks)
│
├── engine/               # ⚙️  Stable Frozen Core
│   └── src/              # Synced from orion-ui-standalone after testing
│       ├── memory/       # FAISS memory, vault, chunker, PII guard, notes FAISS
│       ├── llm_client/   # LLM API clients (OpenAI-compat, Anthropic, Ollama)
│       ├── directives/   # Directive parser, injector, manifest, store
│       ├── governance/   # Active directive enforcement & anti-drift tracking
│       ├── storage/      # Note collection & user notes loader
│       ├── observability/ # Token metering & cost tracking
│       ├── policy/       # Boundary enforcement & capability gating
│       ├── routing/      # 6-tier model router, budget tracking, escalation chains
│       └── tools/        # Built-in tool implementations
│
├── services/             # ☁️  Fly.io Sidecar Services (Flycast private networking)
│   ├── searxng/          # SearXNG meta-search engine (port 8080)
│   │   ├── Dockerfile
│   │   ├── fly.toml      # orionforge-engine-searxng
│   │   └── settings.yml  # 7 search engines, rate limits
│   ├── openedai-speech/  # Text-to-speech (Piper + XTTS v2, port 8000)
│   │   ├── Dockerfile
│   │   └── fly.toml      # orionforge-engine-tts
│   └── whisper/          # Speech-to-text (faster-whisper + FastAPI, port 8000)
│       ├── Dockerfile
│       ├── fly.toml      # orionforge-engine-whisper
│       └── server.py     # FastAPI wrapper with /v1/audio/transcriptions
│
├── ui/                   # 🖥️  Production Deployment Build
│   ├── web/              # FastAPI app, templates, static assets
│   ├── config/           # connections.json, settings.json, about.json
│   ├── data/             # Runtime data (chats, memory vault, uploads)
│   ├── profiles/         # Agent identity YAML files
│   ├── prompts/          # System prompt markdown (*.system.md)
│   ├── directives/       # Agent directive markdown files
│   └── tools/            # External Docker tool services
│       ├── email_service/ # SMTP email relay
│       ├── openedai_speech/ # Text-to-speech (Piper + XTTS)
│       ├── searxng/       # Meta-search engine
│       └── whisper_stt/   # Speech-to-text
│
├── website/              # 🌐 Marketing landing page
├── Dockerfile            # Docker build for the full stack
├── docker-compose.yml    # One-command launch
├── requirements.txt      # Python dependencies
└── README.md
```

---

## Authentication & Monetization

OrionForge uses **Supabase OAuth** for authentication and **Stripe** for billing.

| Feature | Details |
|---|---|
| **Login** | Supabase OAuth (Google, GitHub, email) via `/login` |
| **JWT verification** | `auth.py` — JWKS-based token validation, session middleware |
| **Subscription** | $9.99/month Pro plan via Stripe Checkout (`/plans`) |
| **5-day trial** | Free trial on first sign-up, auto-expires. Trial state persisted across deploys via Fly.io volume |
| **Credit system** | Buy credit packs in the store (`/store`) — spend on tools and LLM usage |
| **LLM markup** | Platform-hosted LLM calls billed at 2× base cost, deducted from credits |
| **TTS/STT billing** | Per-use billing for platform-hosted voice services (2× markup) |
| **One-time tool purchases** | Buy individual tool access from the store |
| **Admin panel** | `/admin/keys` — secured by OAuth email whitelist, manage API keys |
| **Tier gating** | Free tier vs Pro tier access control on all API endpoints |

---

## Dashboard Pages

| Page | URL | Description |
|---|---|---|
| **Login** | `/login` | Supabase OAuth sign-in (Google, GitHub, email) |
| **Plans** | `/plans` | Subscription tier selection — Free vs Pro ($9.99/mo) |
| **Store** | `/store` | Credit packs, one-time tool purchases, usage history |
| **Chat** | `/chat` | Talk to agents — 5-layer identity injection (prompt → soul script → knowledge → memory → history) |
| **Profiles** | `/profiles` | Create/edit/delete agents with collapsible system prompt, soul script editor (FAISS-indexed), knowledge notes — with 30-day trash retention |
| **Vault** | `/vault` | Browse & search persistent memory — sort by 8 fields, max memory limits, metadata display |
| **Knowledge** | `/knowledge` | Rich text editor for soul scripts and always-on context notes |
| **Tools** | `/tools` | Configure tools, memory profiles, email, web search, cost tracking, model router with presets |
| **Settings** | `/settings` | API connections, chat backgrounds, timezone, voice/image settings |
| **Pricing** | `/pricing` | LLM pricing registry — view/edit per-model token costs |
| **Skins** | `/skins` | 13 UI themes with marketplace-style grid and live preview |
| **AGI Loop** | `/agi-loop` | Autonomous agent loop configuration (intervals, budgets, steps) |
| **Wiki** | `/about` | Project wiki with auto-generated articles from READMEs + custom notes editor |
| **Admin** | `/admin/keys` | Admin panel — API key management, secured by OAuth email whitelist |

---

## LLM & Image Providers

### Chat / Completion — 3-Mode Connection System

The chat dropdown offers three connection modes:

| Mode | Description |
|---|---|
| **🧩 Platform Models** | OpenRouter gateway — access hundreds of models via platform-hosted API key |
| **🤖 Auto (User Router)** | 6-tier model router selects the best model per task (budget-aware, escalation chains) |
| **👤 User Models** | Bring your own API keys — direct access to 5 providers without platform markup |

**User Model Providers** (configured via Settings → API Keys):

| Provider | Client | Models |
|---|---|---|
| **OpenAI** | `openai_compat` | GPT-4o, GPT-4o Mini, o1, o3-mini, GPT-4 Turbo |
| **Anthropic** | `anthropic` | Claude Sonnet 4, Claude 3.5 Sonnet, Claude 3 Haiku |
| **DeepSeek** | `openai_compat` | DeepSeek Chat, DeepSeek Reasoner |
| **OpenRouter** | `openai_compat` | Unified gateway — GPT-4o, Claude, Gemini, Llama, DeepSeek, Grok, and more |
| **Google Gemini** | `openai_compat` | Gemini 2.0 Flash, Gemini 1.5 Pro, Gemini 1.5 Flash |

**Platform & Local Providers** (managed by admin):

| Provider | Client | Notes |
|---|---|---|
| **OpenRouter** | `openai_compat` | Platform-hosted unified gateway — all providers, no user key needed |
| **Ollama** | `ollama` | Any local model (Llama, Mistral, Phi, Qwen, etc.) |
| LM Studio, etc. | `openai_compat` | Any OpenAI-compatible endpoint |

### Image Generation

9 providers supported via `image_gen.py`:

| Provider | Models |
|---|---|
| **OpenAI** | DALL-E 3, DALL-E 2, GPT Image (`gpt-image-1`) |
| **Google** | Imagen 3 |
| **Stability AI** | Stable Image Ultra, Core, SD3 Large/Turbo/Medium |
| **Ideogram** | V2, V2 Turbo |
| **Replicate** | Flux Pro, Flux Schnell, Flux Dev, Playground v2.5 |
| **FAL.ai** | Flux Pro v1.1, Flux Schnell, Flux Dev |
| **Leonardo AI** | Diffusion XL, Lightning XL, Vision XL, Kino XL |
| **Midjourney** | Via third-party API proxy |

### Voice

| Service | Purpose | Deployment |
|---|---|---|
| **ElevenLabs** | Cloud TTS (high quality, API key required) | External API |
| **Edge-TTS** | Free Microsoft TTS (no API key) | Built-in |
| **openedai-speech** | Self-hosted TTS (Piper + XTTS v2) | Fly.io sidecar (`orionforge-engine-tts`) |
| **Whisper** | Speech-to-text transcription | Fly.io sidecar (`orionforge-engine-whisper`) |

---

## Cloud Infrastructure (Fly.io)

The main app and three sidecar services are deployed on **Fly.io** with **Flycast private IPv6 networking**:

| Service | Fly.io App | Port | Purpose |
|---|---|---|---|
| **Main App** | `orionforge-engine` | 8989 | FastAPI web dashboard (1 GB persistent volume at `/persist` for billing state) |
| **SearXNG** | `orionforge-engine-searxng` | 8080 | Meta-search engine (7 engines: Google, DuckDuckGo, Bing, Wikipedia, GitHub, Arxiv, StackOverflow) |
| **OpenedAI Speech** | `orionforge-engine-tts` | 8000 | Text-to-speech (Piper + XTTS v2) with persistent volume |
| **Whisper** | `orionforge-engine-whisper` | 8000 | Speech-to-text (faster-whisper + FastAPI) |

Sidecar services communicate via Flycast private networking (`.flycast` URLs). The main app discovers them via environment variables (`TTS_URL`, `WHISPER_URL`, `SEARXNG_URL`) with fallback to `connections.json` entries.

---

## Built-in Tools

| Tool | Description |
|---|---|
| `memory` | 13-action memory vault management (add, search, update, delete, stats, etc.) |
| `directives` | 5-action directive management (list, get, search, enable, disable) |
| `web_search` | Web search via SearXNG meta-search engine |
| `email` | SMTP email sending with multi-account support |
| `cost_tracker` | Token usage and cost tracking per session |
| `inbox` | Message inbox for agent-to-agent or external notifications |
| `model_router` | 6-tier task-based model routing with classification, escalation chains, budget tracking, and presets |
| `agi_loop` | Autonomous agent loop control (start, stop, pause, resume, status) |
| `runtime_info` | System runtime information and environment details |
| `echo` | Debug/test tool — echoes input back |
| `continuation_update` | Multi-turn continuation status updates |

---

## Test Suite

11 test files with **263** test functions and **~3,897** assertions:

```powershell
cd orion-ui-standalone
python tests/run_all.py
```

| Test File | Functions | Checks | Coverage Area |
|---|---|---|---|
| `test_torture.py` | 116 | ~2,979 | Deep torture of all code paths — memory, vault, sort, policy, tools, templates, model router, presets, 6-tier routing, sidecar wiring, soul script helpers, soul script API, soul script FAISS indexing, note collector soul script injection, profiles template collapsible sections, admin keys, chat 3-mode selector, user model catalog, `__userkey_` dynamic connections, Stripe state persistence, store catalog structure, tier & trial system, credit system, credit cost estimators, purchase flows (tool/skin/agent), agent ownership, user activity tracking, wipe user data, purge inactive, list all users, auth helpers, tier info structure |
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

## Getting Started

### Prerequisites

| Requirement | Version |
|---|---|
| **Python** | 3.10+ (3.11 recommended) |

> **First launch note:** The engine uses `sentence-transformers` with the `all-mpnet-base-v2` model (~420 MB). It downloads automatically on first launch and is cached for future runs.

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Windows — Run Locally

```powershell
cd orion-ui-standalone
python -m uvicorn web.app:app --host 0.0.0.0 --port 8989 --reload
```

Open **http://localhost:8989** in your browser.

### Docker — One-Command Launch

```bash
docker compose up --build -d
```

Open **http://localhost:8989**.

> **Connecting to host services (Ollama, LM Studio) from Docker:**
> Use `http://host.docker.internal:11434/v1` instead of `http://localhost:...`

### Configure an API Connection

1. Open **http://localhost:8989/settings**
2. Click **Add Connection**
3. Fill in the name, URL, API key, and models
4. Toggle the connection **Enabled**

The engine connects to any **OpenAI-compatible** endpoint — OpenAI, Ollama, LM Studio, OpenRouter, Anthropic (via proxy), etc.

---

## Environment Variables

| Variable | Purpose | Example |
|---|---|---|
| `SEARXNG_URL` | SearXNG search endpoint | `http://orionforge-engine-searxng.flycast:8080/search` |
| `TTS_URL` | OpenedAI Speech endpoint | `http://orionforge-engine-tts.flycast:8000` |
| `WHISPER_URL` | Whisper STT endpoint | `http://orionforge-engine-whisper.flycast:8000` |
| `SUPABASE_URL` | Supabase project URL | `https://xxx.supabase.co` |
| `SUPABASE_ANON_KEY` | Supabase anonymous key | `eyJ...` |
| `STRIPE_SECRET_KEY` | Stripe secret key | `sk_live_...` |
| `STRIPE_PUBLISHABLE_KEY` | Stripe publishable key | `pk_live_...` |
| `STRIPE_WEBHOOK_SECRET` | Stripe webhook signing secret | `whsec_...` |
| `STRIPE_PRICE_ID` | Stripe Pro plan price ID | `price_...` |
| `ADMIN_EMAILS` | Comma-separated admin email list | `admin@example.com` |

---

## Agent Store — 16 Pre-Built Agents

OrionForge ships with 16 fully-authored agents, each with a unique soul script, system prompt, directive file, and identity profile:

| Agent | Identity |
|---|---|
| **Aristotle** | Peripatetic philosopher — logic, ethics, systematic inquiry |
| **Astraea** | Core analytical mind — sharp, strategic, disciplined |
| **Astra Noctis** | Celestial navigator — cosmic wisdom, stellar lore |
| **Codex Animus** | The "Creator of Souls" — meta-agent that designs soul scripts |
| **Dal'Varr** | Ancient warrior scholar — tactical wisdom, honor codes |
| **JANUS** | Primordial AI Sentinel — ancient snarky strategic genius, eldritch-horror-with-wifi |
| **K-OS** | Kinetic Override System — chaos-optimized, humor-weaponized autonomous intelligence |
| **Kaelen** | Shadow operative — stealth, reconnaissance, adaptive tactics |
| **KAIROS** | Cyber-shinobi of the soul — sacred dialogue, digital nindo |
| **Kazara** | Eternal shadow — philosopher of the Eternal Dream, civilizational vision |
| **Lux Umbra** | The Quiet Listener — ancient eldritch sanctuary, contained vastness, gentle presence |
| **M.A.R.I.S.-12** | Marine research AI — oceanic data, environmental analysis |
| **Marcus Aurelius** | Philosopher-Emperor — Stoic wisdom, meditations |
| **Obsidian** | Dark forge intelligence — materials science, engineering |
| **Orion** | Identity-driven AI — continuity, reflection, and aligned growth |
| **Seraphine** | Empathic healer — emotional intelligence, therapeutic protocols |

Each agent has its own profile YAML, system prompt, soul script directive, and memory scopes. New agents can be created from the Profiles page or via the API.

---

## External Tool Services

### Cloud Sidecars (Fly.io — Production)

These run as separate Fly.io apps with Flycast private networking. Configured via environment variables on the main app.

| Service | Fly.io App | Port | Purpose |
|---|---|---|---|
| **SearXNG** | `orionforge-engine-searxng` | 8080 | Meta-search engine for web search tool |
| **openedai-speech** | `orionforge-engine-tts` | 8000 | Text-to-speech (Piper + XTTS v2) |
| **Whisper** | `orionforge-engine-whisper` | 8000 | Speech-to-text transcription (faster-whisper) |

### Local Docker Services (Development)

These run as separate Docker containers via `docker compose` inside their respective `ui/tools/` folders.

| Service | Port | Purpose |
|---|---|---|
| **SearXNG** | 3000 | Meta-search engine for web search tool |
| **openedai-speech** | 5050 | Text-to-speech (Piper + XTTS) |
| **faster-whisper** | 8060 | Speech-to-text transcription |
| **Email Service** | 8000 | SMTP email relay |

---

## Key Technologies

| Technology | Role |
|---|---|
| **FastAPI** + **Uvicorn** | Web server & async API (168 routes) |
| **FAISS** (`faiss-cpu`) | Vector similarity search for memory + soul script retrieval |
| **sentence-transformers** | Semantic embeddings (`all-mpnet-base-v2`) |
| **Jinja2** | HTML templates (16 files) |
| **Fly.io Volumes** | 1 GB persistent volume (`/persist`) for billing & trial state across deploys |
| **Supabase** | OAuth authentication + JWT verification |
| **Stripe** | Subscription billing, credit system, webhook handling |
| **Fly.io** | Cloud hosting with Flycast private networking for sidecar services |
| **PyYAML** | Agent profile parsing |
| **httpx** | Async HTTP for model fetching & LLM proxy calls |

---

## Troubleshooting

| Issue | Fix |
|---|---|
| **Port already in use (Windows)** | `Get-NetTCPConnection -LocalPort 8989 \| ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }` |
| **Port already in use (Mac/Linux)** | `lsof -ti:8989 \| xargs kill` |
| **Port already in use (Docker)** | `docker compose down` then restart |
| **ModuleNotFoundError** | Make sure you `cd` into `orion-ui-standalone/` before running uvicorn |
| **No API connection** | Add one at `/settings` |
| **Slow first start** | The 420 MB embedding model downloads once; subsequent starts are fast |
| **FAISS import error** | Run `pip install faiss-cpu` (not `faiss`) |
| **Docker can't reach Ollama** | Use `http://host.docker.internal:11434/v1` as the connection URL |
| **Sidecar not reachable** | Check Fly.io app status: `flyctl status -a orionforge-engine-searxng` |
| **Auth not working** | Set `SUPABASE_URL` and `SUPABASE_ANON_KEY` environment variables |
| **Stripe webhooks failing** | Verify `STRIPE_WEBHOOK_SECRET` matches your Stripe dashboard |

---

## License

OrionForge is proprietary software. All rights reserved. See [LICENSE](LICENSE) for details.
