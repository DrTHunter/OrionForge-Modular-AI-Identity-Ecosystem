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
2. **Soul Script** — FAISS semantic retrieval from directive-mode knowledge notes
3. **Always-On Knowledge** — Verbatim text from always-mode attached knowledge
4. **Memory Vault** — FAISS search over the agent's persistent memories (`vault.jsonl`)
5. **Conversation History** — Recent user/assistant turns (truncated to 30k char budget)

Agents can also **save memories** during conversation using `[MEMORY_SAVE: ...]` tags, which are automatically extracted and written to the vault.

---

## Project Structure

OrionForge is organized into three directories — an active development branch, a stable frozen core, and a production deployment build:

```
OrionForge/
├── orion-ui-standalone/  # 🔧 Active Development Branch
│   ├── web/              # FastAPI app (4,168 lines, 127 routes, 12 templates)
│   │   ├── app.py        # Main application — all page & API routes
│   │   ├── image_gen.py  # Image generation (8 providers)
│   │   ├── static/       # CSS
│   │   └── templates/    # Jinja2 HTML templates (12 pages)
│   ├── src/              # Soul Script Engine modules (34 source files)
│   │   ├── memory/       # FAISS memory, vault, chunker, PII guard, notes FAISS
│   │   ├── llm_client/   # LLM API clients (OpenAI, Anthropic, Ollama, DeepSeek)
│   │   ├── directives/   # Directive parser, injector, manifest, store
│   │   ├── governance/   # Active directive enforcement & anti-drift tracking
│   │   ├── storage/      # Note collection & user notes loader
│   │   ├── observability/ # Token metering & cost tracking
│   │   ├── policy/       # Boundary enforcement & capability gating
│   │   └── tools/        # 11 tool implementations + registry
│   ├── config/           # 11 config files (connections, pricing, memory profile, etc.)
│   ├── data/             # Runtime data (chats, memory vault, FAISS indexes, uploads)
│   ├── profiles/         # Agent identity YAML files
│   ├── prompts/          # System prompt markdown (*.system.md)
│   ├── directives/       # Agent directive markdown files
│   ├── notes/            # Agent note markdown files
│   ├── scripts/          # Seed scripts (seed_memories.py, seed_ui_knowledge.py)
│   └── tests/            # Test suite (11 files, 210 functions, ~2,895 checks)
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
│       └── tools/        # Built-in tool implementations
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

## Dashboard Pages

| Page | URL | Description |
|---|---|---|
| **Chat** | `/chat` | Talk to agents — 5-layer identity injection (prompt → soul script → knowledge → memory → history) |
| **Profiles** | `/profiles` | Create/edit agents, system prompts, attach knowledge, configure models |
| **Vault** | `/vault` | Browse & search persistent memory — sort by 8 fields, max memory limits, metadata display |
| **Knowledge** | `/knowledge` | Rich text editor for soul scripts and always-on context notes |
| **Tools** | `/tools` | Configure tools, memory profiles, email, web search, cost tracking, model router with presets |
| **Settings** | `/settings` | API connections, chat backgrounds, timezone, voice/image settings |
| **Pricing** | `/pricing` | LLM pricing registry — view/edit per-model token costs |
| **Skins** | `/skins` | 12 UI themes with marketplace-style grid and live preview |
| **AGI Loop** | `/agi-loop` | Autonomous agent loop configuration (intervals, budgets, steps) |
| **About** | `/about` | Editable project about page |

---

## LLM & Image Providers

### Chat / Completion

The engine connects to any **OpenAI-compatible** endpoint. Native provider support:

| Provider | Client | Notes |
|---|---|---|
| **OpenAI** | `openai_compat` | GPT-4o, GPT-4 Turbo, GPT-3.5, o1, o3, etc. |
| **Anthropic** | `anthropic` | Claude 4 Opus/Sonnet, Claude 3.5, native SDK |
| **DeepSeek** | `openai_compat` | DeepSeek-V3, DeepSeek-R1 via OpenAI-compatible API |
| **Ollama** | `ollama` | Any local model (Llama, Mistral, Phi, Qwen, etc.) |
| OpenRouter, LM Studio, etc. | `openai_compat` | Any OpenAI-compatible endpoint |

### Image Generation

8 providers supported via `image_gen.py`:

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

| Service | Purpose |
|---|---|
| **ElevenLabs** | Cloud TTS (high quality, API key required) |
| **Edge-TTS** | Free Microsoft TTS (no API key) |
| **openedai-speech** | Self-hosted TTS (Piper + XTTS v2, Docker) |
| **Whisper** | Speech-to-text transcription (Docker) |

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
| `model_router` | Intelligent task-based model routing with tier classification and presets |
| `agi_loop` | Autonomous agent loop control (start, stop, pause, resume, status) |
| `runtime_info` | System runtime information and environment details |
| `echo` | Debug/test tool — echoes input back |
| `continuation_update` | Multi-turn continuation status updates |

---

## Test Suite

11 test files with **210** test functions and **~2,895** assertions:

```powershell
cd orion-ui-standalone
python tests/run_all.py
```

| Test File | Functions | Checks | Coverage Area |
|---|---|---|---|
| `test_torture.py` | 67 | 1,948 | Deep torture of all code paths — memory, vault, sort, policy, tools, templates, model router, presets |
| `test_memory.py` | 23 | 154 | VaultStore, MemoryVault, Memory types, PII guard |
| `test_stress.py` | 25 | 248 | Rapid-fire ops, concurrent access, boundary conditions, router presets |
| `test_registry_and_tools.py` | 17 | 99 | Tool registry, cost tracker, web search |
| `test_governance.py` | 16 | 72 | ActiveDirectives, validate_manifest |
| `test_directives.py` | 14 | 107 | Parser, store, injector, manifest, DirectivesTool |
| `test_chunker_injector.py` | 14 | 69 | Chunking logic, merge/split, formatting |
| `test_storage_and_llm.py` | 14 | 44 | User notes loader, LLM client base |
| `test_metering.py` | 11 | 91 | Token accounting, cost computation, aggregation |
| `test_data_paths.py` | 5 | 30 | Data directory layout, auto-creation, isolation |
| `test_tools.py` | 4 | 33 | EchoTool, ContinuationUpdateTool, EmailTool, RuntimePolicy |
| **Total** | **210** | **~2,895** | |

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

## Included Agents

| Agent | Description |
|---|---|
| **Astraea** | Default agent profile |
| **Callum** | Secondary agent profile |
| **Codex Animus** | The "Creator of Souls" — meta-agent that helps users design soul scripts and build their own AIs |

Each agent has its own profile YAML, system prompt, directives, and memory scopes.

---

## External Tool Services (Optional)

These run as separate Docker containers via `docker compose` inside their respective `ui/tools/` folders. They are **not required** for the core engine.

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
| **FastAPI** + **Uvicorn** | Web server & async API (127 routes) |
| **FAISS** (`faiss-cpu`) | Vector similarity search for memory + soul script retrieval |
| **sentence-transformers** | Semantic embeddings (`all-mpnet-base-v2`) |
| **Jinja2** | HTML templates (12 pages) |
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

---

## License

OrionForge is proprietary software. All rights reserved. See [LICENSE](LICENSE) for details.
