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
├── web/                # FastAPI application (app.py — 3,556 lines, 113 routes, 12 templates)
│   ├── app.py          # Main application — all page & API routes
│   ├── image_gen.py    # Image generation helper (8 providers)
│   ├── static/         # CSS
│   └── templates/      # Jinja2 HTML (12 pages)
│
├── src/                # Soul Script Engine modules (33 source files)
│   ├── data_paths.py   # Canonical data directory layout & auto-creation
│   ├── runtime_policy.py # RuntimePolicy dataclass — iteration limits, stasis, self-refine
│   ├── directives/     # Directive parsing, storage, injection, manifest system
│   ├── governance/     # Session-scoped directive tracking & change control
│   ├── llm_client/     # Multi-provider LLM abstraction (OpenAI, Anthropic, Ollama, DeepSeek)
│   ├── memory/         # FAISS-backed semantic memory — vault, chunking, PII guard, injection
│   ├── observability/  # Token metering, cost tracking, pricing engine
│   ├── policy/         # Boundary enforcement — risk classification, denial payloads
│   ├── storage/        # Note collection & user notes loading
│   └── tools/          # 8 tool implementations + registry (memory, directives, email, web search, inbox, etc.)
│
├── config/             # Runtime configuration (11 files)
│   ├── connections.json       # LLM provider connections
│   ├── memory_profile.json    # Memory vault settings (retention, categories, safety)
│   ├── identity_profile.json  # FAISS identity indexing profile
│   ├── pricing.yaml           # LLM pricing registry (577 lines, USD per 1M tokens)
│   ├── agi_loop.json          # AGI loop config (intervals, budgets, tiered routing)
│   ├── settings.json          # UI settings (timezone, avatars, backgrounds)
│   └── saved_profiles/        # Named config profile snapshots
│
├── profiles/           # Agent YAML profiles (provider, model, parameters)
├── prompts/            # System prompt templates (*.system.md)
├── directives/         # Agent directive markdown files
├── notes/              # Developer notes per agent
├── scripts/            # Seed scripts (seed_memories.py, seed_ui_knowledge.py)
├── data/               # Runtime data (chats, memory vault, FAISS indexes, uploads, knowledge notes)
└── tests/              # Test suite — 11 files, 205 functions, ~1,905 checks
```

---

## Dashboard Pages

| Page | URL | Description |
|------|-----|-------------|
| **Chat** | `/chat` | Talk to agents — 5-layer identity injection (prompt → soul script → knowledge → memory → history) |
| **Profiles** | `/profiles` | Create/edit agents, system prompts, attach knowledge, configure models |
| **Vault** | `/vault` | Browse & search persistent memory — sort by 8 fields, max memory limits, metadata display |
| **Knowledge** | `/knowledge` | Rich text editor for soul scripts and always-on context notes |
| **Tools** | `/tools` | Configure tools, memory profiles, email, web search, cost tracking |
| **Settings** | `/settings` | API connections, chat backgrounds, timezone, voice/image settings |
| **Pricing** | `/pricing` | LLM pricing registry — view/edit per-model token costs |
| **Skins** | `/skins` | 12 UI themes with marketplace-style grid and live preview |
| **AGI Loop** | `/agi-loop` | Autonomous agent loop configuration (intervals, budgets, steps) |
| **About** | `/about` | Editable project about page |

---

## API Routes (113 endpoints)

The FastAPI app exposes 113 routes across these domains:

| Domain | Endpoints | Examples |
|--------|-----------|---------|
| Chat | ~15 | send, history, new, run, stop, folders |
| Profiles | ~10 | CRUD, avatar, knowledge attachment |
| Vault | ~5 | add, batch_add, stats, delete, compact |
| Knowledge | ~7 | CRUD, folders |
| Connections | ~8 | CRUD, model probing, refresh |
| Pricing | ~6 | CRUD, cost summary, cost log |
| Tools Config | ~10 | web search, email, memory profiles |
| Settings | ~10 | backgrounds, timezone, API keys, voice, image |
| TTS/STT | ~5 | speak, voices, whisper |
| Skins | ~2 | get/set active skin |
| AGI Loop | ~2 | get/set config |
| Model Router | ~3 | get/set/reset config |
| Image Gen | ~5 | generate, providers, settings |

---

## Image Generation

8 providers supported via `web/image_gen.py`:

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
| `test_torture.py` | 65 | ~1,086 | Deep torture of all code paths — memory, vault, sort, policy, tools, templates |
| `test_memory.py` | 23 | 155 | VaultStore, MemoryVault, Memory types, PII guard |
| `test_stress.py` | 22 | 139 | Rapid-fire ops, concurrent access, boundary conditions |
| `test_registry_and_tools.py` | 17 | 86 | Tool registry, cost tracker, web search |
| `test_governance.py` | 16 | 74 | ActiveDirectives, validate_manifest |
| `test_directives.py` | 14 | 108 | Parser, store, injector, manifest, DirectivesTool |
| `test_chunker_injector.py` | 14 | 51 | Chunking logic, merge/split, formatting |
| `test_storage_and_llm.py` | 14 | 45 | User notes loader, LLM client base |
| `test_metering.py` | 11 | 92 | Token accounting, cost computation, aggregation |
| `test_data_paths.py` | 5 | 31 | Data directory layout, auto-creation, isolation |
| `test_tools.py` | 4 | 38 | EchoTool, ContinuationUpdateTool, EmailTool, RuntimePolicy |
| **Total** | **205** | **~1,905** | |

---

## Relationship to Other Directories

| Directory | Role | Updated When |
|-----------|------|--------------|
| **`orion-ui-standalone/`** | Active development | Every feature change |
| `engine/` | Stable frozen core | Features promoted after testing |
| `ui/` | Production deployment | Includes external Docker tool services |

---

## Key Technologies

| Technology | Role |
|------------|------|
| FastAPI + Uvicorn | Web server & async API (113 routes) |
| FAISS (`faiss-cpu`) | Vector similarity search for memory + soul script retrieval |
| sentence-transformers | Semantic embeddings (`all-mpnet-base-v2`) |
| Jinja2 | HTML templates (12 pages) |
| PyYAML | Agent profile parsing |
| httpx | Async HTTP for model fetching & LLM proxy calls |
| PyYAML | Agent profile parsing |
| httpx | Async HTTP for model fetching & LLM calls |
