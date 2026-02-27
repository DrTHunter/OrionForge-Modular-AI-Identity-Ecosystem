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

Users can browse, download, install, and run them in their own virtual environments.

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

## Repository Structure

This repo contains the **Modular Web UI** and its supporting **engine core**, organized as two clearly separated modules:

```
OrionForge-Modular-AI-Identity-Ecosystem/
├── ui/                   # 🖥️  The Web UI
│   ├── web/              # FastAPI app, templates, static assets
│   │   ├── app.py        # Main application (36 routes)
│   │   ├── static/       # CSS
│   │   └── templates/    # Jinja2 HTML templates (9 pages)
│   ├── config/           # connections.json, settings.json, about.json
│   ├── data/             # Runtime data (chats, memory vault, uploads)
│   ├── profiles/         # Agent identity YAML files
│   ├── prompts/          # System prompt markdown (*.system.md)
│   ├── directives/       # Agent directive markdown files
│   ├── notes/            # Agent note markdown files
│   ├── scripts/          # Utility scripts (seed_memories.py)
│   ├── tests/            # Unit tests
│   └── tools/            # External tool services (Docker)
│       ├── email_service/ # SMTP email relay
│       ├── openedai_speech/ # Text-to-speech
│       ├── searxng/       # Meta-search engine
│       └── whisper_stt/   # Speech-to-text
│
├── engine/               # ⚙️  The SoulScript Engine Core
│   └── src/
│       ├── memory/       # FAISS memory, vault, chunker, PII guard, notes FAISS
│       ├── llm_client/   # LLM API clients (OpenAI-compat, Anthropic, Ollama)
│       ├── directives/   # Directive parser, injector, manifest, store
│       ├── governance/   # Active directive enforcement & anti-drift tracking
│       ├── storage/      # Note collection & user notes loader
│       ├── observability/ # Token metering & cost tracking
│       ├── policy/       # Boundary enforcement & capability gating
│       └── tools/        # Built-in tool implementations
│
├── Dockerfile            # Docker build for the full stack
├── docker-compose.yml    # One-command launch
├── requirements.txt      # Python dependencies
└── README.md
```

---

## Dashboard Pages

| Page | URL | Description |
|---|---|---|
| **Chat** | `/chat` | Talk to agents — identity layers injected automatically |
| **Profiles** | `/profiles` | Create/edit agents, system prompts, attach knowledge |
| **Vault** | `/vault` | Browse & search the persistent memory vault |
| **Knowledge** | `/knowledge` | Create notes that agents use as Soul Script or always-on context |
| **Tools** | `/tools` | View available tool services |
| **Settings** | `/settings` | Manage API connections (OpenAI, Ollama, OpenRouter, etc.) |
| **About** | `/about` | Editable project about page |

---

## Getting Started

### Prerequisites

| Requirement | Version |
|---|---|
| **Python** | 3.10+ (3.11 recommended) |
| **Git** | Any |

> **First launch note:** The engine uses `sentence-transformers` with the `all-mpnet-base-v2` model (~420 MB). It downloads automatically on first launch and is cached for future runs.

### Clone & Install

```bash
git clone https://github.com/DrTHunter/OrionForge-Modular-AI-Identity-Ecosystem.git
cd OrionForge-Modular-AI-Identity-Ecosystem
pip install -r requirements.txt
```

### Windows — Run Locally

```powershell
cd ui
python -m uvicorn web.app:app --host 0.0.0.0 --port 8989
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
| **FastAPI** + **Uvicorn** | Web server & API |
| **FAISS** (`faiss-cpu`) | Vector similarity search for memory + soul script retrieval |
| **sentence-transformers** | Semantic embeddings (`all-mpnet-base-v2`) |
| **Jinja2** | HTML templates |
| **PyYAML** | Agent profile parsing |
| **httpx** | Async HTTP for model fetching & LLM proxy calls |

---

## Troubleshooting

| Issue | Fix |
|---|---|
| **Port already in use (Windows)** | `Get-NetTCPConnection -LocalPort 8989 \| ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }` |
| **Port already in use (Mac/Linux)** | `lsof -ti:8989 \| xargs kill` |
| **Port already in use (Docker)** | `docker compose down` then restart |
| **ModuleNotFoundError** | Make sure you `cd` into `ui/` before running uvicorn |
| **No API connection** | Add one at `/settings` |
| **Slow first start** | The 420 MB embedding model downloads once; subsequent starts are fast |
| **FAISS import error** | Run `pip install faiss-cpu` (not `faiss`) |
| **Docker can't reach Ollama** | Use `http://host.docker.internal:11434/v1` as the connection URL |

---

## Related

- **[SoulScript Engine](https://github.com/DrTHunter/SoulScript-Engine)** — The core engine repo with full Soul Script documentation, examples, and the specification format

---

## License

See [LICENSE](LICENSE) for details.
