# UI — Production Deployment Build

The `ui/` directory is the **production-ready deployment build** of OrionForge. It mirrors the structure of `orion-ui-standalone/` but includes additional external tool services for speech, search, and email.

## Relationship to Other Directories

| Directory | Purpose |
|-----------|---------|
| `engine/` | Stable core — frozen reference modules |
| `orion-ui-standalone/` | Active development — new features land here first |
| **`ui/`** | **Production deployment** — includes external services, Docker configs, clean data dirs |

## Structure

```
ui/
├── config/             # Configuration files (connections, settings, pricing, profiles)
├── data/               # Runtime data directories (clean placeholder files)
├── directives/         # Agent directive files (astraea, callum, codex_animus, shared)
├── notes/              # Developer notes per agent
├── profiles/           # Agent YAML profiles (provider, model, parameters)
├── prompts/            # System prompt templates (*.system.md)
├── scripts/            # Seed scripts for initial data population
├── tests/              # Test suite
├── tools/              # External tool services (Docker-based)
│   ├── email_service/  # SMTP relay for agent email sending
│   ├── openedai_speech/# TTS via Piper/XTTS (Docker)
│   ├── searxng/        # Privacy-focused web search (Docker)
│   └── whisper_stt/    # Speech-to-text via Whisper (Docker)
└── web/                # FastAPI web application
```

## External Tool Services

The `tools/` subdirectory contains Docker-based microservices that extend agent capabilities:

| Service | Port | Purpose |
|---------|------|---------|
| SearXNG | 8888 | Privacy-focused web search aggregator |
| OpenedAI Speech | 8000 | Text-to-speech (Piper for CPU, XTTS for GPU) |
| Whisper STT | 9000 | Speech-to-text transcription |
| Email Service | 5050 | SMTP relay for sending emails |

## Quick Start

```bash
# Start external tool services
cd ui/tools/searxng && docker compose up -d
cd ui/tools/openedai_speech && docker compose up -d
cd ui/tools/whisper_stt && docker compose up -d

# Start the main application
cd ui && python -m uvicorn web.app:app --host 0.0.0.0 --port 8989 --reload
```

## Data Directory

The `data/` directory is pre-populated with placeholder files to maintain the directory structure. All runtime data (chats, memories, uploads) is generated during use.
