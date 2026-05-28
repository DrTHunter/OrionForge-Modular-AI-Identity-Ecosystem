# Engine  -  Stable Core

> Status: reviewed and refreshed on 2026-05-28.

The `engine/` directory contains the **frozen, stable core** of the OrionForge Soul Script Engine. This is the canonical reference implementation  -  battle-tested modules that power AI identity persistence, memory management, policy enforcement, and directive governance.

## Relationship to Other Directories

| Directory | Purpose |
|-----------|---------|
| **`engine/`** | Stable core  -  only updated when features are proven in `orion-ui-standalone/` |
| `orion-ui-standalone/` | Active development branch  -  new features land here first |
| `ui/` | Deployment build  -  production-ready with external tool services |
| `services/` | Fly.io sidecar services  -  SearXNG, TTS, Whisper STT |

## Module Map

```
engine/src/
├── data_paths.py          # Canonical data directory layout & auto-creation
├── runtime_policy.py      # RuntimePolicy dataclass  -  iteration limits, stasis, self-refine
│
├── directives/            # Directive parsing, storage, injection, manifest system
├── governance/            # Session-scoped directive tracking & change control
├── llm_client/            # Multi-provider LLM abstraction (OpenAI, Anthropic, Ollama, DeepSeek)
├── memory/                # FAISS-backed semantic memory  -  vault, chunking, PII guard, injection
├── observability/         # Token metering, cost tracking, pricing engine
├── policy/                # Boundary enforcement  -  risk classification, denial payloads
├── routing/               # 6-tier model router, budget tracking, escalation chains
├── storage/               # Note collection & user notes loading
└── tools/                 # 11 tool implementations + registry
```

## Key Capabilities

| Subsystem | What It Does |
|-----------|-------------|
| **Memory** | FAISS-backed semantic vault with 3-tier taxonomy (Canon/Register/Log), append-only JSONL, 8-stage write-gate, topic upsert, PII guard |
| **LLM Client** | Provider abstraction for OpenAI, Anthropic (native SDK), Ollama, and DeepSeek (OpenAI-compat) |
| **Directives** | H2-delimited directive parsing, SHA-256 manifest hashing, scoring (token overlap + SequenceMatcher), change control |
| **Governance** | Session-scoped directive registry, append-only JSONL audit log, drift detection |
| **Observability** | Per-request token metering, cost computation from `pricing.yaml`, aggregation with `+` operator, source tracking (platform vs user), date-range filtering, `by_source` cost aggregation |
| **Policy** | Risk classification (low/med/high), deterministic denial payloads, append-only event logging |
| **Storage** | Note collection (always-on vs directive modes), HTML stripping, dual note systems, soul script auto-injection into FAISS search |
| **Tools** | Memory (13 actions), directives (5 actions), web search, email, inbox, cost tracker, model router, agi loop, runtime info, echo, continuation |
| **Routing** | 6-tier model router (LOCAL_CHEAP, LOCAL_STRONG, CHEAP_CLOUD, EXPENSIVE_CLOUD, CODE_LIGHT, CODE_HEAVY) with task classification, escalation chains, budget-aware gating |

## Stability Contract

Code in `engine/` should not be modified for experimental features.  
New capabilities are developed in `orion-ui-standalone/src/`, tested thoroughly, then promoted here once stable.

Files are synced from `orion-ui-standalone/src/`  ->  `engine/src/` after passing the full test suite (276 functions, ~4,129 assertions).
