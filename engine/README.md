# Engine — Stable Core

The `engine/` directory contains the **frozen, stable core** of the OrionForge Soul Script Engine. This is the canonical reference implementation — battle-tested modules that power AI identity persistence, memory management, policy enforcement, and directive governance.

## Relationship to Other Directories

| Directory | Purpose |
|-----------|---------|
| **`engine/`** | Stable core — only updated when features are proven in `orion-ui-standalone/` |
| `orion-ui-standalone/` | Active development branch — new features land here first |
| `ui/` | Deployment build — production-ready with external tool services |

## Module Map

```
engine/src/
├── data_paths.py          # Canonical data directory layout & auto-creation
├── runtime_policy.py      # RuntimePolicy dataclass — iteration limits, stasis, self-refine
│
├── directives/            # Directive parsing, storage, injection, manifest system
├── governance/            # Session-scoped directive tracking & change control
├── llm_client/            # Multi-provider LLM abstraction (OpenAI, Anthropic, Ollama, DeepSeek)
├── memory/                # FAISS-backed semantic memory — vault, chunking, PII guard, injection
├── observability/         # Token metering, cost tracking, pricing engine
├── policy/                # Boundary enforcement — risk classification, denial payloads
├── storage/               # Note collection & user notes loading
└── tools/                 # Tool implementations — echo, continuation, memory, directives
```

## Stability Contract

Code in `engine/` should not be modified for experimental features.  
New capabilities are developed in `orion-ui-standalone/src/`, tested thoroughly, then promoted here once stable.
