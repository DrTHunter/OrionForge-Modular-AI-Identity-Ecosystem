# config/

> Status: reviewed and refreshed on 2026-05-28.

Configuration files for the runtime and web dashboard. 17 config files (11 core + saved profiles & router presets) controlling connections, billing, routing, identity, and UI behavior.

## Files

| File | Purpose |
|------|---------|
| `config.example.yaml` | Example YAML config structure (default profile, data dir, global policy overrides) |
| `state.example.json` | Example state file format (window size, message array) |
| `connections.json` | LLM provider connections  -  platform-hosted API endpoints, keys, enabled models. Used by the 🧩 Platform Models chat mode. Also stores sidecar service URLs (SearXNG, TTS, Whisper) as fallback when env vars are not set. User API keys for the 👤 User Models mode are stored in `settings.json`. Managed via Dashboard  ->  Settings. |
| `auth.json` | Supabase OAuth config  -  `supabase_url`, `supabase_anon_key`, `jwt_secret`, `admin_emails` whitelist. Required for authentication. |
| `settings.json` | UI settings  -  timezone, chat background, agent avatars, per-agent display/voice/model config. Also stores user API keys (OpenAI, Anthropic, DeepSeek, OpenRouter, Google Gemini) for the User Models chat mode. Auto-created on first save. |
| `stripe_state.json` | Stripe billing, subscription, trial, and credit state. On Fly.io, persisted to `/persist/stripe_state.json` via a 1 GB volume so trial data survives deploys. Falls back to `config/stripe_state.json` locally. |
| `about.json` | About wiki custom notes content (editable from the web UI at `/about`) |
| `agi_loop.json` | AGI loop configuration  -  interval (30 min default), ticks/loop, steps/tick, budget caps ($20/mo hard, $16 soft, $2/session, $0.10/tick), tiered routing |
| `identity_profile.json` | FAISS identity indexing profile  -  chunk size (400 tokens), overlap (80), retrieval top_k, merge strategy for soul script indexing |
| `memory_profile.json` | Memory vault settings  -  retention policy (max memories, decay strategy, max pinned), category policy (open/strict mode, suggested categories), safety policy (custom hard rules) |
| `model_router.json` | Model router config  -  6-tier task -> model routing (LOCAL_CHEAP, LOCAL_STRONG, CHEAP_CLOUD, EXPENSIVE_CLOUD, CODE_LIGHT, CODE_HEAVY) |
| `pricing.yaml` | LLM Pricing Registry  -  USD per 1M tokens across 4 dimensions (input, cached_input, output, reasoning). ~506 lines covering all major providers. |

## Saved Profiles

```
saved_profiles/
├── identity/
│   └── __default__.json     # Default identity FAISS profile
└── memory/
    ├── __default__.json     # Default memory profile
    ├── Test.json            # Named memory profile snapshot
    └── test_234.json        # Named memory profile snapshot
```

Named profiles allow saving and loading different memory/identity configurations via the Tools page.

## Related Directories

- `profiles/`  -  per-agent YAML configs (provider, model, allowed tools, parameters)
- `data/`  -  runtime state (chats, memory vault, FAISS indexes, uploads, knowledge notes)
