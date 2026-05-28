# web/

> Status: reviewed and refreshed on 2026-05-28.

Orion Forge  -  the web dashboard for the agent runtime. A full-featured browser-based control panel built with FastAPI, Jinja2, and vanilla JavaScript. Includes Supabase OAuth authentication, Stripe subscription billing, and a credit-based monetization system.

## Quick Start

```bash
cd orion-ui-standalone
python -m uvicorn web.app:app --host 0.0.0.0 --port 8989 --reload
# Open http://localhost:8989
```

## Files

| File | Purpose |
|------|---------|
| `app.py` | FastAPI application  -  all routes, helpers, and API endpoints (172 routes, ~7,500 lines, multi-tenant aware) |
| `user_data.py` | Per-user data isolation layer  -  path helpers, user_id validation, directory builders, copy-on-write support (152 lines) |
| `auth.py` | Supabase OAuth + JWT verification  -  JWKS validation, session middleware, path whitelist (142 lines) |
| `stripe_billing.py` | Stripe subscription system  -  checkout, webhooks, credits, trial management, tier gating (1,400 lines) |
| `image_gen.py` | Image generation helper (9 providers: OpenAI DALL-E/GPT Image, Google Imagen, Stability, Ideogram, Replicate, FAL, Leonardo, Midjourney) |
| `static/style.css` | Stylesheet for the dashboard |
| `templates/` | Jinja2 HTML templates (17 files  -  16 pages + base layout) |

## Templates

| Template | Page |
|----------|------|
| `base.html` | Shared layout (nav, sidebar, footer, skin theming, auth state, trial banner) |
| `login.html` | Supabase OAuth sign-in (Google, GitHub, email) |
| `plans.html` | Subscription tier selection  -  Free vs Pro ($9.99/mo) |
| `store.html` | Credit packs, one-time tool purchases, usage history |
| `admin_keys.html` | Admin panel  -  API key management (secured by email whitelist) |
| `admin_voices.html` | Admin panel  -  ElevenLabs voice allowlist (search, filter, premium toggle, bulk save) |
| `chat.html` | Real-time chat with agents  -  streaming, tool execution, folders |
| `vault.html` | Memory vault browser  -  sort by 8 fields, metadata display, max memory limits |
| `profiles.html` | Agent profile viewer/editor  -  collapsible system prompt, soul script editor (FAISS-indexed badge), knowledge notes, avatar upload, create new agents |
| `settings.html` | API connections, timezone, chat background, voice/image settings |
| `tools.html` | Tool registry, memory profiles, email config, web search config, cost tracking |
| `knowledge.html` | Knowledge notes browser with folders |
| `knowledge_edit.html` | Rich text knowledge note editor |
| `pricing.html` | LLM pricing registry  -  view/edit per-model token costs |
| `skins.html` | 13 UI themes  -  marketplace grid with live preview |
| `agi_loop.html` | AGI loop configuration (intervals, budgets, steps) |
| `about.html` | Project wiki with auto-generated articles from READMEs + custom notes editor |

## API Endpoints (172 routes)

### Pages (17 routes)

| Route | Description |
|-------|-------------|
| `GET /` | Redirect to `/chat` |
| `GET /login` | OAuth sign-in page |
| `GET /plans` | Subscription plans |
| `GET /store` | Credit store |
| `GET /chat` | Chat interface |
| `GET /profiles` | Profile manager |
| `GET /vault` | Memory vault (supports `?sort=`, `?scope=`, `?category=`, `?q=`) |
| `GET /knowledge` | Knowledge notes |
| `GET /knowledge/{note_id}/edit` | Knowledge note editor |
| `GET /settings` | Settings page |
| `GET /tools` | Tool registry & config |
| `GET /pricing` | Pricing  ->  redirect |
| `GET /skins` | UI skins marketplace |
| `GET /agi-loop` | AGI loop config |
| `GET /about` | Wiki page |
| `GET /admin/keys` | Admin panel  -  API keys |
| `GET /admin/voices` | Admin panel  -  voice allowlist |

### Auth API (~6 routes)

| Route | Description |
|-------|-------------|
| `GET /api/auth/session` | Get current session / user info |
| `POST /api/auth/login` | Create session from Supabase token |
| `POST /api/auth/logout` | Clear session |
| `GET /api/auth/callback` | OAuth callback handler |
| `GET /api/auth/config` | Public Supabase config (URL, anon key) |

### Stripe Billing API (~8 routes)

| Route | Description |
|-------|-------------|
| `POST /api/stripe/checkout` | Create Stripe Checkout session |
| `POST /api/stripe/webhook` | Stripe webhook handler |
| `GET /api/stripe/subscription` | Get subscription status |
| `POST /api/stripe/credits/buy` | Purchase credit pack |
| `GET /api/stripe/credits/balance` | Get credit balance |
| `POST /api/stripe/credits/deduct` | Deduct credits (internal) |
| `GET /api/stripe/trial` | Get trial status |
| `GET /api/stripe/tier` | Get user tier (free/pro) |

### Chat API (~15 routes)

| Route | Description |
|-------|-------------|
| `POST /api/chat/send` | Send message to agent (streaming SSE) |
| `GET /api/chat/history` | List all chats |
| `POST /api/chat/new` | Create new chat |
| `GET /api/chat/{chat_id}` | Get chat by ID |
| `DELETE /api/chat/{chat_id}` | Delete chat |
| `PUT /api/chat/{chat_id}` | Update chat |
| `POST /api/chat/{chat_id}/title` | Update chat title |
| `POST /api/chats/{chat_id}/emoji` | Set chat emoji |
| `POST /api/chat/run` | Start async agent run |
| `GET /api/chat/status/{session_id}` | Check run status |
| `POST /api/chat/stop/{session_id}` | Stop active run |
| `POST /api/chats/folders` | Create chat folder |
| `PUT /api/chats/folders/{folder_id}` | Update folder |
| `DELETE /api/chats/folders/{folder_id}` | Delete folder |
| `PUT /api/chats/{chat_id}/move` | Move chat to folder |

### Profiles API (~14 routes)

| Route | Description |
|-------|-------------|
| `GET /api/profiles/{name}` | Get agent profile |
| `PUT /api/profiles/{name}` | Update agent profile |
| `POST /api/profiles` | Create new agent |
| `POST /api/profiles/create` | Create agent (alternate) |
| `DELETE /api/profiles/{name}` | Soft-delete agent (moves to 30-day trash) |
| `GET /api/profiles/trash` | List trashed agents |
| `POST /api/profiles/trash/{id}/restore` | Restore agent from trash |
| `DELETE /api/profiles/trash/{id}` | Permanently delete trashed agent |
| `PUT /api/profiles/{name}/config` | Update profile config |
| `PUT /api/profiles/{name}/avatar` | Upload agent avatar |
| `PUT /api/profiles/user` | Update user profile |
| `PUT /api/profiles/{name}/knowledge` | Update agent knowledge attachments |

### Vault API (~5 routes)

| Route | Description |
|-------|-------------|
| `POST /api/vault/add` | Add memory to vault |
| `POST /api/vault/batch_add` | Batch add memories |
| `GET /api/vault/stats` | Vault statistics |
| `POST /api/vault/delete` | Delete memory |
| `GET /api/vault/compact` | Compact vault (remove soft-deleted entries) |

### Knowledge API (~7 routes)

| Route | Description |
|-------|-------------|
| `POST /api/knowledge` | Create knowledge note |
| `PUT /api/knowledge/{note_id}` | Update note |
| `DELETE /api/knowledge/{note_id}` | Delete note |
| `GET /api/knowledge/{note_id}` | Get note |
| `GET /api/knowledge/folders` | List folders |
| `POST /api/knowledge/folders` | Create folder |
| `PUT /api/knowledge/folders/{folder_id}` | Update folder |
| `DELETE /api/knowledge/folders/{folder_id}` | Delete folder |

### Connections API (~8 routes)

| Route | Description |
|-------|-------------|
| `GET /api/connections` | List connections |
| `POST /api/connections` | Create connection |
| `PUT /api/connections/{conn_id}` | Update connection |
| `DELETE /api/connections/{conn_id}` | Delete connection |
| `GET /api/connections/{conn_id}/models` | List models for connection |
| `POST /api/connections/probe-models` | Probe a URL for available models |
| `GET /api/connections/all-models` | All models across all connections |
| `POST /api/connections/refresh-all-models` | Refresh model lists |

### Pricing API (~6 routes)

| Route | Description |
|-------|-------------|
| `GET /api/pricing` | Get pricing registry |
| `PUT /api/pricing` | Update pricing registry |
| `PUT /api/pricing/{provider}/{model:path}` | Set model price |
| `DELETE /api/pricing/{provider}/{model:path}` | Delete model price |
| `GET /api/pricing/models` | List priced models |
| `GET /api/pricing/cost-summary` | Cost summary |
| `GET /api/pricing/cost-log` | Cost log |

### Tools Config API (~10 routes)

| Route | Description |
|-------|-------------|
| `GET /api/tools/web_search/config` | Web search config |
| `PUT /api/tools/web_search/config` | Update web search config |
| `GET /api/tools/email/config` | Email config |
| `PUT /api/tools/email/config` | Update email config |
| `GET /api/tools/email/accounts` | List email accounts |
| `POST /api/tools/email/accounts` | Add email account |
| `DELETE /api/tools/email/accounts` | Remove email account |
| `POST /api/tools/email/test` | Test email send |
| `GET /api/tools/memory/profile` | Memory profile |
| `PUT /api/tools/memory/profile` | Update memory profile |
| CRUD `/api/tools/memory/profiles` | Named memory profile snapshots |
| CRUD `/api/tools/memory/identity` | Identity FAISS profile |

### Settings API (~10 routes)

| Route | Description |
|-------|-------------|
| `POST /api/settings/chat-background` | Upload chat background |
| `DELETE /api/settings/chat-background` | Remove background |
| `PUT /api/settings/timezone` | Set timezone |
| `GET /api/settings/api-keys` | Get API keys |
| `PUT /api/settings/api-keys` | Update API keys |
| `GET /api/settings/image` | Image gen config |
| `PUT /api/settings/image` | Update image gen config |
| `GET /api/settings/voice` | Voice settings |
| `PUT /api/settings/voice` | Update voice settings |
| `POST /api/settings/voice/*/voices` | Fetch available voices |
| `POST /api/settings/voice/whisper/test` | Test whisper STT |

### Other APIs

| Route | Description |
|-------|-------------|
| `GET /api/skin` | Get active skin |
| `PUT /api/skin` | Set active skin |
| `GET /api/agi-loop/config` | AGI loop config |
| `POST /api/agi-loop/config` | Update AGI loop config |
| `GET /api/model-router/config` | Model router config |
| `POST /api/model-router/config` | Update model router |
| `POST /api/model-router/reset` | Reset model router |
| `POST /api/about` | Update about content |
| `POST /api/tts/speak` | Text-to-speech |
| `GET /api/tts/voices` | TTS voice list |
| `POST /api/tts/edge/speak` | Edge TTS |
| `GET /api/tts/edge/voices` | Edge TTS voices |
| `POST /api/stt/whisper` | Speech-to-text |
| `GET /api/stt/whisper/status` | Whisper status |
| `GET /api/ollama/models` | List Ollama models |
| `POST /api/image/generate` | Generate image |
| `GET /api/health` | Health check |

### Admin API (~10 routes)

| Route | Description |
|-------|-------------|
| `POST /api/admin/platform-keys` | Save platform API key |
| `POST /api/admin/platform-keys/test` | Test platform API key |
| `DELETE /api/admin/platform-keys/{provider}` | Delete platform key |
| `POST /api/admin/sync-openrouter-pricing` | Sync OpenRouter pricing |
| `GET /api/admin/users` | List all users |
| `DELETE /api/admin/users/{user_id}` | Wipe user data |
| `POST /api/admin/users/wipe-by-email` | Wipe user data by email |
| `POST /api/admin/users/purge-inactive` | Purge inactive users |
| `GET /api/admin/voices/all` | List all ElevenLabs voices |
| `PUT /api/admin/voices/allowed` | Save voice allowlist |

## Multi-Tenant Data Isolation

All user data is fully isolated via `user_data.py`. Each authenticated user gets a private directory tree under `data/users/{user_id}/` containing their chats, memory vault, FAISS indexes, notes, settings, profiles, prompts, directives, uploads, and trash.

| Component | Mechanism |
|---|---|
| **Path routing** | `user_data.py`  -  18 validated path helpers, regex-enforced `user_id` |
| **Request scoping** | `contextvars.ContextVar` set by `AuthMiddleware` on each request |
| **Data helpers** | All `_load_*`/`_save_*` helpers accept optional `user_id`, fall back to contextvar |
| **Copy-on-write** | Profiles, prompts, and directives fall back to global templates when no per-user override exists |
| **Per-user instances** | FAISS indexes and VaultStore instances cached per `user_id` |

## Startup Behavior

On startup, the app rebuilds the NotesFAISS index and initializes a lazy `FAISSMemory` singleton for vault-backed semantic search.

## External Service Dependencies

| Service | Purpose | Local Default | Production (Fly.io) |
|---------|---------|---------------|---------------------|
| SearXNG | Web search for `web_search` tool | `http://localhost:3000` | `SEARXNG_URL` env var  ->  `.flycast` |
| openedai-speech | Text-to-speech | `http://localhost:5050` | `TTS_URL` env var  ->  `.flycast` |
| faster-whisper | Speech-to-text | `http://localhost:8060` | `WHISPER_URL` env var  ->  `.flycast` |

Environment variables (`TTS_URL`, `WHISPER_URL`, `SEARXNG_URL`) take priority over `connections.json` entries.
