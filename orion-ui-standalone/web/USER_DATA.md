# Multi-Tenant Data Isolation — `user_data.py`

Per-user data isolation layer for the OrionForge platform. Every authenticated user gets a fully isolated directory tree. No data is shared between users except global read-only agent templates.

## Architecture

| Component | Mechanism |
|---|---|
| **Path routing** | 18 validated path helper functions in `user_data.py` |
| **User ID validation** | Regex `^[a-zA-Z0-9_-]{1,128}$` — blocks path traversal, null bytes, slashes, oversized IDs |
| **Request scoping** | `contextvars.ContextVar` (`_current_user_id`) set by `AuthMiddleware` on each request |
| **Data helpers** | All `_load_*` / `_save_*` helpers in `app.py` accept optional `user_id`, fall back to the contextvar |
| **Copy-on-write** | Profiles, system prompts, and soul scripts fall back to global templates when no user override exists |
| **Per-user instances** | FAISS indexes and VaultStore instances cached per `user_id` — no cross-user contamination |
| **Directory creation** | `ensure_user_dirs(user_id)` called by `AuthMiddleware` on every authenticated request |
| **Admin wipe** | `DELETE /api/admin/users/{uid}` removes the entire user directory tree |

## Per-User Directory Layout

```
data/users/{user_id}/
├── chats/
│   ├── index.json          # Chat metadata index
│   └── {chat_id}.json      # Individual chat histories
├── memory/
│   ├── vault.jsonl          # User's memory vault (append-only JSONL)
│   └── faiss/               # User's FAISS vector indexes
├── notes/
│   ├── index.json           # Knowledge notes index
│   ├── folders.json         # Knowledge folder structure
│   └── {note_id}.json       # Individual notes
├── settings.json            # Preferences, agent configs, avatars, backgrounds
├── profiles/                # Copy-on-write agent profile overrides (YAML)
├── prompts/                 # Copy-on-write system prompt overrides (*.system.md)
├── directives/              # Copy-on-write soul script overrides (*.md)
├── uploads/                 # User-uploaded images (avatars, backgrounds)
└── trash/
    └── profiles/            # Soft-deleted agents (30-day retention)
```

## Path Helper Functions

| Function | Returns |
|----------|---------|
| `_validate_user_id(uid)` | Sanitized `str` or raises `ValueError` |
| `user_root(uid)` | `Path` — `data/users/{uid}/` (auto-creates) |
| `user_chats_dir(uid)` | `Path` — `data/users/{uid}/chats/` |
| `user_memory_dir(uid)` | `Path` — `data/users/{uid}/memory/` |
| `user_faiss_dir(uid)` | `Path` — `data/users/{uid}/memory/faiss/` |
| `user_vault_path(uid)` | `Path` — `data/users/{uid}/memory/vault.jsonl` |
| `user_notes_dir(uid)` | `Path` — `data/users/{uid}/notes/` |
| `user_settings_path(uid)` | `Path` — `data/users/{uid}/settings.json` |
| `user_profiles_dir(uid)` | `Path` — `data/users/{uid}/profiles/` |
| `user_prompts_dir(uid)` | `Path` — `data/users/{uid}/prompts/` |
| `user_directives_dir(uid)` | `Path` — `data/users/{uid}/directives/` |
| `user_uploads_dir(uid)` | `Path` — `data/users/{uid}/uploads/` |
| `user_trash_dir(uid)` | `Path` — `data/users/{uid}/trash/profiles/` |
| `user_folders_file(uid)` | `Path` — `data/users/{uid}/notes/folders.json` |
| `ensure_user_dirs(uid)` | `Path` — Creates all subdirectories, returns root |

## What's Isolated per User

- Chat histories & index
- Memory vault (JSONL) & FAISS vector indexes
- Knowledge notes & folders
- Settings (preferences, agent configs, avatars, backgrounds)
- Agent profile overrides (copy-on-write from global templates)
- System prompt overrides
- Soul script / directive overrides
- Uploaded images (avatars, backgrounds)
- Trash (soft-deleted agents with 30-day retention)

## What's Shared (Read-Only Global Templates)

| Directory | Content |
|-----------|---------|
| `profiles/*.yaml` | Default agent profiles (model, parameters, provider) |
| `prompts/*.system.md` | Default system prompts |
| `directives/*.md` | Default soul scripts |

When a user customizes an agent, the override is saved in their user directory. Reads check the user directory first, then fall back to the global template.

## Copy-on-Write Flow

```
_load_profile("orion", user_id="abc123")
  1. Check: data/users/abc123/profiles/orion.yaml → exists? return it
  2. Fallback: profiles/orion.yaml → return global template
  3. Not found: return empty dict

_save_profile("orion", data, user_id="abc123")
  → Writes to: data/users/abc123/profiles/orion.yaml
  → Global profiles/orion.yaml is never modified
```

## Security

- Path traversal attacks blocked: `../`, `\`, `/`, null bytes, newlines, HTML tags all rejected
- User ID max length: 128 characters
- User ID charset: `[a-zA-Z0-9_-]` only
- No user can access another user's directory tree
- Admin wipe (`shutil.rmtree`) removes entire user directory tree

## Test Coverage

The multi-tenant system is covered by `tests/test_multi_tenant.py` — 16 test functions, 181 checks including:
- All path helper functions
- 10+ path traversal attack vectors
- Two-user directory isolation
- Per-user chat, settings, knowledge, vault, uploads, trash isolation
- Profile/prompt/directive copy-on-write
- VaultStore per-user instances
- 5-user × 20-chat stress test
- 20-user massive isolation stress with zero-bleed verification
- Admin wipe cleanup
