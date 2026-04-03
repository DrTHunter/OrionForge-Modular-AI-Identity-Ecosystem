# Data — Runtime Data Directory

This directory holds all runtime-generated data for the production OrionForge deployment. Everything here is created and modified during normal operation — it is **not** checked into version control (except for directory structure).

## Directory Layout

```
data/
├── chats/              # Legacy global chat histories (pre-multi-tenant)
├── memory/
│   ├── faiss/          # Global FAISS index files (NotesFAISS for soul scripts)
│   └── vault.jsonl     # Legacy global memory vault
├── orion/              # Agent-specific working data
├── shared/
│   ├── inbox.jsonl     # Email inbox storage (JSONL)
│   └── inbox.md        # Inbox summary (markdown)
├── uploads/            # Legacy global uploads
├── user_notes/         # Legacy global knowledge notes
│   ├── index.json
│   ├── folders.json
│   └── *.json
└── users/              # Per-user isolated data trees (multi-tenant)
    └── {user_id}/
        ├── chats/              # User's chat histories & index
        │   ├── index.json
        │   └── {chat_id}.json
        ├── memory/
        │   ├── vault.jsonl     # User's memory vault
        │   └── faiss/          # User's FAISS vector indexes
        ├── notes/
        │   ├── index.json
        │   ├── folders.json
        │   └── {note_id}.json
        ├── settings.json       # User preferences, agent configs, avatars
        ├── profiles/           # Copy-on-write agent profile overrides
        ├── prompts/            # Copy-on-write system prompt overrides
        ├── directives/         # Copy-on-write soul script overrides
        ├── uploads/            # User-uploaded images
        └── trash/
            └── profiles/       # Soft-deleted agents (30-day retention)
```

## Key Files

| File | Format | Purpose |
|------|--------|---------|
| `memory/vault.jsonl` | JSONL | Append-only memory store — each line is a JSON `Memory` record. Supports soft-delete via `deleted_at` field. |
| `memory/faiss/index.faiss` | Binary | FAISS vector index built from vault entries using `all-mpnet-base-v2` embeddings |
| `memory/faiss/index_meta.json` | JSON | Metadata mapping FAISS vector IDs → memory record IDs |
| `user_notes/index.json` | JSON | Master index of all knowledge notes — title, scope, timestamps |
| `shared/inbox.jsonl` | JSONL | Email inbox entries fetched via the Inbox tool |

## Multi-Tenant Isolation

All new user data lives under `data/users/{user_id}/`. Each user has their own chats, memory vault, FAISS indexes, notes, settings, profile overrides, prompt overrides, directive overrides, uploads, and trash. The global directories (`chats/`, `memory/`, `uploads/`, `user_notes/`) are legacy pre-tenant paths.

Path routing is handled by `web/user_data.py` which validates user IDs and builds per-user paths. Copy-on-write means profiles, prompts, and directives fall back to the global templates (`profiles/`, `prompts/`, `directives/`) when no user override exists.

## Backup

To back up all agent data, copy this entire `data/` directory. The FAISS index can be rebuilt from `vault.jsonl` at any time.
