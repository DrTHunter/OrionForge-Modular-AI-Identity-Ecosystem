# Data — Runtime Data Directory

This directory holds all runtime-generated data for the OrionForge UI. Everything here is created and modified during normal operation — it is **not** checked into version control (except for directory structure).

## Directory Layout

```
data/
├── chats/              # Chat session histories (JSON per session)
├── memory/
│   ├── faiss/          # FAISS index files (index.faiss, index_meta.json)
│   └── vault.jsonl     # Memory vault — append-only JSONL of all Memory records
├── orion/              # Agent-specific working data
├── shared/
│   ├── inbox.jsonl     # Email inbox storage (JSONL)
│   └── inbox.md        # Inbox summary (markdown)
├── uploads/            # User-uploaded files (chat backgrounds, attachments)
└── user_notes/
    ├── index.json      # Note index — maps note IDs to metadata
    ├── folders.json    # Folder structure for the knowledge UI
    └── *.json          # Individual note files (rich text, HTML content)
```

## Key Files

| File | Format | Purpose |
|------|--------|---------|
| `memory/vault.jsonl` | JSONL | Append-only memory store — each line is a JSON `Memory` record. Supports soft-delete via `deleted_at` field. |
| `memory/faiss/index.faiss` | Binary | FAISS vector index built from vault entries using `all-mpnet-base-v2` embeddings |
| `memory/faiss/index_meta.json` | JSON | Metadata mapping FAISS vector IDs → memory record IDs |
| `user_notes/index.json` | JSON | Master index of all knowledge notes — title, scope, timestamps |
| `shared/inbox.jsonl` | JSONL | Email inbox entries fetched via the Inbox tool |

## Backup

To back up all agent data, copy this entire `data/` directory. The `memory/faiss/` index can be rebuilt from `vault.jsonl` at any time.
