# Data — Runtime Data Directory

This directory holds all runtime-generated data for the production OrionForge deployment. Subdirectories are pre-populated with placeholder files to maintain the directory structure.

## Directory Layout

```
data/
├── chats/              # Chat session histories (JSON per session)
├── memory/
│   └── faiss/          # FAISS vector index files
├── orion/              # Agent-specific working data
├── shared/             # Cross-agent shared data (inbox, etc.)
├── test_agent_xx/      # Test agent data directory
├── uploads/            # User-uploaded files
└── user_notes/         # Knowledge notes (JSON, rich text)
```

## Populating Data

All directories start empty (with placeholder files). Data is created during normal operation:

- **Chats** are created when conversations begin via the web UI
- **Memories** are written to `memory/vault.jsonl` by the Memory tool
- **FAISS indexes** are built automatically from vault entries
- **User notes** are created through the Knowledge editor in the web UI
- **Uploads** are saved when users set chat backgrounds or attach files

## Backup

To back up all agent data, copy this entire `data/` directory. The FAISS index can be rebuilt from `vault.jsonl` at any time.
