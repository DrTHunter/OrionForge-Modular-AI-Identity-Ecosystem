# scripts/

Utility scripts for data seeding and maintenance. Run from the project root.

## Files

| File | Purpose |
|------|---------|
| `seed_memories.py` | Seeds the memory vault with test/example memories |
| `seed_ui_knowledge.py` | Seeds knowledge notes with UI documentation and help content |

## seed_memories.py

Populates `data/memory/vault.jsonl` with sample memories for testing and bootstrapping.

### What's Seeded

Example data includes:
- Computer hardware specs (workstation, laptop, NAS, networking)
- Project state and priorities
- Bio facts and user preferences
- Example canon and register tier memories

### Usage

```bash
python scripts/seed_memories.py
```

Uses `VaultStore` from `src/memory/vault.py` to write properly formatted entries with full validation (PII guard, duplicate detection, write-gate).

## seed_ui_knowledge.py

Seeds the knowledge notes system (`data/user_notes/`) with structured documentation about the OrionForge UI — tool descriptions, feature guides, and help content that agents can reference via FAISS retrieval.

### Usage

```bash
python scripts/seed_ui_knowledge.py
```

Creates JSON knowledge note files with proper metadata for the knowledge editor and FAISS indexing pipeline.
