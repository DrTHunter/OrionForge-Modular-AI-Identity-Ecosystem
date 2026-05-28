# Storage  -  Note Collection & User Notes

> Status: reviewed and refreshed on 2026-05-28.

Handles loading, stripping, and injecting user-authored notes into agent context.

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `note_collector.py` | ~204 | Loads per-agent notes respecting always-on vs directive modes. Reads `config/settings.json` for note attachment and mode settings. |
| `user_notes_loader.py` | ~97 | Loads JSON user notes from `data/user_notes/`. Strips HTML tags, decodes entities, extracts plain text for prompt injection. |

## Two Note Systems

OrionForge maintains two parallel note systems for different purposes:

| System | Backing | Mutability | Use Case |
|--------|---------|------------|----------|
| **NotesFAISS** | `data/user_notes/*.json` | Immutable at runtime | Soul scripts, directive notes, knowledge base articles |
| **FAISSMemory** | `data/memory/vault.jsonl` | Mutable (agent-written) | Working memory, conversation recall, observation logs |

## Data Flow

```
data/user_notes/*.json
        │
        ▼
  user_notes_loader.py   ← strips HTML, extracts text
        │
        ▼
  note_collector.py      ← filters by agent, mode, directive context
        │
        ▼
  System prompt injection
```

## Settings

Note attachment behaviour is controlled via `config/settings.json`:

- **always**  -  notes are injected into every prompt
- **directive**  -  notes are only injected when referenced by active directives
