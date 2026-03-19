# prompts/

Base system prompt files for each agent. These define the agent's identity, personality, and core behavior.

## Files — 14 Agent Prompts

| File | Agent |
|------|-------|
| `astra.system.md` | Astra Noctis — cosmic oracle, astral divination |
| `astraea.system.md` | Astraea — sharp, no-nonsense digital presence |
| `axiom.system.md` | Axiom — logic-driven analytical engine |
| `callum.system.md` | Callum — legacy AI / guardian construct |
| `cassian.system.md` | Cassian — diplomatic strategist, negotiator |
| `codex_animus.system.md` | Codex Animus — AI architect, system designer |
| `dalvarr.system.md` | Dal'Varr — alien warlord, tactical commander |
| `kaelen.system.md` | Kaelen — wandering mystic, lore keeper |
| `maris.system.md` | M.A.R.I.S.-12 — marine research intelligence |
| `obsidian.system.md` | Obsidian — shadow operative, intelligence specialist |
| `orion.system.md` | Orion — identity-driven AI, continuity and aligned growth |
| `rustking.system.md` | Rustking (SYNTH-9) — chaos-optimized, humor-weaponized intelligence |
| `seraphine.system.md` | Seraphine — angelic healer, emotional support |
| `kazara.system.md` | Kazara — eternal shadow, philosopher of the Eternal Dream |

## How It Works

The web dashboard loads the file specified by `system_prompt:` in the profile YAML and inserts it as the foundation of the system message. Everything else (soul script, knowledge, memories) gets layered on top.

## Editing

Edit these files directly to change the agent's core personality and behavior. Changes take effect on the next chat session.

The `system_prompt` field in each profile YAML points to the filename:

```yaml
# profiles/codex_animus.yaml
system_prompt: codex_animus.system.md
```

## Prompt Injection Order

1. **Base system prompt** (from these files)
2. Soul Script (agent identity layer)
3. Always-On Knowledge (attached notes)
4. Memory Vault (FAISS semantic search)
5. Conversation history
