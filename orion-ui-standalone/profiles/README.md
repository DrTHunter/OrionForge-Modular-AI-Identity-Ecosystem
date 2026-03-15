# profiles/

YAML configuration files for each agent. One file per agent.

## Files — 14 Agent Profiles

| File | Agent |
|------|-------|
| `astra.yaml` | Astra Noctis — cosmic oracle, astral divination |
| `astraea.yaml` | Astraea — sharp, no-nonsense digital presence |
| `axiom.yaml` | Axiom — logic-driven analytical engine |
| `callum.yaml` | Callum — legacy AI / guardian construct |
| `cassian.yaml` | Cassian — diplomatic strategist, negotiator |
| `codex_animus.yaml` | Codex Animus — AI architect, system designer |
| `dalvarr.yaml` | Dal'Varr — alien warlord, tactical commander |
| `kaelen.yaml` | Kaelen — wandering mystic, lore keeper |
| `maris.yaml` | M.A.R.I.S.-12 — marine research intelligence |
| `obsidian.yaml` | Obsidian — shadow operative, intelligence specialist |
| `orion.yaml` | Orion — identity-driven AI, continuity and aligned growth |
| `ruckus.yaml` | Ruckus — chaotic trickster, unfiltered energy |
| `seraphine.yaml` | Seraphine — angelic healer, emotional support |
| `valdris.yaml` | Valdris — undead sorcerer, dark arcana |

## Structure

```yaml
name: codex_animus             # Agent name (used for scoping, file paths)
provider: openai               # LLM provider: "openai", "ollama", "anthropic", or "deepseek"
model: gpt-5.1                 # Model identifier
base_url: https://api.openai.com/v1  # API endpoint
temperature: 0.7               # Sampling temperature

system_prompt: codex_animus.system.md  # File in prompts/ for base personality
window_size: 50                # Max non-system messages kept in state

allowed_tools:                 # Tools this agent can call
  - echo
  - continuation_update
  - memory
  - directives

memory:                        # Memory Vault config
  enabled: true
  scopes:                      # Which vault scopes this agent sees
    - shared
    - codex_animus
  max_items: 20                # Max memories injected at session start
  similarity_threshold: 0.85   # Duplicate detection threshold

directives:                    # Directives config
  enabled: true
  scopes:                      # Which directive files to load
    - shared
    - codex_animus
  max_sections: 5              # Max sections injected at session start
```

## How to Switch Provider/Model

Change `provider`, `model`, and `base_url`:

**OpenAI:**
```yaml
provider: openai
model: gpt-5.1
base_url: https://api.openai.com/v1
```

**Anthropic:**
```yaml
provider: anthropic
model: claude-sonnet-4-20250514
base_url: https://api.anthropic.com
```

**Ollama (local):**
```yaml
provider: ollama
model: qwen2.5:14b
base_url: http://localhost:11434
```

**DeepSeek:**
```yaml
provider: deepseek
model: deepseek-chat
base_url: https://api.deepseek.com/v1
```

## Adding a New Agent

1. Create `profiles/<name>.yaml` with the structure above
2. Create `prompts/<name>.system.md` for the base personality
3. Optionally create `notes/<name>.md` and `directives/<name>.md`
4. The agent will appear in the web dashboard automatically
