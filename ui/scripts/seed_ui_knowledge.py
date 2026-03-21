"""Seed the memory vault with knowledge about the Orion Forge UI.

These memories teach agents how the web dashboard works — pages, APIs,
navigation, tools, chat flow, memory vault, knowledge system, etc.
No PII is included.

Run once from the orion-ui-standalone directory:
    python scripts/seed_ui_knowledge.py
"""

import sys, os

# Ensure repo root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.memory.vault import VaultStore

vault = VaultStore("data/memory/vault.jsonl")

MEMORIES = [
    # ── UI Overview ──────────────────────────────────────────────
    dict(
        text=(
            "The Orion Forge web dashboard is a FastAPI/Jinja2 application served "
            "by Uvicorn. The sidebar navigation includes: Chat, Profiles, Memory Vault, "
            "Knowledge, Tools, AGI Loop, Settings, and About. The root URL (/) redirects "
            "to /chat."
        ),
        scope="shared", category="meta",
        tags=["ui", "dashboard", "navigation", "overview"],
        source="operator", tier="canon",
    ),
    dict(
        text=(
            "The UI runs on port 8989 by default using uvicorn with --reload. "
            "Static assets are served from web/static/ (CSS) and templates live in "
            "web/templates/. All backend logic is in web/app.py."
        ),
        scope="shared", category="meta",
        tags=["ui", "server", "port", "uvicorn", "fastapi"],
        source="operator", tier="canon",
    ),

    # ── Chat Page ────────────────────────────────────────────────
    dict(
        text=(
            "The Chat page (/chat) is the primary interface. It shows a sidebar with "
            "chat history (conversations grouped by date), an agent selector dropdown, "
            "and a connection/model selector. Users type messages and the agent responds "
            "via the configured LLM API. Each chat is saved as a JSON file in data/chats/."
        ),
        scope="shared", category="capability",
        tags=["ui", "chat", "conversation", "interface"],
        source="operator", tier="canon",
    ),
    dict(
        text=(
            "Chat supports two modes: 'chat' (single turn) and 'burst' (multi-tick "
            "autonomous loops). In burst mode the agent runs multiple ticks with tool calls "
            "before returning a final response. The burst tick count is configurable."
        ),
        scope="shared", category="capability",
        tags=["ui", "chat", "burst", "mode", "autonomous"],
        source="operator", tier="canon",
    ),
    dict(
        text=(
            "When the agent responds in chat, the UI shows token usage (prompt + completion), "
            "estimated cost, the model used, and any tool calls that were executed. Tool calls "
            "appear as expandable cards showing the tool name, arguments, and result."
        ),
        scope="shared", category="capability",
        tags=["ui", "chat", "tokens", "cost", "tool_calls"],
        source="operator", tier="canon",
    ),
    dict(
        text=(
            "Chat auto-titles conversations after the second message by asking the LLM "
            "to generate a short title. Chats can be renamed, deleted, or moved into folders. "
            "The chat index is stored in data/chats/index.json."
        ),
        scope="shared", category="capability",
        tags=["ui", "chat", "titles", "folders", "history"],
        source="operator", tier="canon",
    ),

    # ── Prompt Injection Order ───────────────────────────────────
    dict(
        text=(
            "The prompt injection order for each chat request is: "
            "(1) Base system prompt from prompts/{agent}.system.md, "
            "(2) Soul Script — FAISS semantic retrieval from directive-mode knowledge notes, "
            "(3) Always-on Knowledge — verbatim text from always-mode attached notes, "
            "(4) Memory Vault — FAISS semantic search over the agent's memories in vault.jsonl, "
            "(5) Conversation history — recent user/assistant messages truncated to token budget."
        ),
        scope="shared", category="meta",
        tags=["ui", "prompt", "injection", "soul_script", "memory", "knowledge"],
        source="operator", tier="canon",
    ),

    # ── Profiles Page ────────────────────────────────────────────
    dict(
        text=(
            "The Profiles page (/profiles) manages agent identities. Each agent has: "
            "a YAML profile (profiles/{name}.yaml) with personality, tone, traits; "
            "a system prompt (prompts/{name}.system.md); display name and description; "
            "a preferred model; and avatar. Agents can be created, edited, or deleted."
        ),
        scope="shared", category="capability",
        tags=["ui", "profiles", "agents", "identity", "yaml"],
        source="operator", tier="canon",
    ),
    dict(
        text=(
            "On the Profiles page, each agent card shows the agent's display name, "
            "description, model, temperature, and avatar. Expanding a card reveals "
            "tabs for: Profile YAML, System Prompt, Model Config, Knowledge attachments, "
            "and Tool Toggles."
        ),
        scope="shared", category="capability",
        tags=["ui", "profiles", "agent_card", "configuration"],
        source="operator", tier="canon",
    ),
    dict(
        text=(
            "The Profiles page also has a User Profile section where the operator can "
            "set their display name, avatar, and background for the chat interface. "
            "There is also a 'Create Agent' button to add new agents from scratch."
        ),
        scope="shared", category="capability",
        tags=["ui", "profiles", "user_profile", "create_agent"],
        source="operator", tier="canon",
    ),
    dict(
        text=(
            "Knowledge notes can be attached to agents in two modes: "
            "'always' (verbatim injected into every prompt) or 'directive' "
            "(semantically chunked and retrieved via FAISS Soul Script). "
            "This is configured per-agent on the Profiles page under the Knowledge tab."
        ),
        scope="shared", category="capability",
        tags=["ui", "profiles", "knowledge", "directive", "always", "injection"],
        source="operator", tier="canon",
    ),
    dict(
        text=(
            "Each agent can have individual tools enabled or disabled via the Tool Toggles "
            "section on the Profiles page. This controls which tools are included in the "
            "LLM tool definitions for that specific agent."
        ),
        scope="shared", category="capability",
        tags=["ui", "profiles", "tools", "toggle", "per_agent"],
        source="operator", tier="canon",
    ),

    # ── Memory Vault Page ────────────────────────────────────────
    dict(
        text=(
            "The Memory Vault page (/vault) displays all memories stored in the "
            "FAISS-backed vault (data/memory/vault.jsonl). Memories can be filtered "
            "by scope (shared, or agent-specific) and by category (bio, preference, "
            "project, constraint, capability, meta, etc.)."
        ),
        scope="shared", category="capability",
        tags=["ui", "vault", "memory", "filter", "scope", "category"],
        source="operator", tier="canon",
    ),
    dict(
        text=(
            "The Vault page has a search bar that performs FAISS semantic search "
            "over all memories. It also shows stats: total active memories, storage "
            "utilization, scope breakdown, raw JSONL lines, and compaction ratio."
        ),
        scope="shared", category="capability",
        tags=["ui", "vault", "search", "faiss", "stats"],
        source="operator", tier="canon",
    ),
    dict(
        text=(
            "New memories can be added manually via the 'Add Memory' form on the Vault "
            "page. Fields: text, scope, category, tags (comma-separated), source, and tier "
            "(canon or register). There is also a Compact button to rebuild the FAISS index."
        ),
        scope="shared", category="capability",
        tags=["ui", "vault", "add_memory", "compact", "manual"],
        source="operator", tier="canon",
    ),
    dict(
        text=(
            "Memory tiers: 'canon' = durable invariants (identity, mission, stable facts) — "
            "always high-priority in injection. 'register' = mutable state (current projects, "
            "evolving preferences) — one record per topic_id, updated in place. 'log' tier "
            "is rejected by the write-gate and never stored in the vault."
        ),
        scope="shared", category="meta",
        tags=["ui", "vault", "tiers", "canon", "register", "log"],
        source="operator", tier="canon",
    ),
    dict(
        text=(
            "The vault API endpoints: POST /api/vault/add (single memory), "
            "POST /api/vault/batch_add (multiple memories), POST /api/vault/delete "
            "(delete by IDs), GET /api/vault/stats (utilization stats), and "
            "GET /api/vault/compact (rebuild FAISS index)."
        ),
        scope="shared", category="capability",
        tags=["ui", "vault", "api", "endpoints"],
        source="operator", tier="canon",
    ),

    # ── Knowledge Page ───────────────────────────────────────────
    dict(
        text=(
            "The Knowledge page (/knowledge) is a rich-text note system. Notes have "
            "a title, emoji icon, section/category, and HTML content. They are stored "
            "as JSON files in data/user_notes/. Notes can be attached to agents as "
            "contextual knowledge (always-on or directive-mode FAISS retrieval)."
        ),
        scope="shared", category="capability",
        tags=["ui", "knowledge", "notes", "rich_text", "sections"],
        source="operator", tier="canon",
    ),
    dict(
        text=(
            "Knowledge notes support full CRUD: create, read, update, and soft-delete "
            "(trash). The edit page (/knowledge/{note_id}/edit) provides a rich text "
            "editor. Notes are listed by section on the main Knowledge page."
        ),
        scope="shared", category="capability",
        tags=["ui", "knowledge", "crud", "editor", "trash"],
        source="operator", tier="canon",
    ),

    # ── Tools Page ───────────────────────────────────────────────
    dict(
        text=(
            "The Tools page (/tools) shows all registered and planned tools as cards. "
            "Each card displays the tool name, icon, status (ready, planned, or concept), "
            "description, and parameters. Tools with 'ready' status are functional; "
            "'planned' and 'concept' are future roadmap items."
        ),
        scope="shared", category="capability",
        tags=["ui", "tools", "catalogue", "status", "cards"],
        source="operator", tier="canon",
    ),
    dict(
        text=(
            "Current ready tools: web_search (SearXNG-powered web search with fast/normal/deep "
            "modes), email (SMTP-based with account management), echo (test tool), and "
            "cost_tracker (token pricing and spend monitoring). Planned tools: memory, "
            "directives, task_inbox, continuation_update, runtime_info, inbox. Concept: computer_use."
        ),
        scope="shared", category="capability",
        tags=["ui", "tools", "web_search", "email", "cost_tracker", "ready", "planned"],
        source="operator", tier="canon",
    ),
    dict(
        text=(
            "The Tools page includes configuration panels for each ready tool. "
            "Web Search has SearXNG URL and mode settings. Email has SMTP account "
            "management (add/remove accounts, test send). Cost Tracker shows spending "
            "breakdowns by today/month/all-time with per-model rates."
        ),
        scope="shared", category="capability",
        tags=["ui", "tools", "config", "web_search", "email", "cost_tracker"],
        source="operator", tier="canon",
    ),
    dict(
        text=(
            "The Tools page also has a Memory Tool panel showing FAISS stats (total "
            "memories, vector count, dimensions, embedding model, sync status), the "
            "Identity Profile editor (FAISS write/search/injection settings), and the "
            "Memory Profile editor (tier config, scopes, write-gate rules)."
        ),
        scope="shared", category="capability",
        tags=["ui", "tools", "memory_tool", "faiss", "identity_profile", "memory_profile"],
        source="operator", tier="canon",
    ),
    dict(
        text=(
            "The Memory Tool Identity Profile config includes sections: embedding model, "
            "search top-k, injection budget (max tokens), write-gate rules (which tiers "
            "can be written), and saved profile presets that can be loaded or saved."
        ),
        scope="shared", category="capability",
        tags=["ui", "tools", "identity_profile", "embedding", "write_gate", "presets"],
        source="operator", tier="canon",
    ),

    # ── Settings Page ────────────────────────────────────────────
    dict(
        text=(
            "The Settings page (/settings) manages LLM API connections. Each connection "
            "has: a name, provider (openai, anthropic, deepseek, ollama, etc.), API URL, "
            "API key, enabled/disabled toggle, and a list of available models."
        ),
        scope="shared", category="capability",
        tags=["ui", "settings", "connections", "api", "provider"],
        source="operator", tier="canon",
    ),
    dict(
        text=(
            "Connections are stored in config/connections.json. The Settings page lets "
            "the operator add, edit, delete, and reorder connections. Agent-specific "
            "connection bindings can map each agent to a preferred connection."
        ),
        scope="shared", category="capability",
        tags=["ui", "settings", "connections", "agent_binding"],
        source="operator", tier="canon",
    ),

    # ── AGI Loop Page ────────────────────────────────────────────
    dict(
        text=(
            "The AGI Loop page (/agi-loop) configures autonomous agent operation. "
            "Settings include: interval (hours/minutes between loops), max loops, "
            "ticks per loop, max steps per tick, budget caps (monthly hard/soft cap, "
            "per-session cap, per-tick cap), and auto-pause conditions."
        ),
        scope="shared", category="capability",
        tags=["ui", "agi_loop", "autonomous", "config", "budget"],
        source="operator", tier="canon",
    ),
    dict(
        text=(
            "The AGI Loop page also has a Model Router section. The model router defines "
            "four tiers: local_cheap (e.g. qwen2.5:7b for quick tasks), local_strong "
            "(e.g. llama3:70b for reasoning), cheap_cloud (e.g. deepseek-chat), and "
            "expensive_cloud (e.g. gpt-4o for final polish). Tasks are mapped to tiers "
            "via a task_tier_map (coding, summarization, planning, etc.)."
        ),
        scope="shared", category="capability",
        tags=["ui", "agi_loop", "model_router", "tiers", "task_map"],
        source="operator", tier="canon",
    ),

    # ── About Page ───────────────────────────────────────────────
    dict(
        text=(
            "The About page (/about) displays editable 'about' text stored in "
            "config/about.json. It serves as a project description or personal "
            "statement visible to anyone with dashboard access."
        ),
        scope="shared", category="meta",
        tags=["ui", "about", "description"],
        source="operator", tier="canon",
    ),

    # ── Tool Execution Flow ──────────────────────────────────────
    dict(
        text=(
            "Tool execution in chat: when an LLM response includes tool_calls, the backend "
            "loops up to 10 rounds. Each round: the LLM's tool_call message is appended to "
            "context, each tool is executed via src/tools/registry.execute_tool(), results "
            "are appended as role='tool' messages, and the LLM is called again to either "
            "make more tool calls or produce a final text response."
        ),
        scope="shared", category="capability",
        tags=["ui", "chat", "tool_loop", "execution", "registry"],
        source="operator", tier="canon",
    ),
    dict(
        text=(
            "Every LLM API call in chat is metered: token usage (prompt + completion) is "
            "accumulated across tool-call rounds, and cost is computed via "
            "src/observability/metering and logged to a cost log. The total cost for "
            "each message is shown in the chat UI."
        ),
        scope="shared", category="capability",
        tags=["ui", "chat", "metering", "cost", "tokens", "observability"],
        source="operator", tier="canon",
    ),

    # ── Memory Auto-Save in Chat ─────────────────────────────────
    dict(
        text=(
            "Agents can save memories during chat by including [MEMORY_SAVE: ...] tags "
            "in their responses. The backend extracts these tags, saves each as a vault "
            "memory, and strips the tags from the displayed response. This allows agents "
            "to autonomously build their memory over time."
        ),
        scope="shared", category="capability",
        tags=["ui", "chat", "memory_save", "auto_save", "tags"],
        source="operator", tier="canon",
    ),

    # ── File & Data Layout ───────────────────────────────────────
    dict(
        text=(
            "UI data layout: config/ holds connections.json, settings.json, pricing.yaml, "
            "about.json, agi_loop.json, identity_profile.json, memory_profile.json, and "
            "saved profile presets. data/ holds chats/, memory/ (vault.jsonl + faiss/), "
            "user_notes/, uploads/, and shared/."
        ),
        scope="shared", category="meta",
        tags=["ui", "data", "layout", "config", "files"],
        source="operator", tier="canon",
    ),
    dict(
        text=(
            "Agent profiles live in profiles/{name}.yaml. System prompts live in "
            "prompts/{name}.system.md. Directives live in directives/{name}.md. "
            "Notes (knowledge) live in notes/{name}.md for builtins and "
            "data/user_notes/{id}.json for user-created notes."
        ),
        scope="shared", category="meta",
        tags=["ui", "files", "profiles", "prompts", "directives", "notes"],
        source="operator", tier="canon",
    ),

    # ── Chat API Endpoints ───────────────────────────────────────
    dict(
        text=(
            "Chat API endpoints: POST /api/chat/send (send message + get response), "
            "GET /api/chat/history (list all chats), POST /api/chat/new (create empty chat), "
            "GET /api/chat/{id} (load chat), DELETE /api/chat/{id} (delete), "
            "PUT /api/chat/{id} (update title/folder), POST /api/chat/{id}/title (auto-title)."
        ),
        scope="shared", category="capability",
        tags=["ui", "api", "chat", "endpoints"],
        source="operator", tier="canon",
    ),

    # ── Profiles API Endpoints ───────────────────────────────────
    dict(
        text=(
            "Profile API endpoints: GET /api/profiles/{name} (load profile YAML), "
            "PUT /api/profiles/{name} (save profile YAML), POST /api/profiles (create), "
            "DELETE /api/profiles/{name} (delete), PUT /api/profiles/{name}/config "
            "(save agent settings like model, display name), "
            "PUT /api/profiles/{name}/avatar (upload avatar image)."
        ),
        scope="shared", category="capability",
        tags=["ui", "api", "profiles", "endpoints"],
        source="operator", tier="canon",
    ),

    # ── Knowledge API Endpoints ──────────────────────────────────
    dict(
        text=(
            "Knowledge API endpoints: POST /api/knowledge (create note), "
            "PUT /api/knowledge/{id} (update note), DELETE /api/knowledge/{id} "
            "(soft-delete / trash), GET /api/knowledge/{id} (load note). "
            "Notes are stored as JSON in data/user_notes/."
        ),
        scope="shared", category="capability",
        tags=["ui", "api", "knowledge", "endpoints"],
        source="operator", tier="canon",
    ),

    # ── Pricing / Cost System ────────────────────────────────────
    dict(
        text=(
            "Token pricing is configured in config/pricing.yaml with per-model input and "
            "output rates (dollars per 1M tokens). The cost tracker computes cost for each "
            "LLM call and aggregates spending by today, this month, and all-time. "
            "The old /pricing page now redirects to /tools#cost_tracker."
        ),
        scope="shared", category="capability",
        tags=["ui", "pricing", "cost_tracker", "tokens", "rates"],
        source="operator", tier="canon",
    ),

    # ── FAISS Memory Architecture ────────────────────────────────
    dict(
        text=(
            "The memory system has two layers: VaultStore (append-only JSONL backend in "
            "data/memory/vault.jsonl) and FAISSMemory (vector search + embeddings on top "
            "of VaultStore). VaultStore handles persistence, versioning, and compaction. "
            "FAISSMemory handles semantic search, embedding generation, and index management."
        ),
        scope="shared", category="capability",
        tags=["ui", "memory", "vault", "faiss", "architecture", "layers"],
        source="operator", tier="canon",
    ),
    dict(
        text=(
            "The vault is append-only: adds, updates, and deletes all append a new JSONL "
            "line. The file is never edited in place. On read, each id resolves to its "
            "highest-version line. Compaction rewrites the file to remove superseded lines."
        ),
        scope="shared", category="capability",
        tags=["ui", "memory", "vault", "append_only", "versioning", "compaction"],
        source="operator", tier="canon",
    ),

    # ── NotesFAISS / Soul Script ─────────────────────────────────
    dict(
        text=(
            "NotesFAISS is a separate FAISS index built from knowledge notes marked as "
            "'directive' mode. At startup and after knowledge changes, notes are "
            "semantically chunked (split on ### headers, overlapping fallback) and indexed. "
            "During chat, relevant chunks are injected as 'Soul Script' — the second layer "
            "of the prompt after the base system prompt."
        ),
        scope="shared", category="capability",
        tags=["ui", "notes_faiss", "soul_script", "directive", "semantic_chunking"],
        source="operator", tier="canon",
    ),

    # ── PII Guard ────────────────────────────────────────────────
    dict(
        text=(
            "The memory PII guard (src/memory/pii_guard.py) blocks storage of sensitive data. "
            "It rejects: SSNs, credit card numbers, passwords, API keys, auth tokens, and "
            "secret keys. The guard runs on every vault write and raises ValueError if PII "
            "is detected."
        ),
        scope="shared", category="constraint",
        tags=["ui", "memory", "pii", "guard", "security"],
        source="operator", tier="canon",
    ),
]


def main():
    count = 0
    for m in MEMORIES:
        mem = vault.create_memory(**m)
        print(f"  + [{mem.tier:8s}] {mem.id}: {mem.text[:80]}...")
        count += 1
    print(f"\nDone — {count} memories seeded into vault.")


if __name__ == "__main__":
    main()
