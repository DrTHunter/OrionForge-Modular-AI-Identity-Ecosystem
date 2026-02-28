"""SoulScript Engine — Clean MVP Web Layer.

A minimal FastAPI application demonstrating AI identity persistence
through FAISS-backed memory retrieval and soul script injection.

Prompt Injection Order
──────────────────────
1. Base Prompt        — Agent's system prompt (prompts/{agent}.system.md)
2. Soul Script        — FAISS semantic retrieval from directive-mode knowledge
3. Always-On Knowledge — Verbatim text from always-mode attached knowledge
4. Memory Vault       — FAISS semantic search over agent memories (vault.jsonl)
5. Conversation       — Recent user/assistant messages (truncated to budget)
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# ── Project paths ────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR   = _PROJECT_ROOT / "config"
_DATA_DIR     = _PROJECT_ROOT / "data"
_PROFILES_DIR = _PROJECT_ROOT / "profiles"
_PROMPTS_DIR  = _PROJECT_ROOT / "prompts"
_CHATS_DIR    = _DATA_DIR / "chats"
_NOTES_DIR    = _DATA_DIR / "user_notes"
_VAULT_PATH   = _DATA_DIR / "memory" / "vault.jsonl"
_FAISS_DIR    = _DATA_DIR / "memory" / "faiss"

CONNECTIONS_FILE = _CONFIG_DIR / "connections.json"
SETTINGS_FILE    = _CONFIG_DIR / "settings.json"
PRICING_FILE     = _CONFIG_DIR / "pricing.yaml"

log = logging.getLogger("soulscript")

# ── Ensure data directories exist ────────────────────────────────
for _d in [_CHATS_DIR, _NOTES_DIR, _VAULT_PATH.parent, _FAISS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── FastAPI app ──────────────────────────────────────────────────
from contextlib import asynccontextmanager

@asynccontextmanager
async def _lifespan(application: FastAPI):
    """Build NotesFAISS on startup so Soul Script retrieval works immediately."""
    try:
        _rebuild_notes_faiss()
        from src.storage.note_collector import invalidate_notes_faiss
        invalidate_notes_faiss()          # force singleton to reload fresh index
    except Exception as exc:
        log.warning("[startup] NotesFAISS build skipped: %s", exc)
    yield

app = FastAPI(title="SoulScript Engine", version="0.2.0", lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# ── FAISS memory (lazy singleton) ────────────────────────────────
_faiss_memory = None

def _get_faiss_memory():
    global _faiss_memory
    if _faiss_memory is None:
        try:
            from src.memory.faiss_memory import FAISSMemory
            _faiss_memory = FAISSMemory(
                vault_path=str(_VAULT_PATH),
                faiss_dir=str(_FAISS_DIR),
            )
            log.info("[vault] FAISSMemory loaded — %d memories", len(_faiss_memory.list_all()))
        except Exception as exc:
            log.error("[vault] FAISSMemory init failed: %s", exc)
    return _faiss_memory


# ═══════════════════════════════════════════════════════════════════
#  JSON HELPERS
# ═══════════════════════════════════════════════════════════════════

def _read_json(path: Path, default=None):
    if not path.exists():
        return default if default is not None else {}
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)

def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ═══════════════════════════════════════════════════════════════════
#  CONNECTIONS
# ═══════════════════════════════════════════════════════════════════

def _load_connections() -> dict:
    return _read_json(CONNECTIONS_FILE, {"connections": [], "agent_connections": {}})

def _save_connections(data: dict):
    _write_json(CONNECTIONS_FILE, data)

def _resolve_connection(connection_id: str | None, agent: str) -> dict | None:
    """Pick the best API connection for a request."""
    store = _load_connections()
    conns = store.get("connections", [])
    agent_map = store.get("agent_connections", {})

    if connection_id:
        return next((c for c in conns if c["id"] == connection_id and c.get("enabled")), None)
    mapped_id = agent_map.get(agent)
    if mapped_id:
        found = next((c for c in conns if c["id"] == mapped_id and c.get("enabled")), None)
        if found:
            return found
    return next((c for c in conns if c.get("enabled") and c.get("type") == "external"), None)


# ═══════════════════════════════════════════════════════════════════
#  SETTINGS
# ═══════════════════════════════════════════════════════════════════

def _load_settings() -> dict:
    return _read_json(SETTINGS_FILE, {})

def _save_settings(data: dict):
    _write_json(SETTINGS_FILE, data)


# ═══════════════════════════════════════════════════════════════════
#  PRICING
# ═══════════════════════════════════════════════════════════════════

def _load_pricing() -> dict:
    if not PRICING_FILE.exists():
        return {}
    with open(PRICING_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _save_pricing(data: dict):
    PRICING_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PRICING_FILE, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# ═══════════════════════════════════════════════════════════════════
#  PROFILES
# ═══════════════════════════════════════════════════════════════════

def _list_agents() -> list[str]:
    return sorted(p.stem for p in _PROFILES_DIR.glob("*.yaml"))

def _load_profile(name: str) -> dict:
    path = _PROFILES_DIR / f"{name}.yaml"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _save_profile(name: str, data: dict):
    with open(_PROFILES_DIR / f"{name}.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

def _load_system_prompt(name: str) -> str:
    path = _PROMPTS_DIR / f"{name}.system.md"
    return path.read_text(encoding="utf-8") if path.exists() else ""

def _save_system_prompt(name: str, text: str):
    (_PROMPTS_DIR / f"{name}.system.md").write_text(text, encoding="utf-8")

def _get_agent_config(agent: str) -> dict:
    return _load_settings().get("agent_configs", {}).get(agent, {})

def _save_agent_config(agent: str, cfg: dict):
    settings = _load_settings()
    settings.setdefault("agent_configs", {})[agent] = cfg
    _save_settings(settings)


# ═══════════════════════════════════════════════════════════════════
#  CHATS
# ═══════════════════════════════════════════════════════════════════

def _load_chat_index() -> dict:
    return _read_json(_CHATS_DIR / "index.json", {"folders": [], "chats": []})

def _save_chat_index(data: dict):
    _write_json(_CHATS_DIR / "index.json", data)

def _load_chat(chat_id: str) -> dict | None:
    path = _CHATS_DIR / f"{chat_id}.json"
    return _read_json(path) if path.exists() else None

def _save_chat(chat_id: str, data: dict):
    _write_json(_CHATS_DIR / f"{chat_id}.json", data)

def _create_new_chat(agent: str) -> dict:
    chat_id = str(uuid.uuid4())[:8]
    now = datetime.now(timezone.utc).isoformat()
    chat_data = {
        "id": chat_id, "title": "New Chat", "folder_id": None,
        "agent": agent, "mode": "chat",
        "created": now, "updated": now, "messages": [],
    }
    _save_chat(chat_id, chat_data)
    idx = _load_chat_index()
    idx["chats"].append({
        "id": chat_id, "title": "New Chat", "folder_id": None,
        "agent": agent, "mode": "chat", "created": now, "updated": now,
    })
    _save_chat_index(idx)
    return chat_data


# ═══════════════════════════════════════════════════════════════════
#  KNOWLEDGE (Notes)
# ═══════════════════════════════════════════════════════════════════

def _load_notes_index() -> list[dict]:
    return _read_json(_NOTES_DIR / "index.json", [])

def _save_notes_index(data: list[dict]):
    _write_json(_NOTES_DIR / "index.json", data)

def _load_note(note_id: str) -> dict | None:
    path = _NOTES_DIR / f"{note_id}.json"
    return _read_json(path) if path.exists() else None

def _save_note(note_id: str, data: dict):
    _write_json(_NOTES_DIR / f"{note_id}.json", data)


# ═══════════════════════════════════════════════════════════════════
#  PROMPT ASSEMBLY — The core of SoulScript identity persistence
# ═══════════════════════════════════════════════════════════════════

def _build_chat_messages(agent: str, messages: list[dict]) -> tuple[list[dict], dict, list[dict]]:
    """Assemble the full prompt with identity + knowledge + memory layers.

    Returns (llm_messages, layer_metadata, tool_defs) where:
      - llm_messages: the message array for the LLM
      - layer_metadata: what was injected at each stage (Prompt Inspector UI)
      - tool_defs: OpenAI-format tool definitions for the ``tools`` API param

    Injection order (highest priority first):
      1. Base system prompt    — the agent's personality / instructions
      2. Soul Script retrieval — FAISS search over directive-mode knowledge
      3. Always-on knowledge   — verbatim attached knowledge notes
      4. Memory Vault context  — FAISS search over persistent agent memories
      5. Conversation history  — recent user/assistant turns
      6. Tool registry         — function-calling definitions for allowed tools
    """
    layers = {
        "base_prompt": {"chars": 0, "preview": ""},
        "soul_script": {"chunks": 0, "chars": 0, "preview": ""},
        "always_on":   {"chunks": 0, "chars": 0, "preview": ""},
        "vault":       {"memories": 0, "chars": 0, "snippets": []},
        "conversation": {"turns": 0, "chars": 0},
        "tools":       {"count": 0, "names": []},
    }

    # ── 1. Base prompt ──
    system_prompt = _load_system_prompt(agent)
    layers["base_prompt"]["chars"] = len(system_prompt)
    layers["base_prompt"]["preview"] = system_prompt[:300]

    # Get latest user message for semantic search
    latest_user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            latest_user_msg = m.get("text", "")
            break

    # ── 2 & 3. Soul Script + Always-on knowledge ──
    always_block = ""
    directive_block = ""
    try:
        from src.storage.note_collector import collect_notes
        always_block, directive_block = collect_notes(agent, query=latest_user_msg)
        if directive_block:
            system_prompt += "\n\n" + directive_block
            layers["soul_script"]["chunks"] = directive_block.count("---") + 1
            layers["soul_script"]["chars"] = len(directive_block)
            layers["soul_script"]["preview"] = directive_block[:400]
        if always_block:
            system_prompt += "\n\n" + always_block
            layers["always_on"]["chunks"] = always_block.count("---") + 1
            layers["always_on"]["chars"] = len(always_block)
            layers["always_on"]["preview"] = always_block[:400]
    except Exception as exc:
        log.warning("[prompt] Note collection failed: %s", exc)

    # ── 4. Memory Vault context ──
    if latest_user_msg:
        try:
            fm = _get_faiss_memory()
            if fm:
                results = fm.search(latest_user_msg, scope=agent, top_k=5)
                if results:
                    snippets = [r["text"] for r in results if r.get("text")]
                    if snippets:
                        vault_block = "\n\n---\n\n".join(snippets)
                        system_prompt += (
                            "\n\n## Memory Vault Context\n\n"
                            "The following memories were retrieved from your persistent "
                            "memory vault based on relevance to the current conversation:\n\n"
                            + vault_block
                        )
                        layers["vault"]["memories"] = len(snippets)
                        layers["vault"]["chars"] = len(vault_block)
                        layers["vault"]["snippets"] = [s[:150] for s in snippets]
        except Exception as exc:
            log.warning("[prompt] Vault search failed: %s", exc)

    # ── Memory save instruction ──
    system_prompt += (
        "\n\n## Memory Save Protocol\n\n"
        "You have a persistent memory vault. When you want to save something "
        "to memory (because the user asked you to remember it, or because it is "
        "important biographical/preference/project info worth keeping), include "
        "one or more memory-save tags in your response like this:\n\n"
        "```\n[MEMORY_SAVE: category=preference | The user prefers dark mode and minimal UIs]\n```\n\n"
        "Valid categories: bio, preference, project, lore, session, meta, health, self, other.\n"
        "The system will automatically extract these and write them to your vault. "
        "You can include multiple MEMORY_SAVE tags in a single response.\n"
        "Always confirm to the user what you saved.\n"
        "The MEMORY_SAVE tag will be hidden from the user — they only see your natural text."
    )

    # ── 5. Conversation history (truncated to budget) ──
    MAX_CONTEXT_CHARS = 30_000
    conversation: list[dict] = []
    budget = MAX_CONTEXT_CHARS
    for m in reversed(messages):
        text = m.get("text", "")
        if len(text) > budget:
            break
        budget -= len(text)
        conversation.insert(0, {"role": m["role"], "content": text})

    layers["conversation"]["turns"] = len(conversation)
    layers["conversation"]["chars"] = MAX_CONTEXT_CHARS - budget

    # ── 6. Tool registry ──
    tool_defs: list[dict] = []
    try:
        from src.tools.registry import get_tool_defs_for_agent
        tool_defs = get_tool_defs_for_agent(agent)
        layers["tools"]["count"] = len(tool_defs)
        layers["tools"]["names"] = [d["function"]["name"] for d in tool_defs]
    except Exception as exc:
        log.warning("[prompt] Tool registry failed: %s", exc)

    return [{"role": "system", "content": system_prompt}] + conversation, layers, tool_defs


def _extract_and_save_memories(agent: str, response_text: str) -> list[dict]:
    """Parse [MEMORY_SAVE: ...] tags from response and write to vault.

    Returns list of saved memory summaries (for optional UI feedback).
    """
    import re
    pattern = r'\[MEMORY_SAVE:\s*(?:category=([\w]+)\s*\|)?\s*(.+?)\]'
    matches = re.findall(pattern, response_text, re.DOTALL)
    if not matches:
        return []

    fm = _get_faiss_memory()
    if not fm:
        log.warning("[memory] Vault not available — cannot save memories")
        return []

    saved = []
    for category_raw, text_raw in matches:
        category = (category_raw or "other").strip().lower()
        text = text_raw.strip()
        if not text or len(text) < 5:
            continue
        valid_cats = {"bio", "preference", "project", "lore", "session", "meta", "health", "self", "other"}
        if category not in valid_cats:
            category = "other"
        try:
            mem = fm.add(
                text=text,
                scope=agent,
                category=category,
                source="chat",
                tags=["auto-saved"],
            )
            saved.append({"id": mem.id, "text": text[:120], "category": category})
            log.info("[memory] Saved to vault: scope=%s cat=%s text=%.60s", agent, category, text)
        except Exception as exc:
            log.error("[memory] Failed to save memory: %s", exc)
    return saved


def _strip_memory_tags(text: str) -> str:
    """Remove [MEMORY_SAVE: ...] tags from text shown to user."""
    import re
    return re.sub(r'\[MEMORY_SAVE:\s*(?:category=[\w]+\s*\|)?\s*.+?\]', '', text).strip()


# ═══════════════════════════════════════════════════════════════════
#  PAGE ROUTES
# ═══════════════════════════════════════════════════════════════════

@app.get("/", response_class=RedirectResponse)
async def root():
    return RedirectResponse(url="/chat", status_code=302)

@app.get("/chat", response_class=HTMLResponse)
async def page_chat(request: Request):
    agents = _list_agents()
    store = _load_connections()
    conns = [c for c in store.get("connections", []) if c.get("enabled")]
    settings = _load_settings()
    return templates.TemplateResponse("chat.html", {
        "request": request, "page": "chat",
        "agents": agents, "connections": conns,
        "chat_index": _load_chat_index(),
        "agent_connections": store.get("agent_connections", {}),
        "avatar_map": settings.get("agent_avatars", {}),
    })

@app.get("/profiles", response_class=HTMLResponse)
async def page_profiles(request: Request):
    agents = _list_agents()
    settings = _load_settings()
    agent_data = {}
    for name in agents:
        profile = _load_profile(name)
        cfg = settings.get("agent_configs", {}).get(name, {})
        agent_data[name] = {
            "profile": profile, "config": cfg,
            "system_prompt": _load_system_prompt(name),
            "display_name": cfg.get("display_name", name),
            "description": cfg.get("description", ""),
        }
    store = _load_connections()
    all_models = []
    for c in store.get("connections", []):
        if c.get("enabled"):
            for m in c.get("models", []):
                if m not in all_models:
                    all_models.append(m)
    notes = [n for n in _load_notes_index() if not n.get("trashed")]
    # Builtin notes (agent-specific .md files in notes/ dir)
    notes_dir = _DATA_DIR / "user_notes"
    builtin_notes = {}
    for name in agents:
        agent_notes = []
        for md_file in sorted((_PROJECT_ROOT / "notes").glob("*.md")) if (_PROJECT_ROOT / "notes").exists() else []:
            agent_notes.append({"file": md_file.name, "attached": True, "mode": "always"})
        builtin_notes[name] = agent_notes
    return templates.TemplateResponse("profiles.html", {
        "request": request, "page": "profiles",
        "agents": agents, "agent_data": agent_data,
        "all_models": all_models, "notes": notes,
        "avatar_map": settings.get("agent_avatars", {}),
        "user_profile": settings.get("user_profile", {}),
        "connections": [c for c in store.get("connections", []) if c.get("enabled")],
        "builtin_notes": builtin_notes,
    })

@app.get("/vault", response_class=HTMLResponse)
async def page_vault(request: Request, q: str = "", scope: str = "", category: str = ""):
    fm = _get_faiss_memory()
    memories, scopes, categories = [], [], []
    stats = {"active_count": 0, "max_active": 500, "utilization_pct": 0,
             "by_scope": {}, "raw_lines": 0, "compactable_lines": 0,
             "bloat_ratio": "1.0x", "deleted_count": 0}
    if fm:
        try:
            stats = fm.stats()
        except Exception:
            pass
        if q:
            memories = fm.search(q, scope=scope or None, top_k=50)
        else:
            all_mems = fm.list_all(scope=scope or None)
            if category:
                all_mems = [m for m in all_mems if getattr(m, "category", "") == category]
            memories = [m.__dict__ if hasattr(m, "__dict__") else m for m in all_mems]
        all_raw = fm.list_all()
        scopes = sorted({getattr(m, "scope", "") for m in all_raw} - {""})
        categories = sorted({getattr(m, "category", "") for m in all_raw} - {""})

    return templates.TemplateResponse("vault.html", {
        "request": request, "page": "vault",
        "memories": memories, "stats": stats,
        "scopes": scopes, "categories": categories,
        "search_query": q, "current_scope": scope, "current_category": category,
    })

@app.get("/knowledge", response_class=HTMLResponse)
async def page_knowledge(request: Request):
    notes = [n for n in _load_notes_index() if not n.get("trashed")]
    return templates.TemplateResponse("knowledge.html", {
        "request": request, "page": "knowledge", "notes": notes,
    })

@app.get("/knowledge/{note_id}/edit", response_class=HTMLResponse)
async def page_knowledge_edit(request: Request, note_id: str):
    note = _load_note(note_id)
    if not note:
        return RedirectResponse(url="/knowledge", status_code=302)
    return templates.TemplateResponse("knowledge_edit.html", {
        "request": request, "page": "knowledge", "note": note,
    })

@app.get("/settings", response_class=HTMLResponse)
async def page_settings(request: Request, tab: str = "connections"):
    store = _load_connections()
    return templates.TemplateResponse("settings.html", {
        "request": request, "page": "settings",
        "connections": store.get("connections", []),
        "settings": _load_settings(), "tab": tab,
    })

@app.get("/pricing", response_class=RedirectResponse)
async def page_pricing_redirect():
    """Redirect old pricing page to tools (cost_tracker tool)."""
    return RedirectResponse(url="/tools#cost_tracker", status_code=302)


# ── Tools page ────────────────────────────────────────────────────

# Hardcoded tool catalogue (matches agent-runtime registry).
# Tools can be uploaded/implemented one at a time later.
_TOOL_CATALOGUE = [
    {
        "name": "web_search",
        "icon": "🔍",
        "icon_bg": "rgba(34,211,238,0.12)",
        "icon_color": "#22d3ee",
        "description": "Search the web via SearXNG and scrape/summarise results. Supports fast, normal, and deep modes. Includes knowledge-gate to prevent unnecessary searches.",
        "status": "ready",
        "parameters": [
            {"name": "action", "type": "string", "required": True, "description": "search or scrape", "enum": ["search", "scrape"]},
            {"name": "query", "type": "string", "required": False, "description": "Search query string", "enum": []},
            {"name": "mode", "type": "string", "required": False, "description": "Search depth preset", "enum": ["fast", "normal", "deep"]},
            {"name": "url", "type": "string", "required": False, "description": "URL to scrape directly", "enum": []},
            {"name": "knowledge_check", "type": "string", "required": False, "description": "What you already know about this topic", "enum": []},
            {"name": "reason", "type": "string", "required": False, "description": "Why the internet is needed", "enum": []},
        ],
    },
    {
        "name": "email",
        "icon": "📧",
        "icon_bg": "rgba(244,114,182,0.12)",
        "icon_color": "#f472b6",
        "description": "Draft, preview, and send emails via SMTP. Supports confirmation gating so the user can approve before sending.",
        "status": "planned",
        "parameters": [
            {"name": "to", "type": "string", "required": True, "description": "Recipient email address", "enum": []},
            {"name": "subject", "type": "string", "required": True, "description": "Email subject line", "enum": []},
            {"name": "body", "type": "string", "required": True, "description": "Email body text", "enum": []},
            {"name": "confirmation", "type": "string", "required": False, "description": "Set to 'confirmed' after user approval", "enum": ["confirmed"]},
        ],
    },
    {
        "name": "memory",
        "icon": "🧠",
        "icon_bg": "rgba(129,140,248,0.12)",
        "icon_color": "#818cf8",
        "description": "Read from and write to the agent's persistent FAISS-backed memory vault. Supports search, add, and delete operations.",
        "status": "planned",
        "parameters": [
            {"name": "action", "type": "string", "required": True, "description": "Operation to perform", "enum": ["search", "add", "delete", "list"]},
            {"name": "text", "type": "string", "required": False, "description": "Memory content or search query", "enum": []},
            {"name": "category", "type": "string", "required": False, "description": "Memory category", "enum": ["bio", "preference", "project", "lore", "session", "meta", "health", "self", "other"]},
        ],
    },
    {
        "name": "echo",
        "icon": "📢",
        "icon_bg": "rgba(16,185,129,0.12)",
        "icon_color": "#10b981",
        "description": "Simple echo tool for testing. Returns the input text unchanged. Useful for verifying tool dispatch works correctly.",
        "status": "ready",
        "parameters": [
            {"name": "text", "type": "string", "required": True, "description": "Text to echo back", "enum": []},
        ],
    },
    {
        "name": "directives",
        "icon": "📜",
        "icon_bg": "rgba(251,191,36,0.12)",
        "icon_color": "#fbbf24",
        "description": "Manage runtime directives — the living governance document that shapes agent behavior. Read, update, or query active directives.",
        "status": "planned",
        "parameters": [
            {"name": "action", "type": "string", "required": True, "description": "Operation", "enum": ["read", "update", "list"]},
            {"name": "directive_id", "type": "string", "required": False, "description": "Target directive identifier", "enum": []},
            {"name": "content", "type": "string", "required": False, "description": "New directive content", "enum": []},
        ],
    },
    {
        "name": "task_inbox",
        "icon": "📥",
        "icon_bg": "rgba(168,85,247,0.12)",
        "icon_color": "#a855f7",
        "description": "Read and write to a shared task inbox. Enables multi-agent task coordination and operator-to-agent task assignment.",
        "status": "planned",
        "parameters": [
            {"name": "action", "type": "string", "required": True, "description": "Operation", "enum": ["read", "write", "clear"]},
            {"name": "message", "type": "string", "required": False, "description": "Task message to write", "enum": []},
        ],
    },
    {
        "name": "continuation_update",
        "icon": "🔄",
        "icon_bg": "rgba(52,211,153,0.12)",
        "icon_color": "#34d399",
        "description": "Post continuation updates to keep multi-step workflows running. Agents use this to signal progress and request next steps.",
        "status": "planned",
        "parameters": [
            {"name": "update", "type": "string", "required": True, "description": "Continuation update text", "enum": []},
            {"name": "status", "type": "string", "required": False, "description": "Current task status", "enum": ["in_progress", "completed", "blocked"]},
        ],
    },
    {
        "name": "runtime_info",
        "icon": "📊",
        "icon_bg": "rgba(56,189,248,0.12)",
        "icon_color": "#38bdf8",
        "description": "Query runtime state, configuration, and system info. Returns details about loaded profiles, active tools, memory stats, and uptime.",
        "status": "planned",
        "parameters": [
            {"name": "query", "type": "string", "required": False, "description": "What info to retrieve", "enum": ["status", "config", "tools", "memory", "all"]},
        ],
    },
    {
        "name": "trent_inbox",
        "icon": "💬",
        "icon_bg": "rgba(239,68,68,0.12)",
        "icon_color": "#ef4444",
        "description": "Send a message to Trent (the operator). Used when the agent needs human input, wants to flag something important, or needs approval.",
        "status": "planned",
        "parameters": [
            {"name": "message", "type": "string", "required": True, "description": "Message for the operator", "enum": []},
            {"name": "priority", "type": "string", "required": False, "description": "Message urgency", "enum": ["low", "normal", "high", "critical"]},
        ],
    },
    {
        "name": "computer_use",
        "icon": "🖥️",
        "icon_bg": "rgba(99,102,241,0.12)",
        "icon_color": "#6366f1",
        "description": "Control the computer — take screenshots, move the mouse, click, type, and interact with desktop applications programmatically.",
        "status": "concept",
        "parameters": [
            {"name": "action", "type": "string", "required": True, "description": "Action to perform", "enum": ["screenshot", "click", "type", "move", "scroll", "key"]},
            {"name": "x", "type": "integer", "required": False, "description": "X coordinate", "enum": []},
            {"name": "y", "type": "integer", "required": False, "description": "Y coordinate", "enum": []},
            {"name": "text", "type": "string", "required": False, "description": "Text to type or key to press", "enum": []},
        ],
    },
    {
        "name": "cost_tracker",
        "icon": "💰",
        "icon_bg": "rgba(16,185,129,0.12)",
        "icon_color": "#10b981",
        "description": "Manage token pricing, track costs per model, and view spending across all LLM API calls. Edit per-model rates and monitor usage in real time.",
        "status": "ready",
        "parameters": [
            {"name": "action", "type": "string", "required": True, "description": "Operation to perform", "enum": ["get_pricing", "set_pricing", "list_models", "cost_summary", "cost_log", "session_cost"]},
            {"name": "provider", "type": "string", "required": False, "description": "Provider name (openai, anthropic, deepseek, ollama)", "enum": []},
            {"name": "model", "type": "string", "required": False, "description": "Model name", "enum": []},
            {"name": "period", "type": "string", "required": False, "description": "Time period for cost summary", "enum": ["today", "this_week", "this_month", "all_time"]},
        ],
    },
]

@app.get("/tools", response_class=HTMLResponse)
async def page_tools(request: Request):
    agents = _list_agents()
    # Gather pricing data for the cost_tracker tool panel
    pricing = _load_pricing()
    store = _load_connections()
    connections = [c for c in store.get("connections", []) if c.get("enabled")]
    try:
        from src.observability.metering import read_cost_log, aggregate_costs
        from datetime import timedelta
        now = datetime.now(timezone.utc)
        all_events = read_cost_log(limit=100000)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        today_events = [e for e in all_events if e.get("ts", "") >= today_start]
        month_start = (now - timedelta(days=30)).isoformat()
        month_events = [e for e in all_events if e.get("ts", "") >= month_start]
        cost_stats = {
            "today": aggregate_costs(today_events),
            "this_month": aggregate_costs(month_events),
            "all_time": aggregate_costs(all_events),
        }
    except Exception:
        cost_stats = {"today": {}, "this_month": {}, "all_time": {}}
    return templates.TemplateResponse("tools.html", {
        "request": request,
        "page": "tools",
        "tools": _TOOL_CATALOGUE,
        "agents": agents,
        "total": len(_TOOL_CATALOGUE),
        "pricing": pricing,
        "connections": connections,
        "cost_stats": cost_stats,
    })


# ── AGI Loop page (preview only — no backend connected) ──────────
AGI_LOOP_FILE = _CONFIG_DIR / "agi_loop.json"

_AGI_LOOP_DEFAULTS = {
    "interval_hours": 0,
    "interval_minutes": 30,
    "max_loops": 0,
    "ticks_per_loop": 15,
    "max_steps_per_tick": 3,
    "stimulus": "",
    "profile": "orion",
    "auto_pause_on_budget": True,
    "auto_pause_on_error_streak": 5,
    "monthly_hard_cap": 20.00,
    "monthly_soft_cap": 16.00,
    "per_session_cap": 2.00,
    "per_tick_cap": 0.10,
    "tiers": [
        {
            "id": "t0", "label": "local_cheap", "enabled": True,
            "connection_id": "", "provider": "ollama",
            "primary_model": "qwen2.5:7b",
            "temperature": 0.6, "max_output_tokens": 2048,
            "max_iterations": 8, "retries_before_escalate": 3,
            "alt_models": [],
            "default_for": "Memory ops, reflection, summarization",
            "cost_per_call": "~$0.00",
        },
        {
            "id": "t1", "label": "local_strong", "enabled": True,
            "connection_id": "", "provider": "ollama",
            "primary_model": "llama3:70b",
            "temperature": 0.5, "max_output_tokens": 4096,
            "max_iterations": 8, "retries_before_escalate": 3,
            "alt_models": [],
            "default_for": "Planning, general reasoning",
            "cost_per_call": "~$0.00",
        },
        {
            "id": "t2", "label": "cheap_cloud", "enabled": True,
            "connection_id": "", "provider": "deepseek",
            "primary_model": "deepseek-chat",
            "temperature": 0.4, "max_output_tokens": 8192,
            "max_iterations": 6, "retries_before_escalate": 2,
            "alt_models": [],
            "default_for": "Coding, high-stakes decisions",
            "cost_per_call": "~$0.001",
        },
        {
            "id": "t3", "label": "expensive_cloud", "enabled": True,
            "connection_id": "", "provider": "openai",
            "primary_model": "gpt-4o",
            "temperature": 0.3, "max_output_tokens": 16384,
            "max_iterations": 4, "retries_before_escalate": 2,
            "alt_models": [],
            "default_for": "Final polish, critical review",
            "cost_per_call": "~$0.01\u20130.10",
        },
    ],
}

def _load_agi_loop_config() -> dict:
    saved = _read_json(AGI_LOOP_FILE, {})
    merged = {**_AGI_LOOP_DEFAULTS, **saved}
    # Ensure tiers always present
    if "tiers" not in merged:
        merged["tiers"] = _AGI_LOOP_DEFAULTS["tiers"]
    return merged

def _save_agi_loop_config(data: dict):
    _write_json(AGI_LOOP_FILE, data)


@app.get("/agi-loop", response_class=HTMLResponse)
async def page_agi_loop(request: Request):
    agents = _list_agents()
    config = _load_agi_loop_config()
    connections = _load_connections().get("connections", [])
    return templates.TemplateResponse("agi_loop.html", {
        "request": request, "page": "agi-loop",
        "agents": agents, "config": config,
        "connections": connections,
    })


class AGILoopConfigUpdate(BaseModel):
    interval_hours: int = 0
    interval_minutes: int = 30
    max_loops: int = 0
    ticks_per_loop: int = 15
    max_steps_per_tick: int = 3
    stimulus: str = ""
    profile: str = "orion"
    auto_pause_on_budget: bool = True
    auto_pause_on_error_streak: int = 5
    monthly_hard_cap: float = 20.00
    monthly_soft_cap: float = 16.00
    per_session_cap: float = 2.00
    per_tick_cap: float = 0.10
    tiers: list = []


@app.get("/api/agi-loop/config")
async def api_agi_loop_config_get():
    return JSONResponse(_load_agi_loop_config())


@app.post("/api/agi-loop/config")
async def api_agi_loop_config_save(body: AGILoopConfigUpdate):
    data = body.dict()
    _save_agi_loop_config(data)
    return JSONResponse({"ok": True, "config": data})


# ── About page ───────────────────────────────────────────────────
ABOUT_FILE = _CONFIG_DIR / "about.json"

def _load_about() -> dict:
    return _read_json(ABOUT_FILE, {"text": ""})

def _save_about(data: dict):
    _write_json(ABOUT_FILE, data)

@app.get("/about", response_class=HTMLResponse)
async def page_about(request: Request):
    about = _load_about()
    return templates.TemplateResponse("about.html", {
        "request": request, "page": "about", "about_text": about.get("text", ""),
    })

class AboutUpdate(BaseModel):
    text: str

@app.post("/api/about")
async def api_about_save(body: AboutUpdate):
    _save_about({"text": body.text})
    return JSONResponse({"ok": True})


# ═══════════════════════════════════════════════════════════════════
#  CHAT API
# ═══════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    agent: str
    stimulus: str
    connection_id: Optional[str] = None
    model_override: Optional[str] = None
    chat_id: Optional[str] = None

@app.post("/api/chat/send")
async def api_chat_send(req: ChatRequest):
    conn = _resolve_connection(req.connection_id, req.agent)
    if not conn:
        return JSONResponse({"error": "No API connection available. Add one in Settings."}, 400)

    chat_data = _load_chat(req.chat_id) if req.chat_id else None
    if req.chat_id and not chat_data:
        return JSONResponse({"error": "Chat not found"}, 404)
    if not chat_data:
        chat_data = _create_new_chat(req.agent)

    now = datetime.now(timezone.utc).isoformat()
    chat_data["messages"].append({"role": "user", "text": req.stimulus, "time": now})
    chat_data["updated"] = now

    # Build prompt with all identity layers
    llm_messages, layers, tool_defs = _build_chat_messages(req.agent, chat_data["messages"])

    # Resolve model
    profile = _load_profile(req.agent)
    agent_cfg = _get_agent_config(req.agent)
    model = (
        req.model_override
        or agent_cfg.get("model")
        or profile.get("model", "")
        or (conn["models"][0] if conn.get("models") else "gpt-4o-mini")
    )

    # Call LLM API (with tool-call loop)
    url = conn["url"].rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if conn.get("api_key"):
        headers["Authorization"] = f"Bearer {conn['api_key']}"

    MAX_TOOL_ROUNDS = 10          # safety cap
    tool_call_log: list[dict] = []  # track every tool invocation for the UI
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    cost_data = {}

    running_messages = list(llm_messages)  # mutable copy for the loop

    for _round in range(MAX_TOOL_ROUNDS + 1):
        try:
            payload = {
                "model": model,
                "messages": running_messages,
                "temperature": profile.get("temperature", 0.7),
            }
            if tool_defs:
                payload["tools"] = tool_defs
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            return JSONResponse({"error": f"API {exc.response.status_code}: {exc.response.text[:200]}"}, 502)
        except Exception as exc:
            return JSONResponse({"error": f"Request failed: {exc}"}, 502)

        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        finish = choice.get("finish_reason", "stop")
        usage = data.get("usage", {})

        # Accumulate token usage across rounds
        for k in total_usage:
            total_usage[k] += usage.get(k, 0)

        # ── Cost metering (every round) ──
        provider = conn.get("provider", "openai")
        try:
            from src.observability.metering import meter_from_raw_usage, log_cost_event
            metering = meter_from_raw_usage(usage, provider=provider, model=model)
            cost_data = metering.cost.to_dict()
            log_cost_event(metering, agent=req.agent, chat_id=chat_data["id"])
        except Exception as exc:
            log.warning("[metering] cost computation failed: %s", exc)

        # ── If the LLM wants to call tools ──
        tool_calls = msg.get("tool_calls")
        if finish == "tool_calls" or tool_calls:
            # Append the assistant message WITH tool_calls to the running context
            running_messages.append(msg)

            from src.tools.registry import execute_tool
            import json as _json

            for tc in (tool_calls or []):
                fn_name = tc.get("function", {}).get("name", "")
                fn_args_raw = tc.get("function", {}).get("arguments", "{}")
                tc_id = tc.get("id", "")

                # Parse arguments
                try:
                    fn_args = _json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
                except _json.JSONDecodeError:
                    fn_args = {}

                # Execute
                log.info("[tools] Round %d — calling %s(%s)", _round + 1, fn_name, fn_args)
                try:
                    result = execute_tool(fn_name, fn_args)
                except Exception as exc:
                    result = f"Error: {exc}"
                    log.error("[tools] %s failed: %s", fn_name, exc)

                # Log for UI
                tool_call_log.append({
                    "round": _round + 1,
                    "tool": fn_name,
                    "arguments": fn_args,
                    "result": result[:500],
                })

                # Append tool result message for the next round
                running_messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": result,
                })

            # Continue the loop — LLM will see the tool results
            continue

        # ── Normal text response — done ──
        break

    raw_response = msg.get("content", "") or ""

    # Extract and save any [MEMORY_SAVE: ...] tags to the vault
    saved_memories = _extract_and_save_memories(req.agent, raw_response)
    # Strip memory tags from the text shown to the user
    response_text = _strip_memory_tags(raw_response)

    # Add tool layer to metadata
    layers["tools"]["calls"] = tool_call_log

    chat_data["messages"].append({
        "role": "assistant", "text": response_text, "time": now,
        "usage": total_usage,
        "data": {
            "agent": req.agent, "model": model,
            "usage": total_usage, "cost": cost_data,
            "tool_calls": tool_call_log,
        },
        "layers": layers,
    })
    _save_chat(chat_data["id"], chat_data)

    idx = _load_chat_index()
    for c in idx["chats"]:
        if c["id"] == chat_data["id"]:
            c["updated"] = now
            break
    _save_chat_index(idx)

    return {
        "response": response_text, "chat_id": chat_data["id"],
        "model": model, "usage": total_usage, "cost": cost_data, "layers": layers,
        "saved_memories": saved_memories,
        "tool_calls": tool_call_log,
    }

@app.get("/api/chat/history")
async def api_chat_history():
    return _load_chat_index()

@app.post("/api/chat/new")
async def api_chat_new(request: Request):
    body = await request.json()
    agents = _list_agents()
    agent = body.get("agent", agents[0] if agents else "agent")
    return _create_new_chat(agent)

@app.get("/api/chat/{chat_id}")
async def api_chat_get(chat_id: str):
    data = _load_chat(chat_id)
    return data if data else JSONResponse({"error": "Not found"}, 404)

@app.delete("/api/chat/{chat_id}")
async def api_chat_delete(chat_id: str):
    path = _CHATS_DIR / f"{chat_id}.json"
    if path.exists():
        path.unlink()
    idx = _load_chat_index()
    idx["chats"] = [c for c in idx["chats"] if c["id"] != chat_id]
    _save_chat_index(idx)
    return {"ok": True}

@app.put("/api/chat/{chat_id}")
async def api_chat_update(chat_id: str, request: Request):
    body = await request.json()
    data = _load_chat(chat_id)
    if not data:
        return JSONResponse({"error": "Not found"}, 404)
    for key in ("title", "folder_id"):
        if key in body:
            data[key] = body[key]
    _save_chat(chat_id, data)
    idx = _load_chat_index()
    for c in idx["chats"]:
        if c["id"] == chat_id:
            c.update({k: body[k] for k in ("title", "folder_id") if k in body})
            break
    _save_chat_index(idx)
    return {"ok": True}

@app.post("/api/chat/{chat_id}/title")
async def api_chat_auto_title(chat_id: str):
    data = _load_chat(chat_id)
    if not data or len(data.get("messages", [])) < 2:
        return {"title": None}

    agent = data.get("agent", "")
    conn = _resolve_connection(None, agent)
    if not conn:
        return {"title": None}

    snippet = "\n".join(f"{m['role']}: {m['text'][:200]}" for m in data["messages"][:4])
    profile = _load_profile(agent)
    model = _get_agent_config(agent).get("model") or profile.get("model", "") or "gpt-4o-mini"

    url = conn["url"].rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if conn.get("api_key"):
        headers["Authorization"] = f"Bearer {conn['api_key']}"

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "Generate a short (3-6 word) title for this conversation. Reply with ONLY the title."},
                    {"role": "user", "content": snippet},
                ],
                "temperature": 0.5, "max_tokens": 20,
            }, headers=headers)
            resp.raise_for_status()
            title = resp.json()["choices"][0]["message"]["content"].strip().strip('"')
    except Exception:
        return {"title": None}

    data["title"] = title
    _save_chat(chat_id, data)
    idx = _load_chat_index()
    for c in idx["chats"]:
        if c["id"] == chat_id:
            c["title"] = title
            break
    _save_chat_index(idx)
    return {"title": title}


# ═══════════════════════════════════════════════════════════════════
#  PROFILES API
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/profiles/{name}")
async def api_profile_get(name: str):
    profile = _load_profile(name)
    if not profile:
        return JSONResponse({"error": "Not found"}, 404)
    return {"name": name, "profile": profile, "config": _get_agent_config(name),
            "system_prompt": _load_system_prompt(name)}

@app.put("/api/profiles/{name}")
async def api_profile_update(name: str, request: Request):
    body = await request.json()
    if "system_prompt" in body:
        _save_system_prompt(name, body["system_prompt"])
    profile = _load_profile(name)
    for key in ("model", "temperature"):
        if key in body:
            profile[key] = body[key]
    _save_profile(name, profile)
    if "config" in body:
        cfg = _get_agent_config(name)
        cfg.update(body["config"])
        _save_agent_config(name, cfg)
    return {"ok": True}

@app.post("/api/profiles")
async def api_profile_create(request: Request):
    body = await request.json()
    name = body.get("name", "").strip().lower().replace(" ", "_")
    if not name:
        return JSONResponse({"error": "Name required"}, 400)
    if (_PROFILES_DIR / f"{name}.yaml").exists():
        return JSONResponse({"error": "Already exists"}, 400)
    _save_profile(name, {"name": name, "model": body.get("model", ""), "temperature": 0.7,
                         "system_prompt": f"{name}.system.md"})
    _save_system_prompt(name, body.get("system_prompt", f"You are {name}."))
    return {"ok": True, "name": name}

@app.delete("/api/profiles/{name}")
async def api_profile_delete(name: str):
    for p in [_PROFILES_DIR / f"{name}.yaml", _PROMPTS_DIR / f"{name}.system.md"]:
        if p.exists():
            p.unlink()
    settings = _load_settings()
    settings.get("agent_configs", {}).pop(name, None)
    _save_settings(settings)
    return {"ok": True}

@app.put("/api/profiles/{name}/config")
async def api_profile_config(name: str, request: Request):
    """Update individual config fields for an agent (display_name, description, model, etc.)."""
    body = await request.json()
    cfg = _get_agent_config(name)
    for key in ("display_name", "description", "model", "allowed_tools"):
        if key in body:
            cfg[key] = body[key]
    # Also persist model to profile yaml for compatibility
    if "model" in body:
        profile = _load_profile(name)
        profile["model"] = body["model"]
        _save_profile(name, profile)
    # Also persist system_prompt_text if sent
    if "system_prompt_text" in body:
        _save_system_prompt(name, body["system_prompt_text"])
    _save_agent_config(name, cfg)
    return {"ok": True}

@app.put("/api/profiles/{name}/avatar")
async def api_profile_avatar(name: str, request: Request):
    """Update avatar image or colour for an agent."""
    body = await request.json()
    settings = _load_settings()
    avatars = settings.setdefault("agent_avatars", {})
    entry = avatars.setdefault(name, {})
    if "color" in body:
        entry["color"] = body["color"]
    if "image" in body:
        if body["image"]:
            entry["image"] = body["image"]
        else:
            entry.pop("image", None)
    avatars[name] = entry
    _save_settings(settings)
    return {"ok": True}

@app.put("/api/profiles/user")
async def api_profile_user(request: Request):
    """Update user profile (name, avatar, color)."""
    body = await request.json()
    settings = _load_settings()
    user_profile = settings.setdefault("user_profile", {})
    for key in ("name", "color", "image"):
        if key in body:
            if key == "image" and not body[key]:
                user_profile.pop("image", None)
            else:
                user_profile[key] = body[key]
    _save_settings(settings)
    return {"ok": True}

@app.post("/api/profiles/create")
async def api_profile_create_v2(request: Request):
    """Create a new agent (v2 — supports description and model)."""
    body = await request.json()
    name = body.get("name", "").strip().lower().replace(" ", "_")
    if not name:
        return JSONResponse({"error": "Name required"}, 400)
    if (_PROFILES_DIR / f"{name}.yaml").exists():
        return JSONResponse({"error": "Already exists"}, 400)
    model = body.get("model", "")
    desc = body.get("description", "")
    _save_profile(name, {"name": name, "model": model, "temperature": 0.7,
                         "system_prompt": f"{name}.system.md"})
    _save_system_prompt(name, f"You are {name}.")
    if desc or model:
        cfg = _get_agent_config(name)
        if desc:
            cfg["description"] = desc
        if model:
            cfg["model"] = model
        _save_agent_config(name, cfg)
    return {"ok": True, "name": name}

@app.put("/api/profiles/{name}/knowledge")
async def api_profile_knowledge(name: str, request: Request):
    body = await request.json()
    cfg = _get_agent_config(name)
    cfg["attached_notes"] = body.get("attached_notes", [])
    cfg["note_modes"] = body.get("note_modes", {})
    _save_agent_config(name, cfg)
    try:
        from src.storage.note_collector import invalidate_notes_faiss
        invalidate_notes_faiss()
        _rebuild_notes_faiss()
    except Exception as exc:
        log.warning("[knowledge] FAISS rebuild skipped: %s", exc)
    return {"ok": True}

def _rebuild_notes_faiss():
    """Rebuild NotesFAISS index from all directive-mode notes across agents.

    Uses semantic chunking (split on ### headers) with overlapping fallback
    for long headerless content.  Each chunk carries ``document_id`` so the
    search filter in NotesFAISS matches correctly.
    """
    from src.memory.notes_faiss import NotesFAISS
    from src.storage.user_notes_loader import strip_html

    CHUNK_TARGET = 600   # chars per chunk (sweet spot for mpnet)
    CHUNK_OVERLAP = 150  # overlap between consecutive chunks

    def _chunk_text(text: str, doc_id: str, title: str) -> list[dict]:
        """Split text into overlapping chunks, preferring ### boundaries."""
        import re
        sections: list[tuple[str, str]] = []  # (section_title, body)
        parts = re.split(r'(?m)^###\s+', text)
        if len(parts) > 1:
            # First part is content before any header
            if parts[0].strip():
                sections.append((title, parts[0].strip()))
            for part in parts[1:]:
                lines = part.split('\n', 1)
                sec_title = lines[0].strip()
                sec_body = lines[1].strip() if len(lines) > 1 else ''
                if sec_body:
                    sections.append((sec_title, f'### {sec_title}\n{sec_body}'))
        else:
            sections.append((title, text))

        out = []
        for sec_title, body in sections:
            if len(body) <= CHUNK_TARGET + 100:
                out.append({
                    "text": body,
                    "metadata": {"document_id": doc_id, "document_title": title,
                                 "section_path": sec_title},
                })
            else:
                # Sliding window with overlap
                step = max(CHUNK_TARGET - CHUNK_OVERLAP, 200)
                for i in range(0, len(body), step):
                    chunk = body[i:i + CHUNK_TARGET]
                    if len(chunk) < 80 and out:
                        break  # skip tiny trailing scraps
                    out.append({
                        "text": chunk,
                        "metadata": {"document_id": doc_id, "document_title": title,
                                     "section_path": sec_title},
                    })
        return out

    chunks: list[dict] = []
    seen_note_ids: set[str] = set()
    for agent, cfg in _load_settings().get("agent_configs", {}).items():
        modes = cfg.get("note_modes", {})
        for nid in cfg.get("attached_notes", []):
            if modes.get(nid) == "directive" and nid not in seen_note_ids:
                seen_note_ids.add(nid)
                note = _load_note(nid)
                if note and not note.get("trashed"):
                    text = strip_html(note.get("content_html", ""))
                    if text:
                        chunks.extend(_chunk_text(text, nid, note.get("title", "Untitled")))

    faiss_dir = str(_FAISS_DIR)
    if chunks:
        nf = NotesFAISS(faiss_dir)
        nf.build_index(chunks)
        log.info("[knowledge] NotesFAISS rebuilt — %d chunks from %d notes", len(chunks), len(seen_note_ids))
    else:
        log.info("[knowledge] No directive-mode notes found — NotesFAISS empty")


# ═══════════════════════════════════════════════════════════════════
#  VAULT API
# ═══════════════════════════════════════════════════════════════════

@app.post("/api/vault/add")
async def api_vault_add(request: Request):
    """Manually add a memory to the vault."""
    body = await request.json()
    text = (body.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "Memory text is required"}, 400)
    fm = _get_faiss_memory()
    if not fm:
        return JSONResponse({"error": "Vault not available"}, 500)
    try:
        mem = fm.add(
            text=text,
            scope=body.get("scope", "shared"),
            category=body.get("category", "other"),
            source=body.get("source", "manual"),
            tags=body.get("tags", []),
        )
        return {"status": "saved", "id": mem.id, "text": text[:120]}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, 500)

@app.get("/api/vault/stats")
async def api_vault_stats():
    fm = _get_faiss_memory()
    return fm.stats() if fm else {"error": "Vault not available"}

@app.post("/api/vault/delete")
async def api_vault_delete(request: Request):
    body = await request.json()
    fm = _get_faiss_memory()
    if not fm:
        return {"error": "Vault not available"}
    deleted = [mid for mid in body.get("ids", []) if fm.delete(mid)]
    return {"deleted": deleted}

@app.get("/api/vault/compact")
async def api_vault_compact():
    fm = _get_faiss_memory()
    if not fm:
        return {"error": "Vault not available"}
    before = fm.stats().get("raw_lines", 0)
    fm.rebuild_index()
    return {"before_lines": before, "after_lines": fm.stats().get("raw_lines", 0)}


# ═══════════════════════════════════════════════════════════════════
#  KNOWLEDGE API
# ═══════════════════════════════════════════════════════════════════

@app.post("/api/knowledge")
async def api_knowledge_create(request: Request):
    body = await request.json()
    note_id = str(uuid.uuid4())[:8]
    now = datetime.now(timezone.utc).isoformat()
    note = {
        "id": note_id, "title": body.get("title", "Untitled"),
        "emoji": body.get("emoji", "📄"),
        "content_html": body.get("content_html", ""),
        "preview": body.get("preview", ""),
        "section": body.get("section", "Uncategorized"),
        "created": now, "updated": now,
    }
    _save_note(note_id, note)
    idx = _load_notes_index()
    idx.append({k: note[k] for k in ("id", "title", "emoji", "preview", "section", "created", "updated")})
    _save_notes_index(idx)
    return note

@app.put("/api/knowledge/{note_id}")
async def api_knowledge_update(note_id: str, request: Request):
    body = await request.json()
    note = _load_note(note_id)
    if not note:
        return JSONResponse({"error": "Not found"}, 404)
    for key in ("title", "emoji", "content_html", "preview", "section"):
        if key in body:
            note[key] = body[key]
    note["updated"] = datetime.now(timezone.utc).isoformat()
    _save_note(note_id, note)
    idx = _load_notes_index()
    for entry in idx:
        if entry["id"] == note_id:
            for key in ("title", "emoji", "preview", "section", "updated"):
                if key in note:
                    entry[key] = note[key]
            break
    _save_notes_index(idx)
    return note

@app.delete("/api/knowledge/{note_id}")
async def api_knowledge_delete(note_id: str):
    now = datetime.now(timezone.utc).isoformat()
    note = _load_note(note_id)
    if note:
        note["trashed"] = now
        _save_note(note_id, note)
    idx = _load_notes_index()
    for entry in idx:
        if entry["id"] == note_id:
            entry["trashed"] = now
            break
    _save_notes_index(idx)
    return {"ok": True}

@app.get("/api/knowledge/{note_id}")
async def api_knowledge_get(note_id: str):
    note = _load_note(note_id)
    return note if note else JSONResponse({"error": "Not found"}, 404)


# ═══════════════════════════════════════════════════════════════════
#  CONNECTIONS API
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/connections")
async def api_connections_list():
    return _load_connections().get("connections", [])

@app.post("/api/connections")
async def api_connections_create(request: Request):
    body = await request.json()
    store = _load_connections()
    conn = {
        "id": str(uuid.uuid4())[:8],
        "name": body.get("name", "Untitled"),
        "type": body.get("type", "external"),
        "provider": body.get("provider", "openai"),
        "url": body.get("url", ""),
        "api_key": body.get("api_key", ""),
        "models": body.get("models", []),
        "enabled": body.get("enabled", True),
    }
    store["connections"].append(conn)
    _save_connections(store)
    return conn

@app.put("/api/connections/{conn_id}")
async def api_connections_update(conn_id: str, request: Request):
    body = await request.json()
    store = _load_connections()
    for c in store["connections"]:
        if c["id"] == conn_id:
            for key in ("name", "type", "provider", "url", "api_key", "models", "enabled"):
                if key in body:
                    c[key] = body[key]
            break
    _save_connections(store)
    return {"ok": True}

@app.delete("/api/connections/{conn_id}")
async def api_connections_delete(conn_id: str):
    store = _load_connections()
    store["connections"] = [c for c in store["connections"] if c["id"] != conn_id]
    _save_connections(store)
    return {"ok": True}

@app.get("/api/connections/{conn_id}/models")
async def api_connections_fetch_models(conn_id: str):
    store = _load_connections()
    conn = next((c for c in store["connections"] if c["id"] == conn_id), None)
    if not conn:
        return JSONResponse({"error": "Not found"}, 404)

    provider = conn.get("provider", "openai")
    base_url = conn["url"].rstrip("/")
    headers = {"Authorization": f"Bearer {conn['api_key']}"} if conn.get("api_key") else {}

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            if provider == "ollama":
                resp = await client.get(f"{base_url}/api/tags", headers=headers)
                resp.raise_for_status()
                models = sorted(m["name"] for m in resp.json().get("models", []))
            else:
                resp = await client.get(f"{base_url}/models", headers=headers)
                resp.raise_for_status()
                models = sorted(m["id"] for m in resp.json().get("data", []))
    except Exception as exc:
        return {"error": str(exc)}

    for c in store["connections"]:
        if c["id"] == conn_id:
            c["models"] = models
            break
    _save_connections(store)
    return {"models": models}

@app.post("/api/connections/probe-models")
async def api_connections_probe_models(request: Request):
    """Fetch available models from a connection without it being saved first.
    Useful when adding a new connection — avoids browser CORS restrictions."""
    body = await request.json()
    provider = body.get("provider", "openai")
    base_url = (body.get("url") or "").rstrip("/")
    api_key = body.get("api_key", "")
    if not base_url:
        return JSONResponse({"error": "URL is required"}, 400)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            if provider == "ollama":
                resp = await client.get(f"{base_url}/api/tags", headers=headers)
                resp.raise_for_status()
                models = sorted(m["name"] for m in resp.json().get("models", []))
            else:
                resp = await client.get(f"{base_url}/models", headers=headers)
                resp.raise_for_status()
                models = sorted(m["id"] for m in resp.json().get("data", []))
    except Exception as exc:
        return {"error": str(exc)}
    return {"models": models}


# ═══════════════════════════════════════════════════════════════════
#  PRICING API
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/pricing")
async def api_pricing_get():
    """Return the full pricing registry."""
    return _load_pricing()

@app.put("/api/pricing")
async def api_pricing_update(request: Request):
    """Replace the entire pricing registry."""
    body = await request.json()
    _save_pricing(body)
    try:
        from src.observability.metering import reset_pricing_cache
        reset_pricing_cache()
    except Exception:
        pass
    return {"ok": True}

@app.put("/api/pricing/{provider}/{model:path}")
async def api_pricing_set_model(provider: str, model: str, request: Request):
    """Update pricing for a single provider/model."""
    body = await request.json()
    pricing = _load_pricing()
    if provider not in pricing:
        pricing[provider] = {}
    entry = pricing[provider].get(model, {})
    for key in ("input_per_1m", "cached_input_per_1m", "output_per_1m", "training_per_1m"):
        if key in body:
            entry[key] = float(body[key])
    pricing[provider][model] = entry
    _save_pricing(pricing)
    try:
        from src.observability.metering import reset_pricing_cache
        reset_pricing_cache()
    except Exception:
        pass
    return {"ok": True, "pricing": entry}

@app.delete("/api/pricing/{provider}/{model:path}")
async def api_pricing_delete_model(provider: str, model: str):
    """Remove pricing for a single model."""
    pricing = _load_pricing()
    if provider in pricing:
        pricing[provider].pop(model, None)
        if not pricing[provider]:
            del pricing[provider]
    _save_pricing(pricing)
    try:
        from src.observability.metering import reset_pricing_cache
        reset_pricing_cache()
    except Exception:
        pass
    return {"ok": True}

@app.get("/api/pricing/models")
async def api_pricing_all_models():
    """Return all models from all enabled connections + current pricing."""
    store = _load_connections()
    pricing = _load_pricing()
    result = []
    seen = set()
    for conn in store.get("connections", []):
        if not conn.get("enabled"):
            continue
        provider = conn.get("provider", "openai")
        conn_name = conn.get("name", provider)
        for m in conn.get("models", []):
            key = f"{provider}:{m}"
            if key in seen:
                continue
            seen.add(key)
            # Look up current pricing
            prov_prices = pricing.get(provider, {})
            mp = prov_prices.get(m)
            if not mp:
                for pk, pv in prov_prices.items():
                    if pk.startswith("_"):
                        continue
                    if isinstance(pv, dict) and m.startswith(pk):
                        mp = pv
                        break
            if not mp:
                mp = prov_prices.get("_default", {})
            result.append({
                "provider": provider,
                "connection": conn_name,
                "model": m,
                "input_per_1m": mp.get("input_per_1m", 0.0),
                "cached_input_per_1m": mp.get("cached_input_per_1m", 0.0),
                "output_per_1m": mp.get("output_per_1m", 0.0),
                "training_per_1m": mp.get("training_per_1m", 0.0),
            })
    return {"models": result}

@app.get("/api/pricing/cost-summary")
async def api_pricing_cost_summary(agent: str = "", period: str = "all"):
    """Return aggregated cost stats."""
    try:
        from src.observability.metering import read_cost_log, aggregate_costs
        from datetime import timedelta
        now_utc = datetime.now(timezone.utc)
        kwargs = {}
        if agent:
            kwargs["agent"] = agent
        if period == "today":
            kwargs["since"] = now_utc.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        elif period == "week":
            kwargs["since"] = (now_utc - timedelta(days=7)).isoformat()
        elif period == "month":
            kwargs["since"] = (now_utc - timedelta(days=30)).isoformat()
        events = read_cost_log(**kwargs, limit=100000)
        return aggregate_costs(events)
    except Exception as exc:
        return {"error": str(exc)}

@app.get("/api/pricing/cost-log")
async def api_pricing_cost_log(agent: str = "", since: str = "", limit: int = 100):
    """Return recent cost log entries."""
    try:
        from src.observability.metering import read_cost_log
        kwargs = {"limit": limit}
        if agent:
            kwargs["agent"] = agent
        if since:
            kwargs["since"] = since
        return {"events": read_cost_log(**kwargs)}
    except Exception as exc:
        return {"error": str(exc)}


# ═══════════════════════════════════════════════════════════════════
#  HEALTH
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def api_health():
    fm = _get_faiss_memory()
    return {"status": "ok", "agents": _list_agents(), "vault_loaded": fm is not None}
