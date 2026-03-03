"""Memory tool - lets agents store, search, and manage memories
via the FAISS-backed memory system.

Architecture:
  - vault.jsonl is the durable storage (each memory = one JSONL line)
  - FAISS provides semantic (meaning-based) search over vault contents
  - This tool is the agent-facing interface to both

All writes go to vault first, then sync to FAISS index automatically.
Each memory can be individually viewed and deleted.

Category policy (from memory_profile.json → category_policy):
  - "suggested" — enum is set to the suggested_categories list
  - "custom"    — enum is suggested + custom categories from config
  - "open"      — no enum at all; agent can use any category string
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.memory.faiss_memory import FAISSMemory
from src.memory.types import (
    VALID_CATEGORIES, VALID_SCOPES, VALID_SOURCES,
    MAX_MEMORY_TEXT_LENGTH, HARD_MAX_TOTAL_MEMORIES,
    MAX_TAGS_PER_MEMORY, MAX_BATCH_SIZE,
)
from src.data_paths import vault_path as _get_vault_path, faiss_dir as _get_faiss_dir

_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"


def _load_category_policy() -> dict:
    """Read category_policy from memory_profile.json."""
    mp_file = _CONFIG_DIR / "memory_profile.json"
    try:
        import json as _j
        data = _j.loads(mp_file.read_text(encoding="utf-8"))
        return data.get("category_policy", {})
    except Exception:
        return {}


def _build_category_field() -> dict:
    """Build the category parameter dict based on the active policy."""
    policy = _load_category_policy()
    mode = policy.get("mode", "open")
    suggested = policy.get("suggested_categories", sorted(VALID_CATEGORIES))
    custom = policy.get("custom_categories", [])

    base: dict = {
        "type": "string",
        "description": "Memory category.",
    }

    if mode == "suggested":
        base["enum"] = sorted(set(suggested))
        base["description"] = "Memory category (pick from suggested list)."
    elif mode == "custom":
        merged = sorted(set(suggested) | set(custom))
        base["enum"] = merged
        base["description"] = "Memory category (suggested + custom categories)."
    else:  # "open"
        # No enum — agent is free to use any category string.
        base["description"] = (
            "Memory category — use any descriptive string. "
            "Common categories: " + ", ".join(sorted(set(suggested))) + "."
        )

    return base


class MemoryTool:
    """Tool exposing FAISS-backed memory operations to agents."""

    def __init__(self):
        self._mem: FAISSMemory = None

    def _get_mem(self) -> FAISSMemory:
        """Lazy-create the FAISSMemory instance."""
        if self._mem is None:
            self._mem = FAISSMemory(
                vault_path=_get_vault_path(),
                faiss_dir=_get_faiss_dir(),
            )
        return self._mem

    @staticmethod
    def definition() -> Dict[str, Any]:
        return {
            "name": "memory",
            "description": (
                "Store, search, and manage durable memories. "
                "Memories persist across sessions in vault.jsonl and are "
                "searchable by meaning via FAISS vector embeddings.\n\n"
                "ACTIONS:\n"
                "- add: store a new memory (text + scope + category required). "
                "Returns the new memory id, scope, and category.\n"
                "- add_many: batch-store multiple memories in one call (max "
                f"{MAX_BATCH_SIZE} per batch). Returns list of created ids.\n"
                "- remember: quick-store with sensible defaults (scope=shared, "
                "category=other, source=tool). Only text is required.\n"
                "- search: semantic vector search — finds memories by meaning. "
                "Requires 'query'. Optionally filter by scope and category.\n"
                "- recall: list memories newest-first (no embedding needed). "
                "Filter by scope, category, or tags.\n"
                "- get: retrieve a single memory by id.\n"
                "- update: change text, category, or tags on an existing memory "
                "(does NOT accept scope or source changes).\n"
                "- delete: soft-delete a memory by id.\n"
                "- bulk_delete: soft-delete multiple memories by id list.\n"
                "- list: list all active memories (default limit 50).\n"
                "- stats: vault + FAISS health dashboard (total count, index size, etc).\n"
                "- compact: remove old versions/tombstones and rebuild index.\n"
                "- rebuild_index: rebuild FAISS index from vault (use if search "
                "seems stale)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "add", "add_many", "remember", "search", "recall",
                            "get", "update", "delete", "bulk_delete", "list",
                            "stats", "compact", "rebuild_index",
                        ],
                        "description": "The operation to perform.",
                    },
                    "text": {
                        "type": "string",
                        "description": (
                            "Memory content text. Required for add, remember. "
                            "Optional for update (replaces existing text). "
                            f"Max {MAX_MEMORY_TEXT_LENGTH} characters."
                        ),
                    },
                    "scope": {
                        "type": "string",
                        "enum": sorted(VALID_SCOPES),
                        "description": (
                            "Memory scope — determines who can access it. "
                            "Required for add. Used as filter for search/recall/list. "
                            "'shared' is visible to all agents."
                        ),
                    },
                    "category": _build_category_field(),
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            f"Optional tags (max {MAX_TAGS_PER_MEMORY} per memory). "
                            "Used for add/remember/update. Also used as filter for recall."
                        ),
                    },
                    "source": {
                        "type": "string",
                        "enum": sorted(VALID_SOURCES),
                        "description": "Origin of the memory (for add/remember). Default 'tool'.",
                    },
                    "tier": {
                        "type": "string",
                        "description": (
                            "Memory importance tier (for add/add_many). "
                            "Default 'register'. Use to classify memory priority."
                        ),
                    },
                    "topic_id": {
                        "type": "string",
                        "description": (
                            "Optional topic ID to group related memories together "
                            "(for add/add_many). Useful for threading memories about "
                            "the same subject."
                        ),
                    },
                    "query": {
                        "type": "string",
                        "description": "Semantic search query text. Required for 'search' action.",
                    },
                    "memory_id": {
                        "type": "string",
                        "description": "Memory ID. Required for get, update, and delete.",
                    },
                    "memory_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of memory IDs. Required for bulk_delete.",
                    },
                    "memories": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "scope": {"type": "string"},
                                "category": {"type": "string"},
                                "tags": {"type": "array", "items": {"type": "string"}},
                                "source": {"type": "string"},
                                "tier": {"type": "string"},
                                "topic_id": {"type": "string"},
                            },
                            "required": ["text", "scope", "category"],
                        },
                        "description": (
                            f"Array of memory objects for add_many batch (max {MAX_BATCH_SIZE}). "
                            "Each item requires text, scope, category. Optional: "
                            "tags, source, tier, topic_id."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": (
                            "Max results to return. Defaults: 10 for search, "
                            "20 for recall, 50 for list."
                        ),
                    },
                },
                "required": ["action"],
            },
        }

    def execute(self, arguments: Dict[str, Any]) -> str:
        action = arguments.get("action", "")
        try:
            handler = {
                "add": self._add,
                "add_many": self._add_many,
                "remember": self._remember,
                "search": self._search,
                "recall": self._recall,
                "get": self._get,
                "update": self._update,
                "delete": self._delete,
                "bulk_delete": self._bulk_delete,
                "list": self._list,
                "stats": self._stats,
                "compact": self._compact,
                "rebuild_index": self._rebuild_index,
            }.get(action)
            if handler is None:
                return json.dumps({"status": "error", "message": f"Unknown action '{action}'"})
            return handler(arguments)
        except (ValueError, KeyError) as exc:
            return json.dumps({"status": "error", "message": str(exc)})

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _add(self, args: Dict[str, Any]) -> str:
        text = args.get("text", "")
        scope = args.get("scope", "")
        category = args.get("category", "")
        if not text:
            return json.dumps({"status": "error", "message": "text is required"})
        if not scope:
            return json.dumps({"status": "error", "message": "scope is required"})
        if not category:
            return json.dumps({"status": "error", "message": "category is required"})
        # ── Hard limits ──
        if len(text) > MAX_MEMORY_TEXT_LENGTH:
            return json.dumps({"status": "error", "message": f"text too long: {len(text)} chars (max {MAX_MEMORY_TEXT_LENGTH})"})
        tags = args.get("tags") or []
        if len(tags) > MAX_TAGS_PER_MEMORY:
            return json.dumps({"status": "error", "message": f"too many tags: {len(tags)} (max {MAX_TAGS_PER_MEMORY})"})

        mem = self._get_mem().add(
            text=text,
            scope=scope,
            category=category,
            tags=args.get("tags"),
            source=args.get("source", "tool"),
            tier=args.get("tier", "register"),
            topic_id=args.get("topic_id"),
        )
        return json.dumps({
            "status": "stored",
            "id": mem.id,
            "scope": mem.scope,
            "category": mem.category,
        })

    def _add_many(self, args: Dict[str, Any]) -> str:
        memories = args.get("memories")
        if not memories or not isinstance(memories, list):
            return json.dumps({"status": "error",
                               "message": "memories array is required"})
        # ── Hard limit: batch size ──
        if len(memories) > MAX_BATCH_SIZE:
            return json.dumps({"status": "error",
                               "message": f"batch too large: {len(memories)} items (max {MAX_BATCH_SIZE})"})
        # Validate each item has required fields before sending to batch.
        for i, item in enumerate(memories):
            if not isinstance(item, dict):
                return json.dumps({"status": "error",
                                   "message": f"memories[{i}] must be an object"})
            if not item.get("text"):
                return json.dumps({"status": "error",
                                   "message": f"memories[{i}].text is required"})
            if len(item["text"]) > MAX_MEMORY_TEXT_LENGTH:
                return json.dumps({"status": "error",
                                   "message": f"memories[{i}].text too long: {len(item['text'])} chars (max {MAX_MEMORY_TEXT_LENGTH})"})
            if not item.get("scope"):
                return json.dumps({"status": "error",
                                   "message": f"memories[{i}].scope is required"})
            if not item.get("category"):
                return json.dumps({"status": "error",
                                   "message": f"memories[{i}].category is required"})
            item_tags = item.get("tags") or []
            if len(item_tags) > MAX_TAGS_PER_MEMORY:
                return json.dumps({"status": "error",
                                   "message": f"memories[{i}] has too many tags: {len(item_tags)} (max {MAX_TAGS_PER_MEMORY})"})
            # Default source to "tool" for agent-originated batch writes.
            item.setdefault("source", "tool")
            item.setdefault("tier", "register")

        created = self._get_mem().batch_add(memories)
        return json.dumps({
            "status": "stored",
            "count": len(created),
            "ids": [m.id for m in created],
        })

    def _remember(self, args: Dict[str, Any]) -> str:
        text = args.get("text", "")
        if not text:
            return json.dumps({"status": "error", "message": "text is required"})

        result = self._get_mem().remember(
            text=text,
            scope=args.get("scope", "shared"),
            category=args.get("category", "other"),
            source=args.get("source", "tool"),
            tags=args.get("tags"),
        )
        return json.dumps(result)

    def _search(self, args: Dict[str, Any]) -> str:
        query = args.get("query", "")
        if not query:
            return json.dumps({"status": "error", "message": "query is required"})

        results = self._get_mem().search(
            query=query,
            scope=args.get("scope"),
            category=args.get("category"),
            top_k=args.get("limit", 10),
        )
        return json.dumps({
            "status": "ok",
            "count": len(results),
            "memories": results,
        })

    def _recall(self, args: Dict[str, Any]) -> str:
        memories = self._get_mem().recall(
            scope=args.get("scope"),
            category=args.get("category"),
            tags=args.get("tags"),
            limit=args.get("limit", 20),
        )
        return json.dumps({
            "status": "ok",
            "count": len(memories),
            "memories": [self._fmt(m) for m in memories],
        })

    def _get(self, args: Dict[str, Any]) -> str:
        memory_id = args.get("memory_id", "")
        if not memory_id:
            return json.dumps({"status": "error", "message": "memory_id is required"})
        mem = self._get_mem().get(memory_id)
        if mem is None:
            return json.dumps({"status": "not_found"})
        return json.dumps({"status": "ok", "memory": self._fmt(mem)})

    def _update(self, args: Dict[str, Any]) -> str:
        memory_id = args.get("memory_id", "")
        if not memory_id:
            return json.dumps({"status": "error", "message": "memory_id is required"})

        new_ver = self._get_mem().update(
            memory_id,
            text=args.get("text"),
            category=args.get("category"),
            tags=args.get("tags"),
        )
        return json.dumps({
            "status": "updated",
            "id": new_ver.id,
            "version": new_ver.version,
        })

    def _delete(self, args: Dict[str, Any]) -> str:
        memory_id = args.get("memory_id", "")
        if not memory_id:
            return json.dumps({"status": "error", "message": "memory_id is required"})
        ok = self._get_mem().delete(memory_id)
        return json.dumps({"status": "deleted" if ok else "not_found"})

    def _bulk_delete(self, args: Dict[str, Any]) -> str:
        memory_ids = args.get("memory_ids", [])
        if not memory_ids:
            return json.dumps({"status": "error", "message": "memory_ids is required"})
        result = self._get_mem().bulk_delete(memory_ids)
        return json.dumps({
            "status": "ok",
            "deleted_count": len(result["deleted"]),
            "deleted": result["deleted"],
            "not_found": result["not_found"],
        })

    def _list(self, args: Dict[str, Any]) -> str:
        memories = self._get_mem().list_all(scope=args.get("scope"))
        limit = args.get("limit", 50)
        return json.dumps({
            "status": "ok",
            "count": len(memories[:limit]),
            "total": len(memories),
            "memories": [self._fmt(m) for m in memories[:limit]],
        })

    def _stats(self, args: Dict[str, Any]) -> str:
        return json.dumps({"status": "ok", **self._get_mem().stats()})

    def _compact(self, args: Dict[str, Any]) -> str:
        result = self._get_mem().compact()
        return json.dumps({"status": "ok", **result})

    def _rebuild_index(self, args: Dict[str, Any]) -> str:
        result = self._get_mem().rebuild_index()
        return json.dumps(result)

    @staticmethod
    def _fmt(m) -> Dict[str, Any]:
        d = {
            "id": m.id,
            "text": m.text,
            "scope": m.scope,
            "category": m.category,
            "tier": m.tier,
            "tags": m.tags,
            "source": m.source,
            "created_at": m.created_at,
            "version": m.version,
        }
        if m.topic_id:
            d["topic_id"] = m.topic_id
        return d
