"""Vault Store - append-only JSONL storage backend for the FAISS memory system.

This is a pure storage layer.  Every memory record lives as a JSON line
in `data/memory/vault.jsonl`.  Adds, updates, and deletes all append a
new line - the file is never edited in place.  On read, each id resolves
to its highest-version line.

The FAISS memory system (`faiss_memory.py`) composes over this store
for semantic search and embedding management.  This module handles only
persistence, versioning, and compaction.
"""

import json
import os
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, Set, Union
from zoneinfo import ZoneInfo

from src.memory.types import (
    Memory, MAX_MEMORY_TEXT_LENGTH, HARD_MAX_TOTAL_MEMORIES,
    MAX_TAGS_PER_MEMORY, MAX_BATCH_SIZE,
)
from src.memory.pii_guard import check_pii

# US Central Time - used for all vault timestamps.
_CT = ZoneInfo("America/Chicago")


def _now_ct() -> str:
    """Return the current time in US Central as an ISO string."""
    return datetime.now(_CT).isoformat()


class VaultStore:
    """Append-only JSONL storage for Memory records."""

    def __init__(self, path: str):
        self.path = path
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        # Batch state (managed by the batch() context manager)
        self._batch_active = False
        self._batch_file = None
        self._batch_resolve_cache: Optional[Dict[str, Memory]] = None

    # ------------------------------------------------------------------
    # Batch context manager
    # ------------------------------------------------------------------

    @contextmanager
    def batch(self) -> Generator[None, None, None]:
        """Context manager that batches writes into a single file handle.

        Within a batch:
        - ``_append()`` writes are buffered to one open file handle and
          flushed at the end instead of opening/closing per write.
        - ``resolve_latest()`` caches in memory and is updated by each
          ``_append()`` so we never re-read the whole JSONL mid-batch.

        Usage::

            with vault.batch():
                vault.create_memory("one", "shared", "fact")
                vault.create_memory("two", "shared", "fact")
        """
        if self._batch_active:
            # Already inside a batch — just pass through (re-entrant).
            yield
            return

        # Pre-load the resolve cache so all reads during the batch are
        # served from memory.
        self._batch_resolve_cache = self.resolve_latest()
        self._batch_active = True

        try:
            with open(self.path, "a", encoding="utf-8") as f:
                self._batch_file = f
                yield
                f.flush()
        finally:
            self._batch_file = None
            self._batch_active = False
            self._batch_resolve_cache = None

    def _active_count(self) -> int:
        """Return the number of active (non-deleted) memories."""
        return len(self.read_active())

    def batch_create_many(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Memory]:
        """Create multiple memories in a single batch.

        Each item dict may contain: text, scope, category, tags, source,
        tier, topic_id.  Returns the list of created Memory objects.
        PII is checked per-item; a failure on one item raises ValueError
        and none of the subsequent items are written.

        The caller does **not** need to wrap this in ``batch()`` — it
        manages the batch internally.
        """
        # ── Hard limit: batch size ────────────────────────────────
        if len(items) > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch too large: {len(items)} items (max {MAX_BATCH_SIZE})"
            )

        # ── Hard limit: total memory count ────────────────────────
        current = self._active_count()
        if current + len(items) > HARD_MAX_TOTAL_MEMORIES:
            room = max(0, HARD_MAX_TOTAL_MEMORIES - current)
            raise ValueError(
                f"Vault full: {current} active memories + {len(items)} new "
                f"exceeds hard limit of {HARD_MAX_TOTAL_MEMORIES:,}. "
                f"Room for {room} more."
            )

        # Validate all items up front before writing anything.
        prepared: List[Memory] = []
        for item in items:
            text = (item.get("text") or "").strip()
            if not text:
                raise ValueError("Memory text must not be empty")
            if len(text) > MAX_MEMORY_TEXT_LENGTH:
                raise ValueError(
                    f"Memory text too long: {len(text)} chars "
                    f"(max {MAX_MEMORY_TEXT_LENGTH})"
                )
            tags = item.get("tags") or []
            if len(tags) > MAX_TAGS_PER_MEMORY:
                raise ValueError(
                    f"Too many tags: {len(tags)} (max {MAX_TAGS_PER_MEMORY})"
                )
            pii = check_pii(text)
            if pii:
                raise ValueError(f"PII detected - memory blocked: {'; '.join(pii)}")

            mem = Memory(
                id=uuid.uuid4().hex[:12],
                text=text,
                scope=(item.get("scope") or "shared").lower(),
                category=(item.get("category") or "other").lower(),
                tier=(item.get("tier") or "canon").lower(),
                topic_id=item.get("topic_id"),
                tags=tags,
                created_at=_now_ct(),
                source=item.get("source") or "manual",
            )
            prepared.append(mem)

        # Write in a single batch (one file open/close).
        with self.batch():
            for mem in prepared:
                self._append(mem)

        return prepared

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add(self, mem: Memory) -> None:
        """Append a memory record to the vault file."""
        self._append(mem)

    def create_memory(
        self,
        text: str,
        scope: str,
        category: str = "other",
        tags: Optional[List[str]] = None,
        source: str = "manual",
        tier: str = "canon",
        topic_id: Optional[str] = None,
    ) -> Memory:
        """Create and persist a new memory.  Returns the Memory object.

        Validates PII, text length, tag count, and total-memory ceiling.
        Raises ValueError on any violation.
        """
        text = text.strip()
        if not text:
            raise ValueError("Memory text must not be empty")
        if len(text) > MAX_MEMORY_TEXT_LENGTH:
            raise ValueError(
                f"Memory text too long: {len(text)} chars "
                f"(max {MAX_MEMORY_TEXT_LENGTH})"
            )
        if tags and len(tags) > MAX_TAGS_PER_MEMORY:
            raise ValueError(
                f"Too many tags: {len(tags)} (max {MAX_TAGS_PER_MEMORY})"
            )

        # ── Hard limit: total memory count ────────────────────────
        current = self._active_count()
        if current >= HARD_MAX_TOTAL_MEMORIES:
            raise ValueError(
                f"Vault full: {current:,} active memories "
                f"(hard limit {HARD_MAX_TOTAL_MEMORIES:,}). "
                "Delete or compact before adding more."
            )

        pii = check_pii(text)
        if pii:
            raise ValueError(f"PII detected - memory blocked: {'; '.join(pii)}")

        mem = Memory(
            id=uuid.uuid4().hex[:12],
            text=text,
            scope=scope.lower(),
            category=category.lower(),
            tier=tier.lower(),
            topic_id=topic_id,
            tags=tags or [],
            created_at=_now_ct(),
            source=source,
        )
        self._append(mem)
        return mem

    def update_memory(
        self,
        memory_id: str,
        text: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        tier: Optional[str] = None,
        topic_id: Optional[str] = None,
    ) -> Memory:
        """Update a memory by appending a new version.

        Returns the new version.  Raises KeyError if not found.
        """
        resolved = self.resolve_latest()
        current = resolved.get(memory_id)
        if current is None or not current.is_active():
            raise KeyError(f"Memory '{memory_id}' not found or already deleted")

        if text is not None:
            text = text.strip()
            if not text:
                raise ValueError("Memory text must not be empty")
            if len(text) > MAX_MEMORY_TEXT_LENGTH:
                raise ValueError(
                    f"Memory text too long: {len(text)} chars "
                    f"(max {MAX_MEMORY_TEXT_LENGTH})"
                )
            pii = check_pii(text)
            if pii:
                raise ValueError(f"PII detected: {'; '.join(pii)}")
        if tags is not None and len(tags) > MAX_TAGS_PER_MEMORY:
            raise ValueError(
                f"Too many tags: {len(tags)} (max {MAX_TAGS_PER_MEMORY})"
            )

        new_version = Memory(
            id=current.id,
            text=text if text is not None else current.text,
            scope=current.scope,
            category=(category.lower() if category else current.category),
            tier=(tier.lower() if tier else current.tier),
            topic_id=(topic_id if topic_id is not None else current.topic_id),
            tags=(tags if tags is not None else list(current.tags)),
            created_at=current.created_at,
            updated_at=_now_ct(),
            source=current.source,
            deleted_at=None,
            version=current.version + 1,
        )
        self._append(new_version)
        return new_version

    def delete_memory(self, memory_id: str) -> bool:
        """Soft-delete by appending a tombstone.  Returns True if found."""
        resolved = self.resolve_latest()
        current = resolved.get(memory_id)
        if current is None or not current.is_active():
            return False

        tombstone = Memory(
            id=current.id,
            text=current.text,
            scope=current.scope,
            category=current.category,
            tier=current.tier,
            topic_id=current.topic_id,
            tags=list(current.tags),
            created_at=current.created_at,
            updated_at=current.updated_at,
            source=current.source,
            deleted_at=_now_ct(),
            version=current.version + 1,
        )
        self._append(tombstone)
        return True

    def bulk_delete(self, memory_ids: List[str]) -> Dict[str, List[str]]:
        """Soft-delete multiple memories.  Returns {deleted, not_found}."""
        resolved = self.resolve_latest()
        deleted, not_found = [], []
        now = _now_ct()
        for mid in memory_ids:
            current = resolved.get(mid)
            if current is None or not current.is_active():
                not_found.append(mid)
                continue
            tombstone = Memory(
                id=current.id, text=current.text, scope=current.scope,
                category=current.category, tier=current.tier,
                topic_id=current.topic_id, tags=list(current.tags),
                created_at=current.created_at, updated_at=current.updated_at,
                source=current.source, deleted_at=now,
                version=current.version + 1,
            )
            self._append(tombstone)
            deleted.append(mid)
        return {"deleted": deleted, "not_found": not_found}

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def read_all(self) -> List[Memory]:
        """Read every raw line (all versions, including tombstones)."""
        if not os.path.exists(self.path):
            return []
        records: List[Memory] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(Memory.from_dict(json.loads(line)))
        return records

    def resolve_latest(self) -> Dict[str, Memory]:
        """Resolve each id to its highest-version record.

        Inside a ``batch()`` context, returns the in-memory cache
        (kept in sync by ``_append``) instead of re-reading the file.
        """
        if self._batch_active and self._batch_resolve_cache is not None:
            return dict(self._batch_resolve_cache)  # shallow copy for safety

        latest: Dict[str, Memory] = {}
        for m in self.read_all():
            prev = latest.get(m.id)
            if prev is None or m.version > prev.version:
                latest[m.id] = m
        return latest

    def read_active(self) -> List[Memory]:
        """Return only non-deleted latest-version records."""
        return [m for m in self.resolve_latest().values() if m.is_active()]

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Get a single memory by id (latest version, must be active)."""
        resolved = self.resolve_latest()
        m = resolved.get(memory_id)
        if m and m.is_active():
            return m
        return None

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def compact(self) -> Dict[str, int]:
        """Rewrite vault to only active latest versions."""
        active = sorted(
            self.read_active(),
            key=lambda m: (m.category, m.created_at),
        )
        raw_before = len(self.read_all())
        tmp_path = self.path + ".compact.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for m in active:
                f.write(json.dumps(m.to_dict(), ensure_ascii=False) + "\n")
        os.replace(tmp_path, self.path)
        return {
            "lines_before": raw_before,
            "lines_after": len(active),
            "lines_removed": raw_before - len(active),
        }

    def stats(self) -> Dict[str, Any]:
        """Basic storage stats."""
        resolved = self.resolve_latest()
        active = [m for m in resolved.values() if m.is_active()]
        raw_lines = len(self.read_all())
        by_scope: Dict[str, int] = {}
        by_category: Dict[str, int] = {}
        by_tier: Dict[str, int] = {}
        for m in active:
            by_scope[m.scope] = by_scope.get(m.scope, 0) + 1
            by_category[m.category] = by_category.get(m.category, 0) + 1
            by_tier[m.tier] = by_tier.get(m.tier, 0) + 1
        return {
            "active_count": len(active),
            "deleted_count": len(resolved) - len(active),
            "raw_lines": raw_lines,
            "by_scope": by_scope,
            "by_category": by_category,
            "by_tier": by_tier,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _append(self, mem: Memory) -> None:
        line = json.dumps(mem.to_dict(), ensure_ascii=False) + "\n"
        if self._batch_active and self._batch_file is not None:
            self._batch_file.write(line)
            # Keep resolve cache in sync so reads within the batch see
            # the just-written record without re-reading the file.
            prev = self._batch_resolve_cache.get(mem.id)
            if prev is None or mem.version > prev.version:
                self._batch_resolve_cache[mem.id] = mem
        else:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)


# Backward-compat alias so existing `from src.memory.vault import MemoryVault`
# statements keep working during the transition.
MemoryVault = VaultStore
