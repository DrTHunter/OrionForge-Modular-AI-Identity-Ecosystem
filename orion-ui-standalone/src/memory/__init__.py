"""Memory subsystem — FAISS semantic search backed by vault.jsonl storage."""

try:
    from src.memory.faiss_memory import FAISSMemory
except ImportError:
    FAISSMemory = None  # type: ignore[assignment,misc]

from src.memory.vault import VaultStore
from src.memory.types import Memory

__all__ = ["FAISSMemory", "VaultStore", "Memory"]
