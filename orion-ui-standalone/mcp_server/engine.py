"""OrionForge MCP — engine core (no MCP dependency).

This module is the reusable heart of the per-user MCP server. It is ADDITIVE:
it imports OrionForge's existing modules read-only and writes only to the
existing Memory Vault (the same data/memory/vault.jsonl the app uses), so it
never changes the behavior of the running web app.

Configuration (all via environment, so the website can issue a per-user config):
    ORION_REPO      Path to the orion-ui-standalone repo (default: parent of this file)
    ORION_DATA_DIR  Per-user data dir holding memory/vault.jsonl + faiss
                    (default: {ORION_REPO}/data)
    ORION_USER      A label for the connected user (default: "local")

Capabilities exposed to the wrapper (mcp_server/orion_mcp.py):
    - list_agents / identity / soul_script_full
    - soul_search       : FAISS retrieval over an agent's soul script
    - call_agent        : identity prompt + retrieved soul-script chunks
    - get/set default agent  (default personality)
    - search_memory     : semantic search over the unified Memory Vault
    - save_project_summary : write a project summary into the unified vault
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _default_repo() -> str:
    return str(Path(__file__).resolve().parent.parent)


class OrionEngine:
    def __init__(
        self,
        repo: Optional[str] = None,
        data_dir: Optional[str] = None,
        user: Optional[str] = None,
    ) -> None:
        self.repo = Path(repo or os.environ.get("ORION_REPO", _default_repo())).resolve()
        self.data_dir = Path(
            data_dir or os.environ.get("ORION_DATA_DIR", str(self.repo / "data"))
        ).resolve()
        self.user = user or os.environ.get("ORION_USER", "local")

        self.profiles_dir = self.repo / "profiles"
        self.prompts_dir = self.repo / "prompts"
        self.directives_dir = self.repo / "directives"

        self.vault_path = self.data_dir / "memory" / "vault.jsonl"
        self.mem_faiss_dir = self.data_dir / "memory" / "faiss"
        # Dedicated soul-script index — kept separate from the app's NotesFAISS
        # so we never collide with the web app's index rebuilds.
        self.soul_faiss_dir = self.data_dir / "memory" / "mcp_soul_faiss"
        self.default_file = self.data_dir / "mcp_default_agent.txt"

        # Make OrionForge's src package importable.
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))

        self._notes_faiss = None        # lazy NotesFAISS over soul scripts
        self._faiss_mem = None          # lazy FaissMemory over the vault

    # ── Agents / identity / soul script ────────────────────────────
    def list_agents(self) -> List[str]:
        if not self.profiles_dir.exists():
            return []
        return sorted(p.stem for p in self.profiles_dir.glob("*.yaml"))

    def resolve_agent(self, name: str) -> Optional[str]:
        """Case/spacing-tolerant lookup so 'Marcus' or 'lux umbra' resolve."""
        if not name:
            return None
        agents = self.list_agents()
        key = name.strip().lower().replace(" ", "_").replace("-", "_")
        if key in agents:
            return key
        for a in agents:
            if a.lower() == key or a.lower().replace("_", "") == key.replace("_", ""):
                return a
        return None

    def identity(self, agent: str) -> str:
        p = self.prompts_dir / f"{agent}.system.md"
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def soul_script_full(self, agent: str) -> str:
        p = self.directives_dir / f"{agent}.md"
        return p.read_text(encoding="utf-8") if p.exists() else ""

    # ── Soul-script FAISS retrieval ────────────────────────────────
    @staticmethod
    def _chunk_soul(text: str, doc_id: str, title: str) -> List[Dict[str, Any]]:
        """Split a soul script into per-section chunks.

        Unlike the app's ### -only chunker, this splits on any level-2/level-3
        heading (## or ###) so section labels stay clean for retrieval.
        """
        lines = text.splitlines()
        chunks: List[Dict[str, Any]] = []
        cur_title = title
        cur_body: List[str] = []

        def flush():
            body = "\n".join(cur_body).strip()
            if body:
                chunks.append({
                    "text": body,
                    "metadata": {
                        "document_id": doc_id,
                        "document_title": title,
                        "section_path": cur_title,
                    },
                })

        for ln in lines:
            m = re.match(r"^(#{2,3})\s+(.*)$", ln)
            if m:
                flush()
                cur_title = m.group(2).strip()
                cur_body = []
            else:
                cur_body.append(ln)
        flush()
        return chunks

    def _get_notes_faiss(self):
        if self._notes_faiss is not None:
            return self._notes_faiss
        from src.memory.notes_faiss import NotesFAISS
        chunks: List[Dict[str, Any]] = []
        for agent in self.list_agents():
            ss = self.soul_script_full(agent).strip()
            if ss:
                chunks.extend(
                    self._chunk_soul(ss, f"__soul_script__{agent}", f"Soul Script — {agent}")
                )
        nf = NotesFAISS(str(self.soul_faiss_dir))
        if chunks:
            nf.build_index(chunks)
        self._notes_faiss = nf
        return nf

    def soul_search(self, agent: str, query: str, k: int = 5) -> List[Tuple[str, str, float]]:
        """Return [(section_path, text, score)] from the agent's soul script."""
        agent = self.resolve_agent(agent) or agent
        nf = self._get_notes_faiss()
        try:
            results = nf.search(query, top_k=k, note_ids={f"__soul_script__{agent}"})
        except Exception:
            return []
        out = []
        for chunk, score in results:
            meta = chunk.get("metadata", {})
            out.append((meta.get("section_path", ""), chunk.get("text", ""), float(score)))
        return out

    def call_agent(self, name: str, message: str = "", k: int = 5) -> Dict[str, Any]:
        """Identity prompt + soul-script chunks for an agent (call by name)."""
        agent = self.resolve_agent(name)
        if not agent:
            return {"error": f"Unknown agent '{name}'. Available: {', '.join(self.list_agents())}"}
        identity = self.identity(agent)
        if message.strip():
            retrieved = self.soul_search(agent, message, k=k)
        else:
            # No message yet — seed with the opening sections of the soul script.
            retrieved = self.soul_search(agent, identity[:500] or agent, k=k)
        return {
            "agent": agent,
            "identity_prompt": identity,
            "soul_script_sections": [
                {"section": sp, "text": t, "score": round(s, 3)} for sp, t, s in retrieved
            ],
        }

    # ── Default personality ────────────────────────────────────────
    def get_default_agent(self) -> Optional[str]:
        if self.default_file.exists():
            v = self.default_file.read_text(encoding="utf-8").strip()
            return self.resolve_agent(v)
        return None

    def set_default_agent(self, name: str) -> Dict[str, Any]:
        agent = self.resolve_agent(name)
        if not agent:
            return {"error": f"Unknown agent '{name}'. Available: {', '.join(self.list_agents())}"}
        self.default_file.parent.mkdir(parents=True, exist_ok=True)
        self.default_file.write_text(agent, encoding="utf-8")
        return {"ok": True, "default_agent": agent}

    # ── Unified Memory Vault ───────────────────────────────────────
    def _get_faiss_mem(self):
        if self._faiss_mem is not None:
            return self._faiss_mem
        from src.memory.faiss_memory import FAISSMemory
        self._faiss_mem = FAISSMemory(str(self.vault_path), str(self.mem_faiss_dir))
        return self._faiss_mem

    def search_memory(self, query: str, k: int = 8) -> List[Dict[str, Any]]:
        try:
            fm = self._get_faiss_mem()
            return fm.search(query, top_k=k)
        except Exception as exc:
            return [{"error": f"memory search failed: {exc}"}]

    def save_project_summary(
        self,
        summary: str,
        title: str = "",
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Write a project summary into the unified Memory Vault."""
        summary = (summary or "").strip()
        if not summary:
            return {"error": "summary text is required"}
        text = f"[Project: {title}] {summary}" if title.strip() else summary
        all_tags = ["project", "summary", "mcp"] + (tags or [])
        try:
            fm = self._get_faiss_mem()
            mem = fm.add(
                text=text,
                scope="shared",
                category="project",
                source=f"mcp:{self.user}",
                tags=all_tags,
                tier="register",
            )
            return {"ok": True, "id": mem.id, "stored": text[:120]}
        except Exception as exc:
            return {"error": f"save failed: {exc}"}
