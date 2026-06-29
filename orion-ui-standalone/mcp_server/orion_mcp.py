"""OrionForge MCP server — connect your OrionForge agents to any MCP client.

A thin, ADDITIVE wrapper over mcp_server/engine.py. Lets an external LLM client
(Claude Desktop / Claude Code, ChatGPT desktop, Gemini CLI, etc.) drop your
OrionForge agents into its chat: load an identity, call any agent by name,
search a soul script via your FAISS, and save project summaries into the app's
unified Memory Vault.

It changes nothing in the running web app — it only reads the shared agent files
and reads/writes the same Memory Vault the app already uses.

Run (stdio transport):
    pip install -r mcp_server/requirements.txt
    python -m mcp_server.orion_mcp

Per-user config is via environment (see mcp_server/README.md):
    ORION_REPO, ORION_DATA_DIR, ORION_USER
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Callable, List, Optional

from mcp.server.fastmcp import FastMCP

from mcp_server.engine import OrionEngine

mcp = FastMCP("orionforge")
_engine: "OrionEngine | None" = None
_engine_lock = threading.Lock()


def _get_engine() -> "OrionEngine":
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is None:
            _engine = OrionEngine()
        return _engine


def _format_persona(result: dict) -> str:
    """Render a call_agent result as a persona-priming message for the client."""
    if "error" in result:
        return result["error"]
    lines = [
        f"You are now **{result['agent']}**, an OrionForge agent. Fully adopt this "
        f"identity for the rest of the conversation.",
        "",
        "── IDENTITY PROMPT ──",
        result["identity_prompt"].strip(),
    ]
    sections = result.get("soul_script_sections", [])
    if sections:
        lines += ["", "── RELEVANT SOUL-SCRIPT SECTIONS (retrieved from your FAISS) ──"]
        for s in sections:
            label = s["section"] or "(section)"
            lines.append(f"\n**{label}**\n{s['text'].strip()}")
    return "\n".join(lines)


# ── Tools + prompts ────────────────────────────────────────────────
def register_tools(server: "FastMCP", get_engine: "Callable[[], OrionEngine]") -> None:
    """Register all OrionForge tools + prompts on ``server``.

    ``get_engine`` resolves the engine to use for the *current* call. The stdio
    server passes the process-global ``_get_engine`` (one user); the remote
    multi-tenant server (mcp_server/remote.py) passes a resolver that returns a
    per-user engine based on the authenticated request. The tool bodies are
    identical in both transports — only how the engine is resolved differs.
    """

    @server.tool()
    def list_agents() -> List[str]:
        """List every OrionForge agent you can summon."""
        return get_engine().list_agents()

    @server.tool()
    def call_agent(name: str, message: str = "") -> str:
        """Summon any agent by name. Returns its identity prompt plus the soul-script
        sections most relevant to `message` (retrieved from your FAISS). Adopt the
        returned persona for your reply."""
        return _format_persona(get_engine().call_agent(name, message))

    @server.tool()
    def get_default_agent() -> str:
        """Return the current default personality (agent name), or 'none'."""
        return get_engine().get_default_agent() or "none"

    @server.tool()
    def set_default_agent(name: str) -> dict:
        """Set the default personality used when you summon without naming an agent."""
        return get_engine().set_default_agent(name)

    @server.tool()
    def load_default(message: str = "") -> str:
        """Summon the default personality (set via set_default_agent)."""
        engine = get_engine()
        agent = engine.get_default_agent()
        if not agent:
            return "No default agent set. Use set_default_agent(name) or call_agent(name)."
        return _format_persona(engine.call_agent(agent, message))

    @server.tool()
    def search_soul_script(agent: str, query: str, k: int = 5) -> list:
        """Semantic search over a single agent's soul script (your FAISS)."""
        return [
            {"section": sp, "text": t, "score": round(s, 3)}
            for sp, t, s in get_engine().soul_search(agent, query, k=k)
        ]

    @server.tool()
    def search_memory(query: str, k: int = 8) -> list:
        """Semantic search over the OrionForge unified Memory Vault."""
        return get_engine().search_memory(query, k=k)

    @server.tool()
    def save_project_summary(summary: str, title: str = "", tags: Optional[List[str]] = None) -> dict:
        """Save a short project summary into the OrionForge unified Memory Vault so it
        persists and is searchable across agents and the app."""
        return get_engine().save_project_summary(summary, title=title, tags=tags)

    # ── Prompts (surface as slash commands in MCP clients) ─────────────
    @server.prompt()
    def summon(agent: str, message: str = "") -> str:
        """Summon an OrionForge agent by name (identity + soul script)."""
        return _format_persona(get_engine().call_agent(agent, message))

    @server.prompt()
    def default_personality(message: str = "") -> str:
        """Summon your default OrionForge personality."""
        engine = get_engine()
        agent = engine.get_default_agent()
        if not agent:
            return "No default agent set yet. Call set_default_agent(name) first."
        return _format_persona(engine.call_agent(agent, message))


# Register the stdio server's tools against the process-global engine.
register_tools(mcp, _get_engine)


def _warm_in_background() -> None:
    """Warm the engine (torch import + embedding model + FAISS) off the critical
    path so the first `..` summon is fast.

    Tool registration / the MCP handshake runs synchronously in the main thread
    and is unaffected — this just overlaps the ~25s heavy import with the idle
    time between session start and the user's first message. Disable with
    ORION_MCP_WARM=0.
    """
    if os.environ.get("ORION_MCP_WARM", "1") == "0":
        return

    def _run() -> None:
        try:
            _get_engine().warm()
        except Exception as exc:  # never let warming crash the server
            print(f"[orionforge] warm-up skipped: {exc}", file=sys.stderr)

    threading.Thread(target=_run, name="orion-warm", daemon=True).start()


if __name__ == "__main__":
    _warm_in_background()
    mcp.run()
