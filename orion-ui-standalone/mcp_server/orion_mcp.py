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

from typing import List, Optional

from mcp.server.fastmcp import FastMCP

from mcp_server.engine import OrionEngine

mcp = FastMCP("orionforge")
engine = OrionEngine()


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


# ── Tools ──────────────────────────────────────────────────────────
@mcp.tool()
def list_agents() -> List[str]:
    """List every OrionForge agent you can summon."""
    return engine.list_agents()


@mcp.tool()
def call_agent(name: str, message: str = "") -> str:
    """Summon any agent by name. Returns its identity prompt plus the soul-script
    sections most relevant to `message` (retrieved from your FAISS). Adopt the
    returned persona for your reply."""
    return _format_persona(engine.call_agent(name, message))


@mcp.tool()
def get_default_agent() -> str:
    """Return the current default personality (agent name), or 'none'."""
    return engine.get_default_agent() or "none"


@mcp.tool()
def set_default_agent(name: str) -> dict:
    """Set the default personality used when you summon without naming an agent."""
    return engine.set_default_agent(name)


@mcp.tool()
def load_default(message: str = "") -> str:
    """Summon the default personality (set via set_default_agent)."""
    agent = engine.get_default_agent()
    if not agent:
        return "No default agent set. Use set_default_agent(name) or call_agent(name)."
    return _format_persona(engine.call_agent(agent, message))


@mcp.tool()
def search_soul_script(agent: str, query: str, k: int = 5) -> list:
    """Semantic search over a single agent's soul script (your FAISS)."""
    return [
        {"section": sp, "text": t, "score": round(s, 3)}
        for sp, t, s in engine.soul_search(agent, query, k=k)
    ]


@mcp.tool()
def search_memory(query: str, k: int = 8) -> list:
    """Semantic search over the OrionForge unified Memory Vault."""
    return engine.search_memory(query, k=k)


@mcp.tool()
def save_project_summary(summary: str, title: str = "", tags: Optional[List[str]] = None) -> dict:
    """Save a short project summary into the OrionForge unified Memory Vault so it
    persists and is searchable across agents and the app."""
    return engine.save_project_summary(summary, title=title, tags=tags)


# ── Prompts (surface as slash commands in MCP clients) ─────────────
@mcp.prompt()
def summon(agent: str, message: str = "") -> str:
    """Summon an OrionForge agent by name (identity + soul script)."""
    return _format_persona(engine.call_agent(agent, message))


@mcp.prompt()
def default_personality(message: str = "") -> str:
    """Summon your default OrionForge personality."""
    agent = engine.get_default_agent()
    if not agent:
        return "No default agent set yet. Call set_default_agent(name) first."
    return _format_persona(engine.call_agent(agent, message))


if __name__ == "__main__":
    mcp.run()
