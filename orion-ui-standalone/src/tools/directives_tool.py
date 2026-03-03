"""Directives tool — lets agents search and read user-authored directives.

Read-only: agents cannot modify directive files.
Self-contained: creates its own DirectiveStore on each call so edits
to directive files are picked up immediately.

Scopes are discovered dynamically from profile YAML files via
``manifest.SCOPES`` — new agents get their directives picked up
automatically without code changes.
"""

import json
import os
from typing import Any, Dict, List

from src.directives.store import DirectiveStore
from src.directives.manifest import load_manifest, generate_manifest, audit_changes, SCOPES

_BASE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DIRECTIVES_DIR = os.path.join(_BASE_DIR, "directives")


class DirectivesTool:
    """Read-only tool exposing directive search to agents."""

    @staticmethod
    def definition() -> Dict[str, Any]:
        return {
            "name": "directives",
            "description": (
                "Search and read user-authored directives. Directives contain "
                "structured instructions, protocols, and context organized by "
                "headings within markdown files. Read-only — agents cannot modify "
                "directive files.\n\n"
                "ACTIONS:\n"
                "- search: find relevant sections by semantic keyword query. "
                "Requires 'query'. Returns matching sections with heading, body, scope. "
                "Default limit 5.\n"
                "- list: show all available section headings. Optionally filter by scope.\n"
                "- get: read a specific section by its exact heading name. "
                "Requires 'heading'. Returns the full section body.\n"
                "- manifest: return the full directives manifest with IDs, versions, "
                "hashes, status, and token estimates. Useful for auditing what "
                "directives exist.\n"
                "- changes: diff live directive files against the persisted manifest. "
                "Shows added, removed, and changed entries since last manifest generation.\n\n"
                "Scopes are auto-discovered from profile YAML files. 'shared' scope "
                "is always included when filtering by a specific scope."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["search", "list", "get", "manifest", "changes"],
                        "description": (
                            "'search' finds relevant sections by query, "
                            "'list' shows all headings, "
                            "'get' returns a specific section by exact heading, "
                            "'manifest' returns the full directives manifest "
                            "(IDs, versions, hashes, status, token estimates), "
                            "'changes' diffs live directives against the persisted "
                            "manifest and returns added/removed/changed entries."
                        ),
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query string. Required for 'search' action.",
                    },
                    "heading": {
                        "type": "string",
                        "description": (
                            "Exact section heading to retrieve. Required for 'get'. "
                            "Use 'list' first to discover available headings."
                        ),
                    },
                    "scope": {
                        "type": "string",
                        "description": (
                            "Limit results to a specific scope (e.g. agent name). "
                            "'shared' scope is always included automatically. "
                            "Omit to search all scopes."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results for 'search' action. Default 5.",
                    },
                },
                "required": ["action"],
            },
        }

    @staticmethod
    def _resolve_scopes(scope: str | None) -> List[str]:
        """Resolve the scope parameter into a list of scopes to search.

        - Explicit scope  → ``["shared", <scope>]`` (always includes shared)
        - No scope        → all dynamically-discovered scopes from profiles
        """
        if scope:
            s = scope.lower()
            return ["shared", s] if s != "shared" else ["shared"]
        return list(SCOPES)

    @staticmethod
    def execute(arguments: Dict[str, Any]) -> str:
        action = arguments.get("action", "")
        scope = arguments.get("scope")
        scopes = DirectivesTool._resolve_scopes(scope)

        # Fresh store each call — picks up file edits immediately
        store = DirectiveStore(_DIRECTIVES_DIR, scopes=scopes)

        if action == "search":
            query = arguments.get("query", "")
            if not query:
                return json.dumps({"status": "error", "message": "query is required for search"})
            limit = arguments.get("limit", 5)
            results = store.search(query, limit=limit)
            return json.dumps({
                "status": "ok",
                "count": len(results),
                "sections": [
                    {"heading": s.heading, "body": s.body, "scope": s.scope}
                    for s in results
                ],
            })

        elif action == "list":
            headings = store.list_headings()
            return json.dumps({
                "status": "ok",
                "count": len(headings),
                "headings": headings,
            })

        elif action == "get":
            heading = arguments.get("heading", "")
            if not heading:
                return json.dumps({"status": "error", "message": "heading is required for get"})
            section = store.get_section(heading)
            if section is None:
                return json.dumps({"status": "not_found", "message": f"No section '{heading}'"})
            return json.dumps({
                "status": "ok",
                "heading": section.heading,
                "body": section.body,
                "scope": section.scope,
            })

        elif action == "manifest":
            # Try persisted manifest first, fall back to live generation
            manifest = load_manifest()
            if manifest is None:
                manifest = generate_manifest()
            # Optionally filter by scope
            directives = manifest.get("directives", [])
            if scope:
                target_scopes = {"shared", scope.lower()} if scope.lower() != "shared" else {"shared"}
                directives = [d for d in directives if d["scope"] in target_scopes]
            return json.dumps({
                "status": "ok",
                "manifest_version": manifest.get("manifest_version"),
                "generated_utc": manifest.get("generated_utc"),
                "hash_algo": manifest.get("hash_algo"),
                "count": len(directives),
                "directives": directives,
            })

        elif action == "changes":
            diff = audit_changes()
            return json.dumps({
                "status": "ok",
                "total_added": diff["total_added"],
                "total_removed": diff["total_removed"],
                "total_changed": diff["total_changed"],
                "unchanged_count": diff["unchanged_count"],
                "added": diff["added"],
                "removed": diff["removed"],
                "changed": diff["changed"],
            })

        else:
            return json.dumps({"status": "error", "message": f"Unknown action '{action}'"})
