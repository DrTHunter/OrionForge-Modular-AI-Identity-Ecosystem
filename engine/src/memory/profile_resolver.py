"""Identity FAISS profile resolver.

Resolves the effective identity profile for a given agent:

  1. Read the agent's ``identity_faiss_profile`` field from its YAML.
  2. If it names a saved preset (e.g. ``"dense_chunking"``), load that
     from ``config/saved_profiles/identity/<name>.json``.
  3. Otherwise fall back to the global ``config/identity_profile.json``.
  4. If nothing exists on disk, return hardcoded safe defaults.

All downstream consumers (chunker, NotesFAISS, note_collector) call
``resolve_identity_profile()`` instead of using hardcoded constants.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"
_PROFILES_DIR = _PROJECT_ROOT / "profiles"
_SAVED_IDENTITY_DIR = _CONFIG_DIR / "saved_profiles" / "identity"
_GLOBAL_IDENTITY_FILE = _CONFIG_DIR / "identity_profile.json"

# Hardcoded fallback — used only when no config files exist at all.
_HARDCODED_DEFAULTS: Dict[str, Any] = {
    "indexing_policy": {
        "chunk_size_tokens": 400,
        "chunk_overlap_tokens": 80,
        "max_chunks_per_document": 200,
        "merge_short_paragraphs": True,
        "auto_section_tagging": True,
        "section_tag_format": "bracketed",
    },
    "retrieval_policy": {
        "top_k": 10,
        "max_tokens_from_identity": 1200,
        "semantic_match_weight": 0.8,
        "directive_match_weight": 1.0,
    },
    "source_priority_policy": {
        "boost_primary_in_retrieval": True,
        "primary_boost_multiplier": 1.3,
    },
}


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        log.warning("[profile_resolver] Failed to read %s: %s", path, exc)
        return None


def _load_agent_yaml(agent_name: str) -> dict:
    path = _PROFILES_DIR / f"{agent_name}.yaml"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        log.warning("[profile_resolver] Failed to read %s: %s", path, exc)
        return {}


def resolve_identity_profile(agent_name: Optional[str] = None) -> Dict[str, Any]:
    """Return the effective identity profile for *agent_name*.

    Resolution order:
      1. Agent YAML → ``identity_faiss_profile`` → saved preset file
      2. Global ``config/identity_profile.json``
      3. Hardcoded defaults (safe fallback)
    """
    # Step 1: try agent-specific preset
    if agent_name:
        agent_cfg = _load_agent_yaml(agent_name)
        preset_name = agent_cfg.get("identity_faiss_profile")
        if preset_name and preset_name != "__default__":
            preset = _read_json(_SAVED_IDENTITY_DIR / f"{preset_name}.json")
            if preset:
                log.debug("[profile_resolver] Agent '%s' using preset '%s'",
                          agent_name, preset_name)
                return preset

    # Step 2: global identity profile
    global_profile = _read_json(_GLOBAL_IDENTITY_FILE)
    if global_profile:
        return global_profile

    # Step 3: hardcoded defaults
    log.info("[profile_resolver] No identity profile found — using hardcoded defaults")
    return dict(_HARDCODED_DEFAULTS)


# ── Convenience accessors ──────────────────────────────────────────

def get_indexing_policy(agent_name: Optional[str] = None) -> Dict[str, Any]:
    """Return the indexing policy sub-dict (chunk_size, overlap, etc.)."""
    profile = resolve_identity_profile(agent_name)
    return profile.get("indexing_policy", _HARDCODED_DEFAULTS["indexing_policy"])


def get_retrieval_policy(agent_name: Optional[str] = None) -> Dict[str, Any]:
    """Return the retrieval policy sub-dict (top_k, weights, etc.)."""
    profile = resolve_identity_profile(agent_name)
    return profile.get("retrieval_policy", _HARDCODED_DEFAULTS["retrieval_policy"])


def get_source_priority_policy(agent_name: Optional[str] = None) -> Dict[str, Any]:
    """Return the source priority policy sub-dict."""
    profile = resolve_identity_profile(agent_name)
    return profile.get("source_priority_policy",
                        _HARDCODED_DEFAULTS["source_priority_policy"])
