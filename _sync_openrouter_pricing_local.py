"""One-off: refresh the `openrouter:` pricing block in pricing.yaml from the LIVE
OpenRouter models API, then write it back to the app config.

Replicates the exact conversion used by web/app.py::_sync_openrouter_pricing:
  input_per_1m        = pricing.prompt      * 1_000_000
  output_per_1m       = pricing.completion  * 1_000_000
  cached_input_per_1m = input_per_1m * 0.5   (same 50% estimate as the server sync)
  training_per_1m     = 0.0

Additive merge (never deletes): existing entries are updated in place, brand-new
models are appended, and models no longer listed are left untouched. Only the
`openrouter:` section is rewritten — every other provider section, the Ollama
section, and the file header are preserved byte-for-byte.
"""
from __future__ import annotations

import datetime
import json
import re
import sys
import urllib.request
from pathlib import Path

import yaml

URL = "https://openrouter.ai/api/v1/models"
FILES = [
    Path("orion-ui-standalone/config/pricing.yaml"),
]


def fetch_models() -> list[dict]:
    req = urllib.request.Request(
        URL, headers={"User-Agent": "Mozilla/5.0 (OrionForge pricing sync)"}
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.load(resp).get("data", [])


def build_entry(model: dict) -> dict | None:
    pr = model.get("pricing") or {}
    prompt, completion = pr.get("prompt"), pr.get("completion")
    if prompt is None and completion is None:
        return None
    try:
        inp = float(prompt or 0) * 1_000_000
        out = float(completion or 0) * 1_000_000
    except (ValueError, TypeError):
        return None
    if inp < 0 or out < 0:  # dynamic routers (auto/fusion/pareto) price as "-1"
        return None
    return {
        "input_per_1m": round(inp, 4),
        "cached_input_per_1m": round(inp * 0.5, 4),
        "output_per_1m": round(out, 4),
        "training_per_1m": 0.0,
    }


def main() -> int:
    models = fetch_models()
    print(f"Fetched {len(models)} models from OpenRouter")

    api_entries: dict[str, dict] = {}
    for m in models:
        mid = m.get("id")
        if not mid:
            continue
        entry = build_entry(m)
        if entry is not None:
            api_entries[mid] = entry
    print(f"Priced entries usable from API: {len(api_entries)}")

    today = datetime.date.today().isoformat()

    for path in FILES:
        if not path.is_file():
            print(f"  SKIP (missing): {path}")
            continue
        text = path.read_text(encoding="utf-8")
        full = yaml.safe_load(text) or {}
        existing = full.get("openrouter") or {}

        added = updated = 0
        for mid, entry in api_entries.items():
            if mid in existing:
                if existing[mid] != entry:
                    updated += 1
                existing[mid] = entry
            else:
                existing[mid] = entry
                added += 1

        # Prune any pre-existing junk entries with negative prices (dynamic
        # routers like openrouter/auto report "-1" and must never be billed).
        pruned = [
            k for k, v in existing.items()
            if isinstance(v, dict)
            and (v.get("input_per_1m", 0) < 0 or v.get("output_per_1m", 0) < 0)
        ]
        for k in pruned:
            del existing[k]

        if "_default" not in existing:
            existing["_default"] = {
                "input_per_1m": 2.5,
                "cached_input_per_1m": 1.25,
                "output_per_1m": 10.0,
                "training_per_1m": 0.0,
            }
        # Keep _default first for readability
        ordered = {"_default": existing.pop("_default")}
        ordered.update(existing)

        block = yaml.safe_dump(
            {"openrouter": ordered},
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )
        if not block.endswith("\n"):
            block += "\n"

        m_or = re.search(r"^openrouter:", text, re.M)
        m_oll = re.search(r"^ollama:", text, re.M)
        if not m_or or not m_oll:
            print(f"  SKIP (no openrouter/ollama boundary): {path}")
            continue

        between = text[m_or.start():m_oll.start()]
        tail = re.search(r"(?:^[ \t]*#.*\n|^[ \t]*\n)+\Z", between, re.M)
        preserve = tail.group(0) if tail else ""

        new_text = text[: m_or.start()] + block + preserve + text[m_oll.start():]
        new_text = re.sub(
            r"#\s*Last updated:.*", f"#  Last updated: {today}", new_text, count=1
        )
        path.write_text(new_text, encoding="utf-8")
        print(
            f"  {path}: openrouter now {len(ordered)} entries "
            f"(added {added}, updated {updated}, pruned {len(pruned)})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
