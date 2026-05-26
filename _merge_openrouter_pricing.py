"""One-off: merge OpenRouter pricing block from live Fly container into local pricing.yaml.

Reads:  ./pricing_live.yaml   (just sftp'd from Fly)
Writes: ./orion-ui-standalone/config/pricing.yaml

Only replaces the `openrouter:` top-level key; everything else (openai, anthropic,
google, deepseek, xai, mistral, ollama) is preserved from the local file.
Also bumps the 'Last updated' header comment.
"""
from __future__ import annotations
import datetime
import io
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML not available — install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


LIVE = Path("pricing_live.yaml")
LOCAL = Path("orion-ui-standalone/config/pricing.yaml")


def main() -> int:
    live_raw = LIVE.read_text(encoding="utf-8")
    local_raw = LOCAL.read_text(encoding="utf-8")

    live = yaml.safe_load(live_raw) or {}
    local = yaml.safe_load(local_raw) or {}

    live_or = live.get("openrouter") or {}
    if not live_or:
        print("ERROR: live file has no openrouter section", file=sys.stderr)
        return 1

    old_or = local.get("openrouter") or {}
    local["openrouter"] = live_or

    # Stats
    added = sorted(set(live_or) - set(old_or))
    removed = sorted(set(old_or) - set(live_or))
    common = sorted(set(old_or) & set(live_or))
    changed: list[tuple[str, dict, dict]] = []
    for k in common:
        if old_or[k] != live_or[k]:
            changed.append((k, old_or[k], live_or[k]))

    # Preserve the header comment block from local file (lines starting with #)
    header_lines: list[str] = []
    for line in local_raw.splitlines():
        if line.startswith("#") or line.strip() == "":
            header_lines.append(line)
        else:
            break
    # Bump 'Last updated' line
    today = datetime.date.today().isoformat()
    header_lines = [
        ("#  Last updated: " + today) if l.startswith("#  Last updated:") else l
        for l in header_lines
    ]

    # Dump body
    buf = io.StringIO()
    yaml.safe_dump(local, buf, sort_keys=False, default_flow_style=False, allow_unicode=True)
    body = buf.getvalue()

    final = "\n".join(header_lines).rstrip() + "\n\n" + body
    LOCAL.write_text(final, encoding="utf-8")

    print(f"openrouter models: was={len(old_or)} now={len(live_or)}")
    print(f"  added={len(added)} removed={len(removed)} changed={len(changed)}")
    if changed[:10]:
        print("\nSample rate changes (first 10):")
        for k, a, b in changed[:10]:
            ai = a.get("input_per_1m"); bi = b.get("input_per_1m")
            ao = a.get("output_per_1m"); bo = b.get("output_per_1m")
            print(f"  {k}: in {ai} -> {bi} | out {ao} -> {bo}")
    print(f"\nWrote: {LOCAL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
