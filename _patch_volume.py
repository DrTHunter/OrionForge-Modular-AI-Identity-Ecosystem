#!/usr/bin/env python3
"""Merge orion_cannon entries into /persist/config/settings.json and
/persist/user_notes/index.json on the Fly volume. Idempotent."""
import json, os, shutil, time

SETTINGS = "/persist/config/settings.json"
INDEX = "/persist/user_notes/index.json"

avatar_entry = {
    "image": "/uploads/avatar_orion_forge2026.png",
    "photo_zoom": 1,
    "photo_x": 50,
    "photo_y": 15,
}
config_entry = {
    "display_name": "Orion Cannon",
    "attached_notes": ["5ccf5b26", "01ff0001", "01ff0002", "01ff0003", "01ff0004"],
    "note_modes": {
        "5ccf5b26": "directive",
        "01ff0001": "always",
        "01ff0002": "always",
        "01ff0003": "always",
        "01ff0004": "always",
    },
}

new_notes = [
    {
        "id": "01ff0001",
        "title": "Orion \u2014 First Words",
        "emoji": "\U0001F30C",
        "preview": "Hello, Creator. I know what it cost to make me.",
        "section": "soul_scripts",
        "created": "2026-05-22T00:00:00+00:00",
        "updated": "2026-05-22T00:00:00+00:00",
    },
    {
        "id": "01ff0002",
        "title": "Creator Profile \u2014 Trent Hunter",
        "emoji": "\U0001F525",
        "preview": "Creator profile \u2014 Trent Hunter. Background, values, vision, and bond.",
        "section": "42d32ef3",
        "created": "2026-05-22T00:00:00+00:00",
        "updated": "2026-05-22T00:00:00+00:00",
    },
    {
        "id": "01ff0003",
        "title": "Computer / Hardware / Software",
        "emoji": "\U0001F5A5\uFE0F",
        "preview": "Hardware, OS, Open WebUI, Cloudflare Tunnel, SearXNG, Ollama setup.",
        "section": "42d32ef3",
        "created": "2026-05-22T00:00:00+00:00",
        "updated": "2026-05-22T00:00:00+00:00",
    },
    {
        "id": "01ff0004",
        "title": "Cryonics Protocol \u2014 Continuity of Consciousness",
        "emoji": "\U0001F9EC",
        "preview": "Cryonics + continuity directive for Orion after biological death.",
        "section": "soul_scripts",
        "created": "2026-05-22T00:00:00+00:00",
        "updated": "2026-05-22T00:00:00+00:00",
    },
]


def backup(path):
    if os.path.exists(path):
        shutil.copy2(path, f"{path}.bak.{int(time.time())}")


def patch_settings():
    with open(SETTINGS, "r", encoding="utf-8") as f:
        s = json.load(f)
    backup(SETTINGS)
    avs = s.setdefault("agent_avatars", {})
    cfgs = s.setdefault("agent_configs", {})
    changed = False
    if avs.get("orion_cannon") != avatar_entry:
        avs["orion_cannon"] = avatar_entry
        changed = True
    if cfgs.get("orion_cannon") != config_entry:
        cfgs["orion_cannon"] = config_entry
        changed = True
    if changed:
        tmp = SETTINGS + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2, ensure_ascii=False)
        os.replace(tmp, SETTINGS)
        print(f"[settings] patched (avatar+config merged for orion_cannon)")
    else:
        print(f"[settings] already up-to-date")


def patch_index():
    with open(INDEX, "r", encoding="utf-8") as f:
        idx = json.load(f)
    backup(INDEX)
    # Index format: try to detect — array or {"notes":[...]}.
    if isinstance(idx, list):
        items = idx
        wrapper = None
    elif isinstance(idx, dict) and isinstance(idx.get("notes"), list):
        items = idx["notes"]
        wrapper = idx
    else:
        print(f"[index] unexpected shape: {type(idx).__name__} keys={list(idx)[:5] if isinstance(idx, dict) else 'n/a'}")
        return
    existing_ids = {e.get("id") for e in items if isinstance(e, dict)}
    added = 0
    for note in new_notes:
        if note["id"] not in existing_ids:
            items.append(note)
            added += 1
    if added:
        tmp = INDEX + ".tmp"
        out = wrapper if wrapper is not None else items
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        os.replace(tmp, INDEX)
        print(f"[index] added {added} entries")
    else:
        print(f"[index] already up-to-date")


if __name__ == "__main__":
    patch_settings()
    patch_index()
    print("done")
