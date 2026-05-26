"""Ensure user a2cddfbd has full orion_cannon config (display name + attached notes)
in their per-user settings.json, plus matching index.json entries for the 4
new notes.  Idempotent.
"""
import json, os, glob, shutil, time

UID = "a2cddfbd-dabb-4d50-b515-b600a08ce8e8"
SETTINGS = f"/persist/users/{UID}/settings.json"
NOTES_INDEX = f"/persist/users/{UID}/notes/index.json"
NOTES_DIR = f"/persist/users/{UID}/notes"

avatar_entry = {
    "image": "/uploads/avatar_orion_cannon.png",
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

new_notes_index_entries = [
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


def backup(p):
    if os.path.exists(p):
        shutil.copy2(p, f"{p}.bak.{int(time.time())}")


# ---- settings.json ----
with open(SETTINGS, "r", encoding="utf-8") as f:
    s = json.load(f)
print("[before] agent_avatars.orion_cannon =", s.get("agent_avatars", {}).get("orion_cannon"))
print("[before] agent_configs.orion_cannon =", s.get("agent_configs", {}).get("orion_cannon"))
backup(SETTINGS)
s.setdefault("agent_avatars", {})["orion_cannon"] = avatar_entry
s.setdefault("agent_configs", {})["orion_cannon"] = config_entry
tmp = SETTINGS + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(s, f, indent=2, ensure_ascii=False)
os.replace(tmp, SETTINGS)
print("[after ] agent_avatars.orion_cannon =", s["agent_avatars"]["orion_cannon"])
print("[after ] agent_configs.orion_cannon =", s["agent_configs"]["orion_cannon"])

# ---- notes index ----
if os.path.exists(NOTES_INDEX):
    with open(NOTES_INDEX, "r", encoding="utf-8") as f:
        idx = json.load(f)
    if isinstance(idx, list):
        items = idx
        wrapper = None
    elif isinstance(idx, dict) and isinstance(idx.get("notes"), list):
        items = idx["notes"]
        wrapper = idx
    else:
        print(f"[index] unexpected shape: {type(idx).__name__}")
        items = None
        wrapper = None
    if items is not None:
        backup(NOTES_INDEX)
        existing_ids = {e.get("id") for e in items if isinstance(e, dict)}
        added = 0
        for note in new_notes_index_entries:
            if note["id"] not in existing_ids:
                items.append(note)
                added += 1
        out = wrapper if wrapper is not None else items
        tmp = NOTES_INDEX + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        os.replace(tmp, NOTES_INDEX)
        print(f"[index] added {added} entries")
else:
    print(f"[index] no notes/index.json at {NOTES_INDEX}; skipping")

# ---- copy note JSON files into user notes dir if missing ----
src_dir = "/persist/user_notes"
if os.path.isdir(src_dir) and os.path.isdir(NOTES_DIR):
    for nid in ("01ff0001", "01ff0002", "01ff0003", "01ff0004"):
        src = os.path.join(src_dir, f"{nid}.json")
        dst = os.path.join(NOTES_DIR, f"{nid}.json")
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"[notes] copied {nid}.json -> {dst}")

print("done")
