"""Register madara in the active user's per-user settings.json."""
import json, os, glob, shutil, time

UID = "a2cddfbd-dabb-4d50-b515-b600a08ce8e8"
SETTINGS = f"/persist/users/{UID}/settings.json"

madara_config = {
    "display_name": "Madara Uchiha",
    "attached_notes": ["5ccf5b26"],
    "note_modes": {"5ccf5b26": "directive"},
}

with open(SETTINGS, "r", encoding="utf-8") as f:
    s = json.load(f)
shutil.copy2(SETTINGS, f"{SETTINGS}.bak.{int(time.time())}")
s.setdefault("agent_configs", {})["madara"] = madara_config
tmp = SETTINGS + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(s, f, indent=2, ensure_ascii=False)
os.replace(tmp, SETTINGS)
print("[ok] madara registered:", s["agent_configs"]["madara"])
print("done")
