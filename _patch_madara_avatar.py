"""Set madara avatar in active user settings.json."""
import json, os, shutil, time

UID = "a2cddfbd-dabb-4d50-b515-b600a08ce8e8"
SETTINGS = f"/persist/users/{UID}/settings.json"

entry = {
    "image": "/uploads/avatar_madara.jpg",
    "photo_zoom": 1,
    "photo_x": 50,
    "photo_y": 25,
}

with open(SETTINGS, "r", encoding="utf-8") as f:
    s = json.load(f)
shutil.copy2(SETTINGS, f"{SETTINGS}.bak.{int(time.time())}")
s.setdefault("agent_avatars", {})["madara"] = entry
tmp = SETTINGS + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(s, f, indent=2, ensure_ascii=False)
os.replace(tmp, SETTINGS)
print("[ok] madara avatar set:", s["agent_avatars"]["madara"])
