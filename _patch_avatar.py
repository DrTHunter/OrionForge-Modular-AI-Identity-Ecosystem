import json, os
p = "/persist/config/settings.json"
with open(p, "r", encoding="utf-8") as f:
    s = json.load(f)
s.setdefault("agent_avatars", {}).setdefault("orion_cannon", {})
s["agent_avatars"]["orion_cannon"] = {
    "image": "/uploads/avatar_orion_cannon.png",
    "photo_zoom": 1,
    "photo_x": 50,
    "photo_y": 15,
}
with open(p + ".tmp", "w", encoding="utf-8") as f:
    json.dump(s, f, indent=2, ensure_ascii=False)
os.replace(p + ".tmp", p)
print("ok:", s["agent_avatars"]["orion_cannon"])
