"""Patch every per-user settings.json on the Fly volume so that
agent_avatars.orion_cannon points at the new avatar image.
Idempotent.
"""
import json, os, glob, shutil, time

NEW_AVATAR = {
    "image": "/uploads/avatar_orion_cannon.png",
    "photo_zoom": 1,
    "photo_x": 50,
    "photo_y": 15,
}

paths = sorted(glob.glob("/persist/users/*/settings.json"))
print(f"found {len(paths)} per-user settings files")

for p in paths:
    try:
        with open(p, "r", encoding="utf-8") as f:
            s = json.load(f)
    except Exception as exc:
        print(f"[SKIP] {p}: {exc}")
        continue
    avs = s.setdefault("agent_avatars", {})
    existing = avs.get("orion_cannon")
    if existing == NEW_AVATAR:
        print(f"[ok ] {p} (already set)")
        continue
    # Backup once
    bak = f"{p}.bak.{int(time.time())}"
    shutil.copy2(p, bak)
    avs["orion_cannon"] = NEW_AVATAR
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2, ensure_ascii=False)
    os.replace(tmp, p)
    print(f"[upd] {p}  (had={existing})")

print("done")
