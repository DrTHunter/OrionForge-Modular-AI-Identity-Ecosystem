"""Flip orion_cannon note_modes from 'always' to 'directive' for the live user."""
import json, os, glob, shutil, time

NOTE_IDS = ["01ff0001", "01ff0002", "01ff0003", "01ff0004"]
for p in sorted(glob.glob("/persist/users/*/settings.json")):
    try:
        with open(p, "r", encoding="utf-8") as f:
            s = json.load(f)
    except Exception as exc:
        print(f"[skip] {p}: {exc}")
        continue
    cfg = s.get("agent_configs", {}).get("orion_cannon")
    if not cfg:
        print(f"[--] {p}: no orion_cannon config")
        continue
    modes = cfg.setdefault("note_modes", {})
    changed = False
    for nid in NOTE_IDS:
        if modes.get(nid) != "directive":
            modes[nid] = "directive"
            changed = True
    if not changed:
        print(f"[ok] {p} (already directive)")
        continue
    shutil.copy2(p, f"{p}.bak.{int(time.time())}")
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2, ensure_ascii=False)
    os.replace(tmp, p)
    print(f"[upd] {p}")
print("done")
