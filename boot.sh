#!/bin/bash
# ── OrionForge Fly.io Boot Script ──────────────────────────────
# Ensures uploads, memory vault, config, user notes, chat history,
# and trash are on the persistent volume so they survive deploys.
# ─────────────────────────────────────────────────────────────────

set -e

PERSIST_UPLOADS="/persist/uploads"
APP_UPLOADS="/app/app/data/uploads"
BUNDLED_UPLOADS="/app/_bundled_uploads"

PERSIST_MEMORY="/persist/memory"
APP_MEMORY="/app/app/data/memory"
BUNDLED_MEMORY="/app/_bundled_memory"

PERSIST_CONFIG="/persist/config"
APP_CONFIG="/app/app/config"
BUNDLED_CONFIG="/app/_bundled_config"

PERSIST_NOTES="/persist/user_notes"
APP_NOTES="/app/app/data/user_notes"

# 1. Create persistent uploads dir if it doesn't exist
mkdir -p "$PERSIST_UPLOADS"

# 2. Sync bundled avatars into persistent storage.
#    The Dockerfile stages git-tracked uploads to /app/_bundled_uploads
#    so they're always accessible even after the live dir is symlinked.
#    Force-overwrite so updated avatars in the repo always take effect;
#    user-uploaded files (not in bundled) are unaffected.
if [ -d "$BUNDLED_UPLOADS" ]; then
    cp -a "$BUNDLED_UPLOADS"/* "$PERSIST_UPLOADS/" 2>/dev/null || true
fi

# 3. Remove the real uploads dir (or old symlink) and point to persistent volume
if [ -L "$APP_UPLOADS" ]; then
    rm "$APP_UPLOADS"
elif [ -d "$APP_UPLOADS" ]; then
    rm -rf "$APP_UPLOADS"
fi
ln -sfn "$PERSIST_UPLOADS" "$APP_UPLOADS"

echo "[boot] Uploads directory linked to persistent volume."

# ââ MEMORY VAULT PERSISTENCE âââââââââââââââââââââââââââ
# 4. Create persistent memory dir if it doesn't exist
mkdir -p "$PERSIST_MEMORY"
mkdir -p "$PERSIST_MEMORY/faiss"

# 5. Seed bundled vault into persistent storage on first deploy.
#    If vault.jsonl doesn't exist yet on the volume, copy the bundled one.
#    If it already exists, leave it alone (user memories are preserved).
if [ -d "$BUNDLED_MEMORY" ]; then
    if [ ! -f "$PERSIST_MEMORY/vault.jsonl" ]; then
        echo "[boot] First deploy — seeding bundled memory vault."
        cp -a "$BUNDLED_MEMORY"/* "$PERSIST_MEMORY/" 2>/dev/null || true
    else
        echo "[boot] Persistent vault exists — preserving user memories."
    fi
fi

# 6. Remove the real memory dir (or old symlink) and point to persistent volume
if [ -L "$APP_MEMORY" ]; then
    rm "$APP_MEMORY"
elif [ -d "$APP_MEMORY" ]; then
    rm -rf "$APP_MEMORY"
fi
ln -sfn "$PERSIST_MEMORY" "$APP_MEMORY"

echo "[boot] Memory vault linked to persistent volume."

# ── CONFIG PERSISTENCE ─────────────────────────────────
# 7. Create persistent config dir if it doesn't exist
mkdir -p "$PERSIST_CONFIG"

# 8. Seed bundled config on first deploy; preserve user changes thereafter.
if [ -d "$BUNDLED_CONFIG" ]; then
    if [ ! -f "$PERSIST_CONFIG/settings.json" ]; then
        echo "[boot] First deploy — seeding bundled config."
        cp -a "$BUNDLED_CONFIG"/* "$PERSIST_CONFIG/" 2>/dev/null || true
    else
        echo "[boot] Persistent config exists — preserving user settings."
        # Merge in any NEW config files shipped in the image (no-clobber)
        cp -n "$BUNDLED_CONFIG"/* "$PERSIST_CONFIG/" 2>/dev/null || true
    fi
fi

# 9. Symlink config dir to persistent volume
if [ -L "$APP_CONFIG" ]; then
    rm "$APP_CONFIG"
elif [ -d "$APP_CONFIG" ]; then
    rm -rf "$APP_CONFIG"
fi
ln -sfn "$PERSIST_CONFIG" "$APP_CONFIG"

echo "[boot] Config linked to persistent volume."

# ── PROFILES PERSISTENCE ───────────────────────────────
# Agent profile YAMLs define the shipped defaults (voice, model, tools, etc.).
# Always overwrite from bundled copy so new defaults take effect.
# User customizations are stored in settings.json agent_configs (which takes
# priority at render time), so overwriting profiles is safe.
PERSIST_PROFILES="/persist/profiles"
APP_PROFILES="/app/app/profiles"
BUNDLED_PROFILES="/app/_bundled_profiles"

mkdir -p "$PERSIST_PROFILES"

if [ -d "$BUNDLED_PROFILES" ]; then
    echo "[boot] Syncing bundled profiles to persistent volume."
    cp -a "$BUNDLED_PROFILES"/* "$PERSIST_PROFILES/" 2>/dev/null || true
fi

if [ -L "$APP_PROFILES" ]; then
    rm "$APP_PROFILES"
elif [ -d "$APP_PROFILES" ]; then
    rm -rf "$APP_PROFILES"
fi
ln -sfn "$PERSIST_PROFILES" "$APP_PROFILES"

echo "[boot] Profiles linked to persistent volume."

# ── USER NOTES PERSISTENCE ─────────────────────────────
# 10. Create persistent user_notes dir if it doesn't exist
mkdir -p "$PERSIST_NOTES"

# 11. Seed bundled notes on first deploy; preserve user notes thereafter.
if [ ! -f "$PERSIST_NOTES/index.json" ]; then
    echo "[boot] First deploy — seeding bundled user notes."
    cp -a "$APP_NOTES"/* "$PERSIST_NOTES/" 2>/dev/null || true
else
    echo "[boot] Persistent user notes exist — preserving."
    # Sync any bundled notes that are larger than their persisted version.
    # This catches notes that were empty on volume but got content in the repo.
    for src in "$APP_NOTES"/*.json; do
        [ -f "$src" ] || continue
        fname="$(basename "$src")"
        [ "$fname" = "index.json" ] && continue
        dst="$PERSIST_NOTES/$fname"
        if [ -f "$dst" ]; then
            src_size=$(stat -c%s "$src" 2>/dev/null || stat -f%z "$src" 2>/dev/null || echo 0)
            dst_size=$(stat -c%s "$dst" 2>/dev/null || stat -f%z "$dst" 2>/dev/null || echo 0)
            if [ "$src_size" -gt "$dst_size" ]; then
                echo "[boot] Upgrading note $fname ($dst_size → $src_size bytes)"
                cp -a "$src" "$dst"
            fi
        else
            echo "[boot] New bundled note $fname — copying to persistent volume."
            cp -a "$src" "$dst"
        fi
    done
fi

# 12. Symlink user_notes dir to persistent volume
if [ -L "$APP_NOTES" ]; then
    rm "$APP_NOTES"
elif [ -d "$APP_NOTES" ]; then
    rm -rf "$APP_NOTES"
fi
ln -sfn "$PERSIST_NOTES" "$APP_NOTES"

echo "[boot] User notes linked to persistent volume."

# ── CHAT HISTORY PERSISTENCE ───────────────────────────
PERSIST_CHATS="/persist/chats"
APP_CHATS="/app/app/data/chats"

mkdir -p "$PERSIST_CHATS"

if [ -L "$APP_CHATS" ]; then
    rm "$APP_CHATS"
elif [ -d "$APP_CHATS" ]; then
    # First deploy: migrate any existing chats to persistent volume
    cp -a "$APP_CHATS"/* "$PERSIST_CHATS/" 2>/dev/null || true
    rm -rf "$APP_CHATS"
fi
ln -sfn "$PERSIST_CHATS" "$APP_CHATS"

echo "[boot] Chat history linked to persistent volume."

# ── TRASH PERSISTENCE ──────────────────────────────────
PERSIST_TRASH="/persist/trash"
APP_TRASH="/app/app/data/trash"

mkdir -p "$PERSIST_TRASH"

if [ -L "$APP_TRASH" ]; then
    rm "$APP_TRASH"
elif [ -d "$APP_TRASH" ]; then
    cp -a "$APP_TRASH"/* "$PERSIST_TRASH/" 2>/dev/null || true
    rm -rf "$APP_TRASH"
fi
ln -sfn "$PERSIST_TRASH" "$APP_TRASH"

echo "[boot] Trash linked to persistent volume."

# ── USER DATA PERSISTENCE (per-user email accounts, etc.) ──────
PERSIST_USERS="/persist/users"
APP_USERS="/app/app/data/users"

mkdir -p "$PERSIST_USERS"

if [ -L "$APP_USERS" ]; then
    rm "$APP_USERS"
elif [ -d "$APP_USERS" ]; then
    cp -a "$APP_USERS"/* "$PERSIST_USERS/" 2>/dev/null || true
    rm -rf "$APP_USERS"
fi
ln -sfn "$PERSIST_USERS" "$APP_USERS"

echo "[boot] User data linked to persistent volume."

# 13. Start the app (3 workers — MiniLM model uses ~250MB each, fits in 4GB)
exec python -m uvicorn web.app:app --host 0.0.0.0 --port 8989 --workers 3
