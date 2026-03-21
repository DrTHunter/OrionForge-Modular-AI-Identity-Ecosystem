#!/bin/bash
# â”€â”€ OrionForge Fly.io Boot Script â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Ensures uploads and memory vault are on the persistent volume so
# avatars, user-uploaded files, and memories survive across deploys.
# â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

set -e

PERSIST_UPLOADS="/persist/uploads"
APP_UPLOADS="/app/app/data/uploads"
BUNDLED_UPLOADS="/app/_bundled_uploads"

PERSIST_MEMORY="/persist/memory"
APP_MEMORY="/app/app/data/memory"
BUNDLED_MEMORY="/app/_bundled_memory"

# 1. Create persistent uploads dir if it doesn't exist
mkdir -p "$PERSIST_UPLOADS"

# 2. Sync bundled avatars into persistent storage.
#    The Dockerfile stages git-tracked uploads to /app/_bundled_uploads
#    so they're always accessible even after the live dir is symlinked.
#    cp -n (no-clobber) adds new files without overwriting user-modified ones.
if [ -d "$BUNDLED_UPLOADS" ]; then
    cp -n "$BUNDLED_UPLOADS"/* "$PERSIST_UPLOADS/" 2>/dev/null || true
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

# 7. Start the app
exec python -m uvicorn web.app:app --host 0.0.0.0 --port 8989 --workers 2
