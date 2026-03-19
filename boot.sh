#!/bin/bash
# ── OrionForge Fly.io Boot Script ──────────────────────────────
# Ensures the uploads directory is on the persistent volume so
# avatars and user-uploaded files survive across deploys.
# ─────────────────────────────────────────────────────────────────

set -e

PERSIST_UPLOADS="/persist/uploads"
APP_UPLOADS="/app/app/data/uploads"

# 1. Create persistent uploads dir if it doesn't exist
mkdir -p "$PERSIST_UPLOADS"

# 2. Seed: copy any bundled uploads into persistent storage (no overwrite)
#    This handles new default avatars added to the repo.
if [ -d "$APP_UPLOADS" ] && [ ! -L "$APP_UPLOADS" ]; then
    cp -n "$APP_UPLOADS"/* "$PERSIST_UPLOADS/" 2>/dev/null || true
    rm -rf "$APP_UPLOADS"
fi

# 3. Symlink app uploads → persistent volume
ln -sfn "$PERSIST_UPLOADS" "$APP_UPLOADS"

echo "[boot] Uploads directory linked to persistent volume."

# 4. Start the app
exec python -m uvicorn web.app:app --host 0.0.0.0 --port 8989 --workers 2
