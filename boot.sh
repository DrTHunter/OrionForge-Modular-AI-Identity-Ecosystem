#!/bin/bash
# â”€â”€ OrionForge Fly.io Boot Script â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Ensures the uploads directory is on the persistent volume so
# avatars and user-uploaded files survive across deploys.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

set -e

PERSIST_UPLOADS="/persist/uploads"
APP_UPLOADS="/app/app/data/uploads"

BUNDLED_UPLOADS="/app/_bundled_uploads"

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

# 4. Start the app
exec python -m uvicorn web.app:app --host 0.0.0.0 --port 8989 --workers 2
