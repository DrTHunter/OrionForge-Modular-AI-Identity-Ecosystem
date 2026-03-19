#!/bin/bash
# â”€â”€ OrionForge Fly.io Boot Script â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Ensures the uploads directory is on the persistent volume so
# avatars and user-uploaded files survive across deploys.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

set -e

PERSIST_UPLOADS="/persist/uploads"
APP_UPLOADS="/app/app/data/uploads"

# 1. Create persistent uploads dir if it doesn't exist
mkdir -p "$PERSIST_UPLOADS"

# 2. Sync bundled default avatars into persistent storage.
#    Remove any existing symlink first so we can access the Docker-bundled files.
if [ -L "$APP_UPLOADS" ]; then
    rm "$APP_UPLOADS"
fi
if [ -d "$APP_UPLOADS" ]; then
    cp -f "$APP_UPLOADS"/* "$PERSIST_UPLOADS/" 2>/dev/null || true
    rm -rf "$APP_UPLOADS"
fi

# 3. Symlink app uploads → persistent volume
ln -sfn "$PERSIST_UPLOADS" "$APP_UPLOADS"

echo "[boot] Uploads directory linked to persistent volume."

# 4. Start the app
exec python -m uvicorn web.app:app --host 0.0.0.0 --port 8989 --workers 2
