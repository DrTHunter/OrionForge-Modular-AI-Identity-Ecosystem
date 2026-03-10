#!/bin/sh
# Start Ollama server in the background, pull the default model, then keep running

echo "[OrionForge Ollama] Starting server..."
ollama serve &
SERVE_PID=$!

# Wait for Ollama to become ready using the ollama CLI (no curl needed)
echo "[OrionForge Ollama] Waiting for server to be ready..."
READY=0
for i in $(seq 1 30); do
    if ollama list > /dev/null 2>&1; then
        echo "[OrionForge Ollama] Server is ready."
        READY=1
        break
    fi
    sleep 2
done

if [ "$READY" = "1" ]; then
    MODEL="${OLLAMA_DEFAULT_MODEL:-llama3.2:3b}"
    if ollama list 2>/dev/null | grep -q "${MODEL}"; then
        echo "[OrionForge Ollama] Model '${MODEL}' already cached."
    else
        echo "[OrionForge Ollama] Pulling model '${MODEL}'..."
        ollama pull "${MODEL}" && echo "[OrionForge Ollama] Model ready." || echo "[OrionForge Ollama] WARNING: pull failed, continuing anyway."
    fi
else
    echo "[OrionForge Ollama] Server did not become ready in time, continuing without model pull."
fi

# Keep container alive by waiting on the server process
wait $SERVE_PID
