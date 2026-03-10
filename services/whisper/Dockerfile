FROM python:3.11-slim

WORKDIR /app

# System deps for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install faster-whisper and a lightweight OpenAI-compatible server
RUN pip install --no-cache-dir \
    faster-whisper \
    uvicorn \
    fastapi \
    python-multipart \
    && pip cache purge

# Pre-download the tiny model for fast startup
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('tiny', device='cpu', compute_type='int8')" 2>/dev/null || true

# Copy the server script
COPY server.py .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
