FROM python:3.11-slim

WORKDIR /app

# Install system deps for FAISS (libgomp needed for OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy engine and UI
COPY engine/ ./engine/
COPY ui/ ./ui/

# The engine src/ must be importable from the UI's working directory
RUN ln -s /app/engine/src /app/ui/src

WORKDIR /app/ui

EXPOSE 8989

CMD ["python", "-m", "uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8989"]
