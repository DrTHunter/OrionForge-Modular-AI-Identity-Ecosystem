FROM python:3.11-slim

WORKDIR /app

# Install system deps for FAISS (libgomp needed for OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything (flat layout — src/ web/ config/ etc. live side by side)
COPY . .

WORKDIR /app

EXPOSE 8989

CMD ["python", "-m", "uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8989"]
