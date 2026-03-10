"""
OpenAI-compatible Whisper STT server for OrionForge.
Exposes /v1/audio/transcriptions (same API as OpenAI Whisper).
"""
import io
import tempfile
import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

app = FastAPI(title="OrionForge Whisper STT")

# Lazy-load model on first request
_model = None
_model_name = os.environ.get("WHISPER_MODEL", "tiny")

def get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(_model_name, device="cpu", compute_type="int8")
    return _model


@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible models endpoint."""
    return {
        "object": "list",
        "data": [
            {"id": "tiny", "object": "model", "owned_by": "openai"},
            {"id": "base", "object": "model", "owned_by": "openai"},
            {"id": "small", "object": "model", "owned_by": "openai"},
        ]
    }


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("tiny"),
    language: str = Form("en"),
    response_format: str = Form("json"),
):
    """OpenAI-compatible transcription endpoint."""
    audio_bytes = await file.read()

    # Write to temp file (faster-whisper needs a file path)
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        m = get_model()
        segments, info = m.transcribe(tmp_path, language=language, beam_size=5)
        text = " ".join(seg.text.strip() for seg in segments)
    finally:
        os.unlink(tmp_path)

    if response_format == "verbose_json":
        return JSONResponse({
            "text": text,
            "language": info.language,
            "duration": info.duration,
        })

    return JSONResponse({"text": text})


@app.get("/health")
async def health():
    return {"status": "ok"}
