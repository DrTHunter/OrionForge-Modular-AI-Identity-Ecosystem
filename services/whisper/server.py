"""
OpenAI-compatible Whisper STT server for OrionForge.
Exposes /v1/audio/transcriptions (same API as OpenAI Whisper).
"""
import io
import tempfile
import os
import logging
import subprocess
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("whisper-stt")

app = FastAPI(title="OrionForge Whisper STT")

# Lazy-load model on first request
_model = None
_model_name = os.environ.get("WHISPER_MODEL", "base")

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
    log.info("[stt] Received %d bytes, filename=%s", len(audio_bytes), file.filename)

    # Write to temp file
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    # Convert webm→wav via ffmpeg for reliable decoding
    wav_path = tmp_path.replace(".webm", ".wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
            capture_output=True, timeout=30,
        )
        use_path = wav_path if os.path.exists(wav_path) and os.path.getsize(wav_path) > 100 else tmp_path
    except Exception:
        use_path = tmp_path

    try:
        m = get_model()
        segments, info = m.transcribe(use_path, language=language, beam_size=5)
        parts = []
        for seg in segments:
            t = seg.text.strip()
            log.info("[stt] segment: prob=%.2f text=%r", seg.no_speech_prob, t)
            # Skip segments that are almost certainly not speech
            if seg.no_speech_prob > 0.8:
                continue
            if t:
                parts.append(t)
        text = " ".join(parts)
        # Filter repetitive hallucinations (e.g. "you you you you")
        words = text.split()
        if len(words) >= 3 and len(set(w.lower().strip(".,!?") for w in words)) == 1:
            log.info("[stt] filtered hallucination: %r", text)
            text = ""
        log.info("[stt] final text: %r (duration=%.1fs)", text, info.duration)
    finally:
        os.unlink(tmp_path)
        if os.path.exists(wav_path):
            os.unlink(wav_path)

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
