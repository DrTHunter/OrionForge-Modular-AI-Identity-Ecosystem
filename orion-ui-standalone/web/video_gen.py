"""Video generation helper — Google Veo via Generative Language API.

Supported models:
  google_veo2   — veo-2.0-generate-001  (stable, silent 720p)
  google_veo3   — veo-3.0-generate-preview  (audio, 720p)
  google_veo31  — veo-3.1-generate-preview  (audio, portrait, 720p/1080p/4k)

Uses the long-running operations pattern:
  1. POST …/models/{model}:generateVideo  → operation object
  2. Poll GET …/operations/{id}           → until done == True
  3. Download video bytes from Files API  → save to save_dir
  4. Return {"url": "/uploads/videos/…", "provider": …}
"""

import asyncio
import logging
import uuid
from pathlib import Path

import httpx

log = logging.getLogger("orion.video_gen")

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

_VEO_MODELS: dict[str, str] = {
    "google_veo2":  "veo-2.0-generate-001",
    "google_veo3":  "veo-3.0-generate-preview",
    "google_veo31": "veo-3.1-generate-preview",
}

_POLL_INTERVAL = 10   # seconds between status polls
_MAX_POLLS     = 42   # 42 × 10 s = 7 minutes maximum wait


async def _generate_video(
    provider: str,
    prompt: str,
    vid_cfg: dict,
    settings: dict,
    save_dir: Path | None = None,
) -> dict:
    """Generate a video using Google Veo.

    Returns dict with ``url``, ``provider``, ``prompt`` on success,
    or dict with ``error`` on failure.

    ``save_dir`` — directory where the video file is written.
    ``url``      — relative URL path served from the static /uploads mount.
    """
    api_key = (
        vid_cfg.get("google_api_key")
        or settings.get("api_keys", {}).get("google_gemini", "")
    )
    if not api_key:
        return {"error": "Google API key not configured for video generation"}

    model = _VEO_MODELS.get(provider)
    if not model:
        return {"error": f"Unknown Veo provider '{provider}'"}

    aspect_ratio    = vid_cfg.get("aspect_ratio", "16:9")
    duration_secs   = int(vid_cfg.get("duration_seconds", 8))
    person_gen      = "allow_adult"   # safe default; allow_all only for text-to-video

    # ── 1. Submit video generation request ───────────────────────
    payload = {
        "instances": [{"prompt": prompt}],
        "parameters": {
            "aspectRatio":      aspect_ratio,
            "durationSeconds":  duration_secs,
            "personGeneration": person_gen,
            "sampleCount":      1,
        },
    }
    endpoint = f"{_BASE_URL}/models/{model}:predictLongRunning?key={api_key}"

    try:
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(
                endpoint,
                headers={"Content-Type": "application/json"},
                json=payload,
            )
            r.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return {
            "error": (
                f"Veo API {exc.response.status_code}: "
                f"{exc.response.text[:400]}"
            )
        }
    except Exception as exc:
        return {"error": f"Veo request failed: {exc}"}

    op_data = r.json()
    op_name = op_data.get("name", "")
    if not op_name:
        return {"error": "Veo returned no operation name — video generation not started"}

    log.info("[video-gen] Operation started: %s", op_name)

    # ── 2. Poll until done ────────────────────────────────────────
    poll_url = f"{_BASE_URL}/{op_name}?key={api_key}"
    for attempt in range(_MAX_POLLS):
        await asyncio.sleep(_POLL_INTERVAL)
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                poll_r = await c.get(poll_url)
                poll_r.raise_for_status()
        except Exception as exc:
            log.warning("[video-gen] Poll %d/%d error: %s", attempt + 1, _MAX_POLLS, exc)
            continue

        op = poll_r.json()
        if not op.get("done"):
            log.debug("[video-gen] Poll %d/%d — still running", attempt + 1, _MAX_POLLS)
            continue

        # Done — check for API-level error
        if "error" in op:
            err = op["error"]
            return {"error": f"Veo generation failed: {err.get('message', str(err))}"}

        # Parse video URI from response
        response   = op.get("response", {})
        samples    = response.get("generatedSamples", [])
        if not samples:
            return {"error": "Veo returned no video samples in response"}

        video_info = samples[0].get("video", {})
        video_uri  = video_info.get("uri", "")
        mime_type  = video_info.get("mimeType", "video/mp4")

        if not video_uri:
            return {"error": "Veo response contained no video URI"}

        log.info("[video-gen] Video ready — downloading from %s", video_uri[:80])

        # ── 3. Download video bytes ───────────────────────────────
        download_url = (
            video_uri
            if "?" in video_uri
            else f"{video_uri}?alt=media&key={api_key}"
        )
        # Ensure API key is included for Files API downloads
        if "key=" not in download_url:
            download_url += f"&key={api_key}"

        try:
            async with httpx.AsyncClient(timeout=180, follow_redirects=True) as c:
                dl = await c.get(download_url)
                dl.raise_for_status()
            video_bytes = dl.content
        except Exception as exc:
            return {"error": f"Failed to download Veo video: {exc}"}

        if len(video_bytes) < 1024:
            return {"error": "Veo video download returned unexpectedly small file"}

        # ── 4. Save to disk ───────────────────────────────────────
        if save_dir is None:
            save_dir = Path("/tmp/veo_videos")
        save_dir.mkdir(parents=True, exist_ok=True)

        ext      = ".mp4" if "mp4" in mime_type else ".webm"
        filename = f"veo_{uuid.uuid4().hex[:14]}{ext}"
        (save_dir / filename).write_bytes(video_bytes)

        log.info(
            "[video-gen] Saved %d bytes → %s/%s",
            len(video_bytes), save_dir, filename,
        )
        return {
            "url":      f"/uploads/videos/{filename}",
            "provider": provider,
            "prompt":   prompt,
        }

    return {"error": "Veo video generation timed out — please try again"}
