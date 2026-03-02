"""Image generation helper — dispatches to multiple AI image providers.

Supported providers:
  OpenAI       — DALL-E 3, DALL-E 2, GPT Image (gpt-image-1)
  Google       — Imagen 3
  Stability AI — Stable Image Ultra, Core, SD3 Large/Turbo/Medium
  Ideogram     — V2, V2 Turbo
  Replicate    — Flux Pro, Flux Schnell, Flux Dev, Playground v2.5
  FAL.ai       — Flux Pro v1.1, Flux Schnell, Flux Dev
  Leonardo AI  — Diffusion XL, Lightning XL, Vision XL, Kino XL
  Banana Dev   — Custom deployed models
  Midjourney   — Via third-party API proxy
"""

import asyncio
import logging

import httpx

log = logging.getLogger("orion.image_gen")


async def _generate_image(provider: str, prompt: str, img_cfg: dict, settings: dict) -> dict:
    """Generate an image using the specified provider.

    Returns dict with ``url``, ``provider``, and optionally ``revised_prompt``
    on success, or dict with ``error`` on failure.
    """

    # ── OpenAI DALL-E 3 ──────────────────────────────────────────
    if provider == "openai_dalle3":
        api_key = img_cfg.get("openai_api_key") or settings.get("api_keys", {}).get("openai", "")
        if not api_key:
            return {"error": "OpenAI API key not set"}
        try:
            async with httpx.AsyncClient(timeout=120) as c:
                r = await c.post(
                    "https://api.openai.com/v1/images/generations",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={"model": "dall-e-3", "prompt": prompt, "n": 1,
                          "size": "1024x1024", "response_format": "url"},
                )
                r.raise_for_status()
            d = r.json()["data"][0]
            return {"url": d["url"], "provider": "openai_dalle3",
                    "revised_prompt": d.get("revised_prompt")}
        except httpx.HTTPStatusError as exc:
            return {"error": f"OpenAI API {exc.response.status_code}: {exc.response.text[:300]}"}
        except Exception as exc:
            return {"error": f"DALL-E 3 error: {exc}"}

    # ── OpenAI DALL-E 2 ──────────────────────────────────────────
    if provider == "openai_dalle2":
        api_key = img_cfg.get("openai_api_key") or settings.get("api_keys", {}).get("openai", "")
        if not api_key:
            return {"error": "OpenAI API key not set"}
        try:
            async with httpx.AsyncClient(timeout=120) as c:
                r = await c.post(
                    "https://api.openai.com/v1/images/generations",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={"model": "dall-e-2", "prompt": prompt, "n": 1,
                          "size": "1024x1024", "response_format": "url"},
                )
                r.raise_for_status()
            d = r.json()["data"][0]
            return {"url": d["url"], "provider": "openai_dalle2"}
        except httpx.HTTPStatusError as exc:
            return {"error": f"OpenAI API {exc.response.status_code}: {exc.response.text[:300]}"}
        except Exception as exc:
            return {"error": f"DALL-E 2 error: {exc}"}

    # ── OpenAI GPT Image (gpt-image-1) ───────────────────────────
    if provider == "openai_gpt_image":
        api_key = img_cfg.get("openai_api_key") or settings.get("api_keys", {}).get("openai", "")
        if not api_key:
            return {"error": "OpenAI API key not set"}
        try:
            async with httpx.AsyncClient(timeout=120) as c:
                r = await c.post(
                    "https://api.openai.com/v1/images/generations",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={"model": "gpt-image-1", "prompt": prompt, "n": 1,
                          "size": "1024x1024", "output_format": "url"},
                )
                r.raise_for_status()
            d = r.json()["data"][0]
            return {"url": d.get("url", ""), "provider": "openai_gpt_image",
                    "revised_prompt": d.get("revised_prompt")}
        except httpx.HTTPStatusError as exc:
            return {"error": f"OpenAI API {exc.response.status_code}: {exc.response.text[:300]}"}
        except Exception as exc:
            return {"error": f"GPT Image error: {exc}"}

    # ── Google Imagen 3 ──────────────────────────────────────────
    if provider == "google_imagen":
        api_key = img_cfg.get("google_api_key") or settings.get("api_keys", {}).get("google_gemini", "")
        if not api_key:
            return {"error": "Google API key not set"}
        try:
            async with httpx.AsyncClient(timeout=120) as c:
                r = await c.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-002:predict?key={api_key}",
                    headers={"Content-Type": "application/json"},
                    json={"instances": [{"prompt": prompt}],
                          "parameters": {"sampleCount": 1, "aspectRatio": "1:1"}},
                )
                r.raise_for_status()
            preds = r.json().get("predictions", [])
            if not preds:
                return {"error": "No image returned from Imagen"}
            b64 = preds[0].get("bytesBase64Encoded", "")
            mime = preds[0].get("mimeType", "image/png")
            return {"url": f"data:{mime};base64,{b64}", "provider": "google_imagen"}
        except httpx.HTTPStatusError as exc:
            return {"error": f"Google API {exc.response.status_code}: {exc.response.text[:300]}"}
        except Exception as exc:
            return {"error": f"Imagen error: {exc}"}

    # ── Stability AI: Ultra / Core / SD3 variants ────────────────
    if provider.startswith("stability_"):
        api_key = img_cfg.get("stability_api_key", "")
        if not api_key:
            return {"error": "Stability AI API key not set"}
        _EP = {
            "stability_ultra": ("https://api.stability.ai/v2beta/stable-image/generate/ultra", {}),
            "stability_core":  ("https://api.stability.ai/v2beta/stable-image/generate/core", {}),
            "stability_sd3_large":       ("https://api.stability.ai/v2beta/stable-image/generate/sd3", {"model": "sd3-large"}),
            "stability_sd3_large_turbo": ("https://api.stability.ai/v2beta/stable-image/generate/sd3", {"model": "sd3-large-turbo"}),
            "stability_sd3_medium":      ("https://api.stability.ai/v2beta/stable-image/generate/sd3", {"model": "sd3-medium"}),
        }
        ep_info = _EP.get(provider)
        if not ep_info:
            return {"error": f"Unknown Stability model: {provider}"}
        url, extra = ep_info
        form_data = {"prompt": prompt, "output_format": "png", **extra}
        try:
            async with httpx.AsyncClient(timeout=120) as c:
                r = await c.post(
                    url,
                    headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
                    data=form_data,
                )
                r.raise_for_status()
            b64 = r.json().get("image", "")
            if not b64:
                return {"error": "No image data in Stability response"}
            return {"url": f"data:image/png;base64,{b64}", "provider": provider}
        except httpx.HTTPStatusError as exc:
            return {"error": f"Stability AI {exc.response.status_code}: {exc.response.text[:300]}"}
        except Exception as exc:
            return {"error": f"Stability AI error: {exc}"}

    # ── Ideogram (V2 and V2 Turbo) ───────────────────────────────
    if provider in ("ideogram", "ideogram_turbo"):
        api_key = img_cfg.get("ideogram_api_key", "")
        if not api_key:
            return {"error": "Ideogram API key not set"}
        model = "V_2_TURBO" if provider == "ideogram_turbo" else "V_2"
        try:
            async with httpx.AsyncClient(timeout=120) as c:
                r = await c.post(
                    "https://api.ideogram.ai/generate",
                    headers={"Api-Key": api_key, "Content-Type": "application/json"},
                    json={"image_request": {"prompt": prompt, "model": model,
                          "magic_prompt_option": "AUTO"}},
                )
                r.raise_for_status()
            images = r.json().get("data", [])
            if not images:
                return {"error": "No image returned from Ideogram"}
            return {"url": images[0].get("url", ""), "provider": provider}
        except httpx.HTTPStatusError as exc:
            return {"error": f"Ideogram API {exc.response.status_code}: {exc.response.text[:300]}"}
        except Exception as exc:
            return {"error": f"Ideogram error: {exc}"}

    # ── Replicate: Flux Pro / Schnell / Dev / Playground v2.5 ────
    if provider.startswith("replicate_"):
        api_key = img_cfg.get("replicate_api_key", "")
        if not api_key:
            return {"error": "Replicate API token not set"}
        _MODELS = {
            "replicate_flux_pro":     "black-forest-labs/flux-pro",
            "replicate_flux_schnell": "black-forest-labs/flux-schnell",
            "replicate_flux_dev":     "black-forest-labs/flux-dev",
            "replicate_playground":   "playgroundai/playground-v2.5-1024px-aesthetic",
        }
        model = _MODELS.get(provider)
        if not model:
            return {"error": f"Unknown Replicate model: {provider}"}
        try:
            async with httpx.AsyncClient(timeout=180) as c:
                cr = await c.post(
                    f"https://api.replicate.com/v1/models/{model}/predictions",
                    headers={"Authorization": f"Bearer {api_key}",
                             "Content-Type": "application/json", "Prefer": "wait"},
                    json={"input": {"prompt": prompt}},
                )
                cr.raise_for_status()
                pred = cr.json()
                if pred.get("status") == "succeeded":
                    out = pred.get("output")
                    if isinstance(out, list):
                        out = out[0] if out else ""
                    return {"url": out or "", "provider": provider}
                pred_url = pred.get("urls", {}).get("get", "")
                for _ in range(90):
                    await asyncio.sleep(2)
                    poll = await c.get(pred_url,
                                       headers={"Authorization": f"Bearer {api_key}"})
                    poll.raise_for_status()
                    p = poll.json()
                    st = p.get("status", "")
                    if st == "succeeded":
                        out = p.get("output")
                        if isinstance(out, list):
                            out = out[0] if out else ""
                        return {"url": out or "", "provider": provider}
                    elif st in ("failed", "canceled"):
                        return {"error": f"Replicate prediction {st}: {p.get('error', '')}"}
                return {"error": "Replicate prediction timed out"}
        except httpx.HTTPStatusError as exc:
            return {"error": f"Replicate API {exc.response.status_code}: {exc.response.text[:300]}"}
        except Exception as exc:
            return {"error": f"Replicate error: {exc}"}

    # ── FAL.ai: Flux Pro v1.1 / Schnell / Dev ───────────────────
    if provider.startswith("fal_"):
        api_key = img_cfg.get("fal_api_key", "")
        if not api_key:
            return {"error": "FAL.ai API key not set"}
        _FMODELS = {
            "fal_flux_pro":     "fal-ai/flux-pro/v1.1",
            "fal_flux_schnell": "fal-ai/flux/schnell",
            "fal_flux_dev":     "fal-ai/flux/dev",
        }
        model_id = _FMODELS.get(provider)
        if not model_id:
            return {"error": f"Unknown FAL model: {provider}"}
        try:
            async with httpx.AsyncClient(timeout=120) as c:
                r = await c.post(
                    f"https://fal.run/{model_id}",
                    headers={"Authorization": f"Key {api_key}",
                             "Content-Type": "application/json"},
                    json={"prompt": prompt, "image_size": "square_hd", "num_images": 1},
                )
                r.raise_for_status()
            images = r.json().get("images", [])
            if not images:
                return {"error": "No image returned from FAL.ai"}
            return {"url": images[0].get("url", ""), "provider": provider}
        except httpx.HTTPStatusError as exc:
            return {"error": f"FAL.ai {exc.response.status_code}: {exc.response.text[:300]}"}
        except Exception as exc:
            return {"error": f"FAL.ai error: {exc}"}

    # ── Leonardo AI: Diffusion XL / Lightning XL / Vision XL / Kino XL
    if provider.startswith("leonardo_"):
        api_key = img_cfg.get("leonardo_api_key", "")
        if not api_key:
            return {"error": "Leonardo AI API key not set"}
        _LMODELS = {
            "leonardo_diffusion_xl":  "1e60896f-3c26-4296-8ecc-53e2afecc132",
            "leonardo_lightning_xl":  "b24e16ff-06e3-43eb-8d33-4416c2d75876",
            "leonardo_vision_xl":     "5c232a9e-9061-4777-980a-ddc8e65647c6",
            "leonardo_kino_xl":       "aa77f04e-3eec-4034-9c07-d0f619684628",
        }
        model_id = _LMODELS.get(provider)
        if not model_id:
            return {"error": f"Unknown Leonardo model: {provider}"}
        try:
            async with httpx.AsyncClient(timeout=180) as c:
                cr = await c.post(
                    "https://cloud.leonardo.ai/api/rest/v1/generations",
                    headers={"Authorization": f"Bearer {api_key}",
                             "Content-Type": "application/json"},
                    json={"prompt": prompt, "modelId": model_id,
                          "width": 1024, "height": 1024, "num_images": 1},
                )
                cr.raise_for_status()
                gen_id = cr.json().get("sdGenerationJob", {}).get("generationId", "")
                if not gen_id:
                    return {"error": "No generation ID from Leonardo"}
                for _ in range(60):
                    await asyncio.sleep(3)
                    poll = await c.get(
                        f"https://cloud.leonardo.ai/api/rest/v1/generations/{gen_id}",
                        headers={"Authorization": f"Bearer {api_key}"},
                    )
                    poll.raise_for_status()
                    gen = poll.json().get("generations_by_pk", {})
                    st = gen.get("status", "")
                    if st == "COMPLETE":
                        imgs = gen.get("generated_images", [])
                        if imgs:
                            return {"url": imgs[0].get("url", ""), "provider": provider}
                        return {"error": "No images in Leonardo response"}
                    elif st == "FAILED":
                        return {"error": "Leonardo generation failed"}
                return {"error": "Leonardo generation timed out"}
        except httpx.HTTPStatusError as exc:
            return {"error": f"Leonardo AI {exc.response.status_code}: {exc.response.text[:300]}"}
        except Exception as exc:
            return {"error": f"Leonardo AI error: {exc}"}

    # ── Banana Dev ───────────────────────────────────────────────
    if provider == "banana":
        api_key = img_cfg.get("banana_api_key", "")
        model_key = img_cfg.get("banana_model_key", "")
        if not api_key or not model_key:
            return {"error": "Banana Dev API key and model key must be set"}
        try:
            async with httpx.AsyncClient(timeout=180) as c:
                sr = await c.post(
                    "https://api.banana.dev/start/v4/",
                    headers={"Content-Type": "application/json"},
                    json={"apiKey": api_key, "modelKey": model_key,
                          "modelInputs": {"prompt": prompt, "num_outputs": 1,
                                          "width": 1024, "height": 1024}},
                )
                sr.raise_for_status()
                call_id = sr.json().get("callID", "")
                if not call_id:
                    return {"error": "No callID from Banana Dev"}
                for _ in range(60):
                    await asyncio.sleep(3)
                    ck = await c.post(
                        "https://api.banana.dev/check/v4/",
                        headers={"Content-Type": "application/json"},
                        json={"apiKey": api_key, "callID": call_id},
                    )
                    ck.raise_for_status()
                    cd = ck.json()
                    msg = cd.get("message", "").lower()
                    if msg == "success":
                        out = cd.get("modelOutputs", [{}])[0]
                        img_url = out.get("image") or out.get("url") or out.get("output", "")
                        if img_url:
                            return {"url": img_url, "provider": "banana"}
                        return {"error": "No image in Banana Dev response"}
                    elif msg == "error":
                        return {"error": f"Banana Dev error: {cd}"}
                return {"error": "Banana Dev prediction timed out"}
        except httpx.HTTPStatusError as exc:
            return {"error": f"Banana Dev {exc.response.status_code}: {exc.response.text[:300]}"}
        except Exception as exc:
            return {"error": f"Banana Dev error: {exc}"}

    # ── Midjourney (via proxy) ───────────────────────────────────
    if provider == "midjourney":
        mj_url = img_cfg.get("midjourney_url", "").rstrip("/")
        mj_key = img_cfg.get("midjourney_api_key", "")
        if not mj_url or not mj_key:
            return {"error": "Midjourney proxy URL and API key must be set"}
        try:
            async with httpx.AsyncClient(timeout=180) as c:
                r = await c.post(
                    f"{mj_url}/v1/imagine",
                    headers={"Authorization": f"Bearer {mj_key}",
                             "Content-Type": "application/json"},
                    json={"prompt": prompt},
                )
                r.raise_for_status()
                data = r.json()
            img_url = (data.get("imageUrl") or data.get("url")
                       or data.get("output", {}).get("imageUrl", ""))
            if not img_url:
                return {"error": "No image URL in Midjourney response"}
            return {"url": img_url, "provider": "midjourney"}
        except httpx.HTTPStatusError as exc:
            return {"error": f"Midjourney API {exc.response.status_code}: {exc.response.text[:300]}"}
        except Exception as exc:
            return {"error": f"Midjourney error: {exc}"}

    return {"error": f"Unknown image provider: {provider}"}
