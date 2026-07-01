"""Stripe Subscription — Checkout, webhooks, tier gating, credits, and LLM markup.

Implements a Pro-only ($9.99/mo) paywall with 5-day free trial:
  - New users get 5 days of full access to all agents and tools
  - During trial: LLM usage via platform keys costs credits at 2×
  - After trial, users must subscribe to Pro ($9.99/mo) to continue
  - Premium tools (AGI Loop, Email, Voice) cost credits
  - Users who provide their own API keys get free LLM usage

Flow:
  1. User signs up via Supabase → 5-day trial starts automatically
  2. During trial: unlimited agents & tools, LLM costs credits at 2×
  3. After trial: must subscribe to Pro ($9.99/mo) to continue
  4. Premium tools deduct credits; buy credit packs via Stripe
  5. LLM calls bill at 2× when using platform-hosted keys
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger("soulscript.stripe")

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

# Use persistent volume on Fly.io if available, otherwise fall back to config/
_PERSIST_DIR = Path("/persist")
_STRIPE_STATE_FILE = (
    _PERSIST_DIR / "stripe_state.json"
    if _PERSIST_DIR.is_dir()
    else _CONFIG_DIR / "stripe_state.json"
)

# ── Stripe keys from environment (never hardcode) ────────────────
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_PUBLISHABLE_KEY = os.environ.get("STRIPE_PUBLISHABLE_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRO_PRICE_ID = os.environ.get("STRIPE_PRO_PRICE_ID", "")

# Stripe product that all dynamically-priced credit-pack checkouts roll up under.
# Live mode defaults to the Orion Forge "Credits" product. Test mode falls back
# to ad-hoc product_data so local Stripe test keys do not require a mirrored
# product to exist in the test account.
DEFAULT_LIVE_CREDITS_PRODUCT_ID = "prod_UktYCqIzyVKRzb"


def _default_credits_product_id(stripe_secret_key: str) -> str:
    """Return the default Stripe product to use for credit packs.

    Test keys should not inherit the live product id because Stripe products are
    account-scoped and a live product id is invalid in test mode.
    """
    key = (stripe_secret_key or "").strip()
    if key.startswith("sk_live_"):
        return DEFAULT_LIVE_CREDITS_PRODUCT_ID
    return ""


STRIPE_CREDITS_PRODUCT_ID = os.environ.get(
    "STRIPE_CREDITS_PRODUCT_ID",
    _default_credits_product_id(STRIPE_SECRET_KEY),
)

# ── Tier definitions ─────────────────────────────────────────────
FREE_TIER_FEATURES = {
    "chat", "profiles", "knowledge", "vault", "tools",
    "skins_default", "settings", "memory_basic",
}

PRO_TIER_FEATURES = FREE_TIER_FEATURES | {
    "agi_loop", "cost_tracker", "skins_premium", "priority_routing",
    "memory_unlimited", "image_generation", "internet_tools",
    "voice_tts", "voice_stt", "advanced_tools", "api_access",
}

TIER_INFO = {
    "free": {
        "name": "Free",
        "price": 0,
        "features": sorted(FREE_TIER_FEATURES),
        "limits": {
            "messages_per_day": 50,
            "memory_entries": 100,
            "agents": 3,
            "knowledge_files": 5,
        },
    },
    "pro": {
        "name": "Pro",
        "price": 9.99,
        "price_label": "$9.99/mo",
        "features": sorted(PRO_TIER_FEATURES),
        "limits": {
            "messages_per_day": -1,  # unlimited
            "memory_entries": -1,
            "agents": -1,
            "knowledge_files": -1,
        },
    },
}

# ── Free Trial ───────────────────────────────────────────────────
FREE_TRIAL_DAYS = 5
_SECONDS_PER_DAY = 86400


# ── State persistence ────────────────────────────────────────────
_stripe_state_cache: dict | None = None
_stripe_state_cache_ts: float = 0.0
_STRIPE_CACHE_TTL = 30.0  # seconds

def _load_stripe_state() -> dict:
    """Load subscription state from disk (cached for 30s)."""
    global _stripe_state_cache, _stripe_state_cache_ts
    now = time.monotonic()
    if _stripe_state_cache is not None and (now - _stripe_state_cache_ts) < _STRIPE_CACHE_TTL:
        return _stripe_state_cache
    if _STRIPE_STATE_FILE.exists():
        try:
            with open(_STRIPE_STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                _stripe_state_cache = data
                _stripe_state_cache_ts = now
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {"subscriptions": {}}


def _save_stripe_state(state: dict):
    """Persist subscription state and invalidate cache."""
    global _stripe_state_cache, _stripe_state_cache_ts
    _STRIPE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_STRIPE_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    _stripe_state_cache = state
    _stripe_state_cache_ts = time.monotonic()


def _ensure_trial_start(user_id: str) -> float:
    """Record trial start time for a user if not already set. Returns the start timestamp."""
    state = _load_stripe_state()
    trials = state.setdefault("trials", {})
    if user_id not in trials:
        trials[user_id] = {"started_at": time.time()}
        _save_stripe_state(state)
        log.info("[trial] Started %d-day free trial for user %s", FREE_TRIAL_DAYS, user_id)
    return trials[user_id]["started_at"]


def get_trial_status(user_id: str) -> dict:
    """Get a user's free trial status.

    Returns:
        {"active": bool, "days_left": int, "started_at": float, "expires_at": float}
    """
    if not user_id:
        return {"active": False, "days_left": 0, "started_at": 0, "expires_at": 0}
    started_at = _ensure_trial_start(user_id)
    expires_at = started_at + (FREE_TRIAL_DAYS * _SECONDS_PER_DAY)
    remaining = expires_at - time.time()
    days_left = max(0, int(remaining / _SECONDS_PER_DAY) + (1 if remaining % _SECONDS_PER_DAY > 0 else 0))
    return {
        "active": remaining > 0,
        "days_left": days_left,
        "started_at": started_at,
        "expires_at": expires_at,
    }


def get_user_tier(user_id: str) -> str:
    """Get the access tier for a user.

    The platform is pay-per-use with no subscription paywall: every
    authenticated user has full ('pro') access and is billed in credits
    (2× the API cost) per request. Returns 'free' only for an anonymous
    (empty) user id.
    """
    if not user_id:
        return "free"
    return "pro"


def get_user_subscription(user_id: str) -> dict:
    """Get full subscription info for a user (includes trial status)."""
    state = _load_stripe_state()
    sub = state.get("subscriptions", {}).get(user_id, {})
    has_sub = sub.get("status") in ("active", "trialing")
    trial = get_trial_status(user_id)
    tier = get_user_tier(user_id)
    return {
        "tier": tier,
        "tier_info": TIER_INFO[tier],
        "subscription": sub if sub else None,
        "stripe_configured": bool(STRIPE_SECRET_KEY),
        "trial": trial,
        "is_trial": trial["active"] and not has_sub,
    }


def set_user_subscription(user_id: str, subscription_data: dict):
    """Store/update a user's subscription record."""
    state = _load_stripe_state()
    if "subscriptions" not in state:
        state["subscriptions"] = {}
    state["subscriptions"][user_id] = {
        **subscription_data,
        "updated_at": time.time(),
    }
    _save_stripe_state(state)
    log.info("[stripe] Updated subscription for user %s: %s", user_id, subscription_data.get("status"))


def cancel_user_subscription(user_id: str):
    """Mark a user's subscription as canceled."""
    state = _load_stripe_state()
    sub = state.get("subscriptions", {}).get(user_id, {})
    if sub:
        sub["status"] = "canceled"
        sub["canceled_at"] = time.time()
        _save_stripe_state(state)
        log.info("[stripe] Canceled subscription for user %s", user_id)


# ── Stripe API helpers ───────────────────────────────────────────
def _get_stripe():
    """Lazy-import stripe to avoid hard dependency."""
    try:
        import stripe
        stripe.api_key = STRIPE_SECRET_KEY
        return stripe
    except ImportError:
        log.error("[stripe] stripe package not installed. Run: pip install stripe")
        return None


def create_checkout_session(user_id: str, user_email: str, success_url: str, cancel_url: str) -> Optional[dict]:
    """Create a Stripe Checkout session for Pro subscription."""
    stripe = _get_stripe()
    if not stripe or not STRIPE_SECRET_KEY:
        return {"error": "Stripe not configured"}

    if not STRIPE_PRO_PRICE_ID:
        return {"error": "No Pro price ID configured. Set STRIPE_PRO_PRICE_ID env var."}

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            payment_method_types=["card"],
            line_items=[{
                "price": STRIPE_PRO_PRICE_ID,
                "quantity": 1,
            }],
            success_url=success_url,
            cancel_url=cancel_url,
            client_reference_id=user_id,
            customer_email=user_email,
            metadata={"user_id": user_id},
        )
        return {"url": session.url, "session_id": session.id}
    except Exception as exc:
        log.error("[stripe] Checkout creation failed: %s", exc)
        return {"error": str(exc)}


def create_billing_portal_session(user_id: str, return_url: str) -> Optional[dict]:
    """Create a Stripe Billing Portal session for subscription management."""
    stripe = _get_stripe()
    if not stripe or not STRIPE_SECRET_KEY:
        return {"error": "Stripe not configured"}

    state = _load_stripe_state()
    sub = state.get("subscriptions", {}).get(user_id, {})
    customer_id = sub.get("customer_id")

    if not customer_id:
        return {"error": "No active subscription found"}

    try:
        session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=return_url,
        )
        return {"url": session.url}
    except Exception as exc:
        log.error("[stripe] Portal creation failed: %s", exc)
        return {"error": str(exc)}


def handle_webhook_event(payload: bytes, sig_header: str) -> dict:
    """Process a Stripe webhook event."""
    stripe = _get_stripe()
    if not stripe:
        return {"error": "Stripe not available"}

    try:
        if STRIPE_WEBHOOK_SECRET:
            event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
            if hasattr(event, "to_dict"):
                event = event.to_dict()
        else:
            event = json.loads(payload)
            log.warning("[stripe] Webhook signature verification skipped (no secret)")
    except Exception as exc:
        log.error("[stripe] Webhook verification failed: %s", exc)
        return {"error": f"Webhook verification failed: {exc}"}

    event_type = event.get("type", "")
    data = event.get("data", {}).get("object", {})

    if event_type == "checkout.session.completed":
        metadata = data.get("metadata", {})
        user_id = data.get("client_reference_id") or metadata.get("user_id")

        # Credit pack purchase — idempotent so Stripe webhook retries and the
        # success-page fallback can never grant the same purchase twice.
        if metadata.get("type") == "credits":
            grant = _grant_credits_for_checkout(data)
            return {
                "ok": bool(grant.get("ok")),
                "action": "credits_purchased",
                "user_id": grant.get("user_id", user_id),
                "credits": grant.get("credits", 0),
                "already_fulfilled": grant.get("already_fulfilled", False),
            }

        # Subscription purchase
        if user_id:
            set_user_subscription(user_id, {
                "status": "active",
                "subscription_id": data.get("subscription"),
                "customer_id": data.get("customer"),
                "email": data.get("customer_email", ""),
                "plan": "pro",
            })
            return {"ok": True, "action": "subscription_activated", "user_id": user_id}

    elif event_type in ("customer.subscription.updated", "customer.subscription.deleted"):
        sub_status = data.get("status", "")
        customer_id = data.get("customer", "")
        # Look up user by customer ID
        state = _load_stripe_state()
        for uid, sub in state.get("subscriptions", {}).items():
            if sub.get("customer_id") == customer_id:
                sub["status"] = sub_status
                if sub_status in ("canceled", "unpaid", "past_due"):
                    sub["canceled_at"] = time.time()
                _save_stripe_state(state)
                return {"ok": True, "action": f"subscription_{sub_status}", "user_id": uid}

    elif event_type == "invoice.payment_failed":
        customer_id = data.get("customer", "")
        state = _load_stripe_state()
        for uid, sub in state.get("subscriptions", {}).items():
            if sub.get("customer_id") == customer_id:
                sub["status"] = "past_due"
                _save_stripe_state(state)
                return {"ok": True, "action": "payment_failed", "user_id": uid}

    log.info("[stripe] Processed webhook: %s", event_type)
    return {"ok": True, "event_type": event_type}


def _grant_credits_for_checkout(session: dict) -> dict:
    """Idempotently grant the credits for a completed Checkout Session.

    Shared by the Stripe webhook and the post-checkout success page so credits
    are granted exactly once per Stripe checkout session — safe against webhook
    retries and a user reloading the success page.
    """
    metadata = session.get("metadata", {}) or {}
    if metadata.get("type") != "credits":
        return {"ok": False, "reason": "not_a_credit_purchase"}

    user_id = session.get("client_reference_id") or metadata.get("user_id")
    session_id = session.get("id", "") or ""
    credits = int(metadata.get("credits", 0) or 0)
    pack_id = metadata.get("pack_id", "unknown")
    if not user_id or credits <= 0:
        return {"ok": False, "reason": "missing_user_or_credits", "user_id": user_id}

    state = _load_stripe_state()
    fulfilled = state.setdefault("fulfilled_sessions", {})
    if session_id and session_id in fulfilled:
        return {"ok": True, "already_fulfilled": True, "user_id": user_id, "credits": credits}

    # Grant the credits and record the session id in ONE atomic state write so a
    # concurrent webhook retry / success-page call cannot double-credit.
    credits_state = state.setdefault("credits", {})
    bucket = credits_state.setdefault(user_id, {"balance": 0, "history": []})
    bucket["balance"] += credits
    bucket["history"].append({
        "type": "credit",
        "amount": credits,
        "reason": f"purchase:{pack_id}",
        "timestamp": time.time(),
    })
    bucket["history"] = bucket["history"][-200:]
    if session_id:
        fulfilled[session_id] = {"user_id": user_id, "credits": credits, "ts": time.time()}
        # Bound the ledger so the state file can't grow without limit.
        if len(fulfilled) > 1000:
            for stale in list(fulfilled.keys())[:-1000]:
                del fulfilled[stale]
    _save_stripe_state(state)
    log.info("[credits] +%d credits for user %s (purchase:%s, session=%s). Balance: %d",
             credits, user_id, pack_id, session_id or "n/a", bucket["balance"])
    return {"ok": True, "user_id": user_id, "credits": credits}


def fulfill_credits_for_session(session_id: str, expected_user_id: str = "") -> dict:
    """Verify a Checkout Session is paid and idempotently grant its credits.

    Fallback for the post-checkout success page so credits are delivered even
    when the Stripe webhook is delayed or misconfigured. Idempotent with the
    webhook via the shared ``fulfilled_sessions`` ledger.
    """
    if not session_id:
        return {"ok": False, "reason": "no_session"}
    stripe = _get_stripe()
    if not stripe or not STRIPE_SECRET_KEY:
        return {"ok": False, "reason": "stripe_not_configured"}
    try:
        session = stripe.checkout.Session.retrieve(session_id)
        if hasattr(session, "to_dict"):
            session = session.to_dict()
    except Exception as exc:
        log.error("[stripe] Session retrieve failed for %s: %s", session_id, exc)
        return {"ok": False, "reason": "retrieve_failed", "error": str(exc)}

    if session.get("payment_status") != "paid":
        return {"ok": False, "reason": "not_paid", "payment_status": session.get("payment_status")}

    # Only fulfill a session that belongs to the requesting user.
    metadata = session.get("metadata", {}) or {}
    owner = session.get("client_reference_id") or metadata.get("user_id")
    if expected_user_id and owner and owner != expected_user_id:
        return {"ok": False, "reason": "user_mismatch"}

    return _grant_credits_for_checkout(session)


# ═══════════════════════════════════════════════════════════════════
#  CREDIT SYSTEM — per-tool metered billing
# ═══════════════════════════════════════════════════════════════════

# ── Per-use credit costs (only for tools that charge per-use) ────
# Web Search & Image Generation are FREE and don't appear here.
TOOL_CREDIT_COSTS = {}

# Credit packs users can purchase ($5, $10, $20, $30)
CREDIT_PACKS = {
    "pack_5":   {"credits": 500,   "price":  5.00, "label": "500 credits",    "price_label": "$5",   "bonus": ""},
    "pack_10":  {"credits": 1000,  "price": 10.00, "label": "1,000 credits",  "price_label": "$10",  "bonus": ""},
    "pack_20":  {"credits": 2100,  "price": 20.00, "label": "2,100 credits",  "price_label": "$20",  "bonus": "+100 bonus"},
    "pack_30":  {"credits": 3200,  "price": 30.00, "label": "3,200 credits",  "price_label": "$30",  "bonus": "+200 bonus"},
}

# LLM markup multiplier: users pay 2× the actual token cost when using platform keys.
# This is the single source of truth for LLM, TTS, STT, image, and video credit markup.
LLM_MARKUP_MULTIPLIER = 2.0

# Fallback price (USD per 1M tokens) used ONLY for platform-hosted models that are
# missing from pricing.yaml, so unpriced models are never billed as $0 (free).
UNPRICED_LLM_FALLBACK_PER_1M = 15.0

# ── Voice API pricing (USD) ──────────────────────────────────────
# TTS cost per 1,000 characters by provider
TTS_COST_PER_1K_CHARS = {
    "elevenlabs": 0.30,    # ElevenLabs standard tier
    "inworld":    0.005,   # Inworld TTS-1.5 Mini ($5/1M chars)
    "inworld-hd": 0.010,   # Inworld TTS-1.5 Max ($10/1M chars)
    "edge-tts":   0.015,   # Legacy Edge-TTS / Piper
    "default":    0.03,
}

# Premium voice surcharge per 1,000 characters (on top of base rate)
TTS_PREMIUM_SURCHARGE_PER_1K = 0.20

# STT cost per minute of audio by provider
STT_COST_PER_MINUTE = {
    "whisper":     0.006,   # OpenAI Whisper pricing
    "elevenlabs":  0.007,   # ElevenLabs Scribe (~$0.42/hr)
    "default":     0.006,
}

# ── Image generation pricing (USD per generated image) by provider ──
# Image generation always uses platform keys, so it is always metered.
# Keys are matched by prefix (e.g. "stability_ultra" → "stability").
IMAGE_COST_PER_IMAGE = {
    "openai_dalle3":    0.04,   # DALL·E 3 standard 1024²
    "openai_dalle2":    0.02,
    "openai_gpt_image": 0.04,   # gpt-image-1 (medium quality)
    "google_imagen":    0.04,   # Imagen 3/4
    "stability":        0.04,   # Stable Image Ultra/Core/SD3
    "ideogram":         0.08,   # Ideogram V2 / Turbo
    "replicate":        0.02,   # Flux family via Replicate
    "fal":              0.05,   # Flux family via FAL
    "leonardo":         0.02,   # Leonardo XL family
    "midjourney":       0.05,
    "default":          0.05,
}

# ── Video generation pricing (USD per second of output) by provider ──
# Veo is billed per second and is by far the most expensive media op.
VIDEO_COST_PER_SECOND = {
    "google_veo2":  0.35,   # Veo 2 (silent 720p)
    "google_veo3":  0.75,   # Veo 3 (audio, 720p)
    "google_veo31": 0.40,   # Veo 3.1 (audio)
    "default":      0.50,
}

# Skin catalog — cosmetic UI themes, free for every user
# "default" is free; all others cost 75 credits ($0.75)
SKIN_PRICES = {
    "default":         0,
    "cyberpunk_neon":  75,
    "retro_terminal":  75,
    "dark_forest":     75,
    "paper_white":     75,
    "midnight_ocean":  75,
    "blood_moon":      75,
    "aurora_borealis": 75,
    "solarized_dark":  75,
    "frost_glass":     75,
    "synthwave_84":    75,
    "dracula":         75,
    "neon_abyss":      75,
}


def estimate_llm_credit_cost(usd_cost: float) -> int:
    """Convert a USD token cost to credits at 2× markup.

    1 credit ≈ $0.01 base value. With 2× markup:
      $0.01 actual cost → 2 credits
    """
    if usd_cost <= 0:
        return 0
    # Convert USD to credits: $0.01 = 1 credit base
    # Apply the LLM markup multiplier, then round up to whole credits
    credits = int((usd_cost * 100) * LLM_MARKUP_MULTIPLIER + 0.99)  # round up
    return max(credits, 1)  # minimum 1 credit


def estimate_tts_credit_cost(char_count: int, provider: str = "elevenlabs", premium: bool = False) -> int:
    """Convert TTS character count to credits at 2× markup.

    Uses provider-specific per-1K-char pricing, then applies the same
    2× markup as LLM usage.  Returns 0 for zero-length text.
    Premium voices add an extra surcharge per 1K chars.
    """
    if char_count <= 0:
        return 0
    rate = TTS_COST_PER_1K_CHARS.get(provider, TTS_COST_PER_1K_CHARS["default"])
    if premium:
        rate += TTS_PREMIUM_SURCHARGE_PER_1K
    usd_cost = (char_count / 1000) * rate
    return estimate_llm_credit_cost(usd_cost)


def estimate_stt_credit_cost(audio_seconds: float, provider: str = "whisper") -> int:
    """Convert STT audio duration (seconds) to credits at 2× markup.

    Uses provider-specific per-minute pricing, then applies the same
    2× markup as LLM usage.  Returns 0 for zero-length audio.
    """
    if audio_seconds <= 0:
        return 0
    rate = STT_COST_PER_MINUTE.get(provider, STT_COST_PER_MINUTE["default"])
    usd_cost = (audio_seconds / 60) * rate
    return estimate_llm_credit_cost(usd_cost)


def estimate_llm_credit_cost_safe(usd_cost: float, total_tokens: int = 0) -> int:
    """Like estimate_llm_credit_cost, but never returns 0 when tokens were used.

    Protects the operator from models that are missing from pricing.yaml
    (which compute a $0 cost and would otherwise bill nothing).  When the
    metered USD cost is 0 but tokens were consumed, falls back to a
    conservative per-token rate so platform usage is always charged.
    """
    if usd_cost and usd_cost > 0:
        return estimate_llm_credit_cost(usd_cost)
    if total_tokens and total_tokens > 0:
        fallback_usd = total_tokens * UNPRICED_LLM_FALLBACK_PER_1M / 1_000_000
        return estimate_llm_credit_cost(fallback_usd)
    return 0


def estimate_image_credit_cost(provider: str = "default") -> int:
    """Credits charged for one generated image, at the platform markup.

    Image generation always uses the platform's provider keys, so every
    image a user generates costs the operator money.  Provider keys are
    matched by prefix (e.g. "stability_ultra" → "stability").
    """
    p = provider or "default"
    rate = IMAGE_COST_PER_IMAGE.get(p)
    if rate is None:
        for key, val in IMAGE_COST_PER_IMAGE.items():
            if key != "default" and p.startswith(key):
                rate = val
                break
    if rate is None:
        rate = IMAGE_COST_PER_IMAGE["default"]
    return estimate_llm_credit_cost(rate)


def estimate_video_credit_cost(provider: str = "default", duration_seconds: int = 8) -> int:
    """Credits charged for one generated video, at the platform markup.

    Veo video generation is billed per second of output and is by far the
    most expensive media operation, so it is always metered.
    """
    rate = VIDEO_COST_PER_SECOND.get(provider, VIDEO_COST_PER_SECOND["default"])
    try:
        secs = max(1, int(duration_seconds))
    except (TypeError, ValueError):
        secs = 8
    usd_cost = rate * secs
    return estimate_llm_credit_cost(usd_cost)


def get_user_credits(user_id: str) -> int:
    """Get the credit balance for a user."""
    if not user_id:
        return 0
    state = _load_stripe_state()
    return state.get("credits", {}).get(user_id, {}).get("balance", 0)


def add_user_credits(user_id: str, amount: int, reason: str = "purchase"):
    """Add credits to a user's balance."""
    state = _load_stripe_state()
    if "credits" not in state:
        state["credits"] = {}
    if user_id not in state["credits"]:
        state["credits"][user_id] = {"balance": 0, "history": []}
    state["credits"][user_id]["balance"] += amount
    state["credits"][user_id]["history"].append({
        "type": "credit",
        "amount": amount,
        "reason": reason,
        "timestamp": time.time(),
    })
    # Keep last 200 history entries
    state["credits"][user_id]["history"] = state["credits"][user_id]["history"][-200:]
    _save_stripe_state(state)
    log.info("[credits] +%d credits for user %s (%s). Balance: %d",
             amount, user_id, reason, state["credits"][user_id]["balance"])


def deduct_user_credits(user_id: str, amount: int, tool_name: str) -> dict:
    """Deduct credits for a tool use. Returns {ok, balance} or {error}."""
    state = _load_stripe_state()
    if "credits" not in state:
        state["credits"] = {}
    if user_id not in state["credits"]:
        state["credits"][user_id] = {"balance": 0, "history": []}

    balance = state["credits"][user_id]["balance"]
    if balance < amount:
        return {
            "error": f"Insufficient credits. Need {amount}, have {balance}.",
            "balance": balance,
            "needed": amount,
        }

    state["credits"][user_id]["balance"] -= amount
    state["credits"][user_id]["history"].append({
        "type": "debit",
        "amount": -amount,
        "tool": tool_name,
        "timestamp": time.time(),
    })
    state["credits"][user_id]["history"] = state["credits"][user_id]["history"][-200:]
    _save_stripe_state(state)
    new_balance = state["credits"][user_id]["balance"]
    log.info("[credits] -%d credits for user %s (tool: %s). Balance: %d",
             amount, user_id, tool_name, new_balance)
    return {"ok": True, "balance": new_balance}


def get_credit_history(user_id: str, limit: int = 50) -> list:
    """Get recent credit transaction history for a user."""
    state = _load_stripe_state()
    history = state.get("credits", {}).get(user_id, {}).get("history", [])
    return list(reversed(history[-limit:]))


def user_has_purchased_credits(user_id: str) -> bool:
    """Whether a user has ever bought a credit pack (vs. only the free welcome grant).

    Used to decide if a user is still spending their free "trial" balance. Returns
    True once any credit-history entry has a ``purchase`` reason (set by the Stripe
    webhook on a successful credit-pack checkout).
    """
    if not user_id:
        return False
    state = _load_stripe_state()
    history = state.get("credits", {}).get(user_id, {}).get("history", [])
    return any(
        entry.get("type") == "credit"
        and str(entry.get("reason", "")).startswith("purchase")
        for entry in history
    )


def create_credits_checkout_session(
    user_id: str, user_email: str, pack_id: str,
    success_url: str, cancel_url: str
) -> dict:
    """Create a Stripe Checkout session for purchasing a credit pack."""
    stripe = _get_stripe()
    if not stripe or not STRIPE_SECRET_KEY:
        return {"error": "Stripe not configured"}

    pack = CREDIT_PACKS.get(pack_id)
    if not pack:
        return {"error": f"Unknown credit pack: {pack_id}"}

    # Build a dynamic price. When a credits product is configured, every pack rolls
    # up under that one Stripe product (clean reporting) while the amount stays
    # driven by CREDIT_PACKS; otherwise fall back to an ad-hoc product per checkout.
    price_data = {
        "currency": "usd",
        "unit_amount": int(pack["price"] * 100),  # cents
    }
    if STRIPE_CREDITS_PRODUCT_ID:
        price_data["product"] = STRIPE_CREDITS_PRODUCT_ID
    else:
        price_data["product_data"] = {
            "name": f"SoulScript Credits — {pack['label']}",
            "description": f"{pack['credits']} credits for premium tool usage",
        }

    try:
        session = stripe.checkout.Session.create(
            mode="payment",
            payment_method_types=["card"],
            line_items=[{
                "price_data": price_data,
                "quantity": 1,
            }],
            success_url=success_url,
            cancel_url=cancel_url,
            client_reference_id=user_id,
            customer_email=user_email,
            metadata={
                "user_id": user_id,
                "type": "credits",
                "pack_id": pack_id,
                "credits": str(pack["credits"]),
            },
        )
        return {"url": session.url, "session_id": session.id}
    except Exception as exc:
        log.error("[stripe] Credit checkout creation failed: %s", exc)
        return {"error": str(exc)}


# ═══════════════════════════════════════════════════════════════════
#  USER ACTIVITY TRACKING & ACCOUNT LIFECYCLE
# ═══════════════════════════════════════════════════════════════════

# How many days of inactivity before an account is purged
INACTIVE_ACCOUNT_DAYS = 365

# Throttle activity writes — only update once per hour to avoid disk thrash
_ACTIVITY_WRITE_INTERVAL = 3600


def touch_user_activity(user_id: str):
    """Record that a user was active. Throttled to once per hour."""
    if not user_id:
        return
    state = _load_stripe_state()
    activity = state.setdefault("activity", {})
    now = time.time()
    last = activity.get(user_id, 0)
    if now - last < _ACTIVITY_WRITE_INTERVAL:
        return  # already recorded recently
    activity[user_id] = now
    _save_stripe_state(state)


def get_user_last_active(user_id: str) -> float:
    """Return the last-active timestamp for a user (0 if never tracked)."""
    state = _load_stripe_state()
    return state.get("activity", {}).get(user_id, 0)


def wipe_user_data(user_id: str, keep_purchases: bool = True) -> dict:
    """Remove billing/state data for a single user from stripe_state.json.

    Clears: subscriptions, trials, credits, activity.
    Purchases are preserved by default (keep_purchases=True) since they
    represent real money spent and should persist indefinitely.
    Returns a summary of what was removed.
    """
    state = _load_stripe_state()
    removed = {}
    sections = ["subscriptions", "trials", "credits", "activity"]
    if not keep_purchases:
        sections.append("purchases")
    for key in sections:
        bucket = state.get(key, {})
        if user_id in bucket:
            removed[key] = True
            del bucket[user_id]
    _save_stripe_state(state)
    log.info("[account] Wiped data for user %s. Removed: %s (purchases kept: %s)",
             user_id, list(removed.keys()), keep_purchases)
    return {"ok": True, "user_id": user_id, "removed_sections": list(removed.keys()),
            "purchases_kept": keep_purchases}


def wipe_user_by_email(email: str) -> dict:
    """Find and wipe ALL users matching the given email.

    Searches subscriptions for an email match, then wipes every matching UUID.
    Also wipes orphaned entries (UUIDs with no subscription but present in
    trials/credits/purchases/activity).
    """
    email_lower = email.strip().lower()
    state = _load_stripe_state()
    matched_ids = set()

    # Find UUIDs by email in subscriptions
    for uid, sub in state.get("subscriptions", {}).items():
        if sub.get("email", "").lower() == email_lower:
            matched_ids.add(uid)

    if not matched_ids:
        return {"ok": False, "error": f"No user found with email {email}"}

    results = []
    for uid in matched_ids:
        results.append(wipe_user_data(uid))

    return {"ok": True, "email": email, "wiped_users": len(matched_ids), "results": results}


def purge_inactive_users(days: int = INACTIVE_ACCOUNT_DAYS) -> dict:
    """Remove billing data for users inactive longer than *days*.

    Skips users with an active Stripe subscription.
    Returns summary of purged user count.
    """
    state = _load_stripe_state()
    cutoff = time.time() - (days * _SECONDS_PER_DAY)
    activity = state.get("activity", {})

    # Collect all known user IDs across every section
    all_uids: set[str] = set()
    for key in ("subscriptions", "trials", "credits", "purchases", "activity"):
        all_uids.update(state.get(key, {}).keys())

    purged = []
    for uid in all_uids:
        # Skip users with an active paid subscription
        sub = state.get("subscriptions", {}).get(uid, {})
        if sub.get("status") in ("active", "trialing"):
            continue

        last_active = activity.get(uid, 0)
        # If never tracked, use trial start as a proxy
        if last_active == 0:
            trial = state.get("trials", {}).get(uid, {})
            last_active = trial.get("started_at", 0)

        # Still unknown — skip (don't purge users we have no timestamp for)
        if last_active == 0:
            continue

        if last_active < cutoff:
            wipe_user_data(uid)
            purged.append(uid)

    log.info("[cleanup] Purged %d inactive account(s) (cutoff: %d days)", len(purged), days)
    return {"ok": True, "purged_count": len(purged), "purged_user_ids": purged, "cutoff_days": days}


def list_all_users() -> list[dict]:
    """Return a summary of all known users in stripe_state.json."""
    state = _load_stripe_state()
    all_uids: set[str] = set()
    for key in ("subscriptions", "trials", "credits", "purchases", "activity"):
        all_uids.update(state.get(key, {}).keys())

    users = []
    for uid in sorted(all_uids):
        sub = state.get("subscriptions", {}).get(uid, {})
        trial = state.get("trials", {}).get(uid, {})
        credits_data = state.get("credits", {}).get(uid, {})
        last_active = state.get("activity", {}).get(uid, 0)
        users.append({
            "user_id": uid,
            "email": sub.get("email", ""),
            "tier": "pro" if sub.get("status") in ("active", "trialing") else "free",
            "trial_started": trial.get("started_at", 0),
            "credit_balance": credits_data.get("balance", 0),
            "last_active": last_active,
            "last_active_human": (
                datetime.fromtimestamp(last_active).isoformat() if last_active else "never"
            ),
        })
    return users
