"""Stripe Subscription — Checkout, webhooks, and tier gating.

Implements a Free + Pro ($19/mo) model:
  - Free tier: chat, profiles, knowledge, vault, basic tools, default skins
  - Pro tier:  AGI loop, cost tracker, premium skins, priority routing,
               unlimited memory, image generation, internet tools

Flow:
  1. User signs up via Supabase → starts on Free tier
  2. User clicks "Upgrade to Pro" → Stripe Checkout session
  3. Stripe webhook confirms payment → stores subscription in stripe_state.json
  4. Middleware checks subscription status on gated routes
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("soulscript.stripe")

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_STRIPE_STATE_FILE = _CONFIG_DIR / "stripe_state.json"

# ── Stripe keys from environment (never hardcode) ────────────────
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_PUBLISHABLE_KEY = os.environ.get("STRIPE_PUBLISHABLE_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRO_PRICE_ID = os.environ.get("STRIPE_PRO_PRICE_ID", "")

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


# ── State persistence ────────────────────────────────────────────
def _load_stripe_state() -> dict:
    """Load subscription state from disk."""
    if _STRIPE_STATE_FILE.exists():
        try:
            with open(_STRIPE_STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"subscriptions": {}}


def _save_stripe_state(state: dict):
    """Persist subscription state."""
    _STRIPE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_STRIPE_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def get_user_tier(user_id: str) -> str:
    """Get the subscription tier for a user ('free' or 'pro')."""
    if not user_id:
        return "free"
    state = _load_stripe_state()
    sub = state.get("subscriptions", {}).get(user_id, {})
    if sub.get("status") in ("active", "trialing"):
        return "pro"
    return "free"


def get_user_subscription(user_id: str) -> dict:
    """Get full subscription info for a user."""
    state = _load_stripe_state()
    sub = state.get("subscriptions", {}).get(user_id, {})
    tier = "pro" if sub.get("status") in ("active", "trialing") else "free"
    return {
        "tier": tier,
        "tier_info": TIER_INFO[tier],
        "subscription": sub if sub else None,
        "stripe_configured": bool(STRIPE_SECRET_KEY),
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


def user_has_feature(user_id: str, feature: str) -> bool:
    """Check if a user's tier includes a specific feature."""
    tier = get_user_tier(user_id)
    return feature in TIER_INFO[tier]["features"]


def check_tier_limit(user_id: str, limit_key: str, current_count: int) -> bool:
    """Check if a user is within their tier's limit. Returns True if allowed."""
    tier = get_user_tier(user_id)
    limit = TIER_INFO[tier]["limits"].get(limit_key, 0)
    if limit == -1:
        return True  # unlimited
    return current_count < limit


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
        else:
            event = json.loads(payload)
            log.warning("[stripe] Webhook signature verification skipped (no secret)")
    except Exception as exc:
        log.error("[stripe] Webhook verification failed: %s", exc)
        return {"error": f"Webhook verification failed: {exc}"}

    event_type = event.get("type", "")
    data = event.get("data", {}).get("object", {})

    if event_type == "checkout.session.completed":
        user_id = data.get("client_reference_id") or data.get("metadata", {}).get("user_id")
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
