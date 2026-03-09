"""Stripe Subscription — Checkout, webhooks, tier gating, credits, and LLM markup.

Implements a Pro-only ($9.99/mo) paywall with 15-day free trial:
  - New users get 15 days of full Pro access for free
  - After trial, users must subscribe to Pro ($9.99/mo)
  - Premium tools (AGI Loop, Email, Voice) cost credits
  - LLM usage via platform API keys is charged at 1.5× token cost
  - Users who provide their own API keys get free LLM usage

Flow:
  1. User signs up via Supabase → 15-day trial starts automatically
  2. During trial: full access, LLM still costs credits at 1.5×
  3. After trial: must subscribe to Pro ($9.99/mo) to continue
  4. Premium tools deduct credits; buy credit packs via Stripe
  5. LLM calls bill at 1.5× when using platform-hosted keys
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

# ── Free Trial ───────────────────────────────────────────────────
FREE_TRIAL_DAYS = 15
_SECONDS_PER_DAY = 86400


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
    """Get the subscription tier for a user ('free' or 'pro').

    Pro is granted if the user has an active subscription OR is within
    the free trial period.
    """
    if not user_id:
        return "free"
    state = _load_stripe_state()
    sub = state.get("subscriptions", {}).get(user_id, {})
    if sub.get("status") in ("active", "trialing"):
        return "pro"
    # Check free trial
    trial = get_trial_status(user_id)
    if trial["active"]:
        return "pro"
    return "free"


def get_user_subscription(user_id: str) -> dict:
    """Get full subscription info for a user (includes trial status)."""
    state = _load_stripe_state()
    sub = state.get("subscriptions", {}).get(user_id, {})
    has_sub = sub.get("status") in ("active", "trialing")
    trial = get_trial_status(user_id)
    tier = "pro" if (has_sub or trial["active"]) else "free"
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
        metadata = data.get("metadata", {})
        user_id = data.get("client_reference_id") or metadata.get("user_id")

        # Credit pack purchase
        if metadata.get("type") == "credits" and user_id:
            credits = int(metadata.get("credits", 0))
            pack_id = metadata.get("pack_id", "unknown")
            if credits > 0:
                add_user_credits(user_id, credits, reason=f"purchase:{pack_id}")
            return {"ok": True, "action": "credits_purchased", "user_id": user_id, "credits": credits}

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


# ═══════════════════════════════════════════════════════════════════
#  CREDIT SYSTEM — per-tool metered billing
# ═══════════════════════════════════════════════════════════════════

# Tool credit costs — how many credits each premium tool use costs
TOOL_CREDIT_COSTS = {
    "agi_loop":         10,   # per autonomous run
    "email":             2,   # per email sent
    "voice_tts":         3,   # per minute (rounded up)
    "voice_stt":         3,   # per minute (rounded up)
    "image_generation":  5,   # per image generated
    "web_search":        1,   # per search query
}

# Credit packs users can purchase ($10, $20, $30)
CREDIT_PACKS = {
    "pack_10":  {"credits": 1000,  "price": 10.00, "label": "1,000 credits",  "price_label": "$10",  "bonus": ""},
    "pack_20":  {"credits": 2200,  "price": 20.00, "label": "2,200 credits",  "price_label": "$20",  "bonus": "+200 bonus"},
    "pack_30":  {"credits": 3500,  "price": 30.00, "label": "3,500 credits",  "price_label": "$30",  "bonus": "+500 bonus"},
}

# LLM markup multiplier: users pay 1.5× the actual token cost when using platform keys
LLM_MARKUP_MULTIPLIER = 1.5

# Store catalog — tools available for individual credit purchase
STORE_CATALOG = [
    {
        "id": "agi_loop",
        "name": "AGI Loop",
        "description": "Autonomous multi-step reasoning engine. The agent plans, executes, and iterates without manual prompting.",
        "icon": "∞",
        "category": "premium_tool",
        "credit_cost": 10,
        "per_label": "per run",
        "tags": ["autonomy", "reasoning", "advanced"],
    },
    {
        "id": "email",
        "name": "Email Tool",
        "description": "Send and receive emails through your AI agents. Compose, reply, and manage communications.",
        "icon": "✉",
        "category": "premium_tool",
        "credit_cost": 2,
        "per_label": "per email",
        "tags": ["communication", "productivity"],
    },
    {
        "id": "voice_tts",
        "name": "Voice — Text to Speech",
        "description": "Give your agent a voice. High-quality neural TTS synthesis across multiple voices.",
        "icon": "🔊",
        "category": "premium_tool",
        "credit_cost": 3,
        "per_label": "per minute",
        "tags": ["voice", "audio", "accessibility"],
    },
    {
        "id": "voice_stt",
        "name": "Voice — Speech to Text",
        "description": "Talk to your agent. Real-time speech recognition with provider-grade accuracy.",
        "icon": "🎙",
        "category": "premium_tool",
        "credit_cost": 3,
        "per_label": "per minute",
        "tags": ["voice", "audio", "input"],
    },
    {
        "id": "image_generation",
        "name": "Image Generation",
        "description": "Generate images from text prompts via DALL·E, Stable Diffusion, or other providers.",
        "icon": "🖼",
        "category": "premium_tool",
        "credit_cost": 5,
        "per_label": "per image",
        "tags": ["creative", "visual", "generation"],
    },
    {
        "id": "web_search",
        "name": "Web Search",
        "description": "Give agents real-time internet access. Search the web, fetch pages, summarize results.",
        "icon": "🌐",
        "category": "premium_tool",
        "credit_cost": 1,
        "per_label": "per search",
        "tags": ["internet", "research", "real-time"],
    },
    {
        "id": "llm_platform_credits",
        "name": "LLM Usage (Platform Keys)",
        "description": "Use our hosted API keys for OpenAI, Anthropic, DeepSeek, xAI, Google, and more. Charged at 1.5× token cost in credits. Bring your own keys for free!",
        "icon": "🧠",
        "category": "llm_usage",
        "credit_cost": 0,
        "per_label": "1.5× token cost",
        "tags": ["llm", "tokens", "models"],
    },
]


def estimate_llm_credit_cost(usd_cost: float) -> int:
    """Convert a USD token cost to credits at 1.5× markup.

    1 credit ≈ $0.01 base value. With 1.5× markup:
      $0.01 actual cost → 1.5 credits (rounded up to 2)
    """
    if usd_cost <= 0:
        return 0
    # Convert USD to credits: $0.01 = 1 credit base
    # Apply 1.5× markup
    credits = int((usd_cost * 100) * LLM_MARKUP_MULTIPLIER + 0.99)  # round up
    return max(credits, 1)  # minimum 1 credit


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

    try:
        session = stripe.checkout.Session.create(
            mode="payment",
            payment_method_types=["card"],
            line_items=[{
                "price_data": {
                    "currency": "usd",
                    "product_data": {
                        "name": f"SoulScript Credits — {pack['label']}",
                        "description": f"{pack['credits']} credits for premium tool usage",
                    },
                    "unit_amount": int(pack["price"] * 100),  # cents
                },
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
