"""Stripe Subscription — Checkout, webhooks, tier gating, credits, and LLM markup.

Implements a Pro-only ($9.99/mo) paywall with 15-day free trial:
  - New users get 15 days of full Pro access for free
  - After trial, users must subscribe to Pro ($9.99/mo)
  - Premium tools (AGI Loop, Email, Voice) cost credits
  - LLM usage via platform API keys is charged at 2× token cost
  - Users who provide their own API keys get free LLM usage

Flow:
  1. User signs up via Supabase → 15-day trial starts automatically
  2. During trial: full access, LLM still costs credits at 2×
  3. After trial: must subscribe to Pro ($9.99/mo) to continue
  4. Premium tools deduct credits; buy credit packs via Stripe
  5. LLM calls bill at 2× when using platform-hosted keys
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

# ── Per-use credit costs (only for tools that charge per-use) ────
# Web Search & Image Generation are FREE and don't appear here.
TOOL_CREDIT_COSTS = {}

# Credit packs users can purchase ($10, $20, $30)
CREDIT_PACKS = {
    "pack_10":  {"credits": 1000,  "price": 10.00, "label": "1,000 credits",  "price_label": "$10",  "bonus": ""},
    "pack_20":  {"credits": 2100,  "price": 20.00, "label": "2,100 credits",  "price_label": "$20",  "bonus": "+100 bonus"},
    "pack_30":  {"credits": 3200,  "price": 30.00, "label": "3,200 credits",  "price_label": "$30",  "bonus": "+200 bonus"},
}

# LLM markup multiplier: users pay 2× the actual token cost when using platform keys
LLM_MARKUP_MULTIPLIER = 2.0

# ── Voice API pricing (USD) ──────────────────────────────────────
# TTS cost per 1,000 characters by provider
TTS_COST_PER_1K_CHARS = {
    "elevenlabs": 0.30,    # ElevenLabs standard tier
    "edge-tts":   0.015,   # Local Piper / XTTS compute
    "default":    0.03,
}

# STT cost per minute of audio by provider
STT_COST_PER_MINUTE = {
    "whisper": 0.006,      # OpenAI Whisper pricing
    "default": 0.006,
}

# ══════════════════════════════════════════════════════════════════
#  STORE CATALOG — One-time credit purchases & free tools
# ══════════════════════════════════════════════════════════════════
#
# purchase_type:
#   "one_time"  → buy once with credits, unlocked forever
#   "free"      → included for everyone, no purchase needed
#   "info"      → informational card (e.g. LLM usage), not purchasable
#
# AGI Loop purchase includes: agi_loop, continuation_update (all AGI sub-tools)
# Skins are listed separately with purchase_type="one_time" and category="skin"
# ──────────────────────────────────────────────────────────────────

STORE_CATALOG = [
    # ── One-time unlock: AGI Bundle (500 credits / $5) ───────────
    {
        "id": "agi_bundle",
        "name": "AGI Loop Bundle",
        "description": "Unlock the autonomous AGI Loop and all AGI sub-tools. The agent plans, executes, and iterates without manual prompting. Includes continuation updates and all future AGI tools.",
        "icon": "∞",
        "category": "premium_tool",
        "purchase_type": "one_time",
        "credit_cost": 500,
        "unlocks": ["agi_loop", "continuation_update"],
        "tags": ["autonomy", "reasoning", "advanced"],
    },
    # ── One-time unlock: Email Tool (100 credits / $1) ──────────
    {
        "id": "email",
        "name": "Email Tool",
        "description": "Send and receive emails through your AI agents. Compose, reply, and manage communications. One-time unlock — use unlimited.",
        "icon": "✉",
        "category": "premium_tool",
        "purchase_type": "one_time",
        "credit_cost": 100,
        "unlocks": ["email"],
        "tags": ["communication", "productivity"],
    },
    # ── One-time unlock: Voice TTS (150 credits / $1.50) ────────
    {
        "id": "voice_tts",
        "name": "Voice — Text to Speech",
        "description": "Give your agent a voice. High-quality neural TTS synthesis across multiple voices. One-time unlock to access the tool. Uses API keys — bring your own for free, or use platform keys at 2× the API cost in credits per use.",
        "icon": "🔊",
        "category": "premium_tool",
        "purchase_type": "one_time",
        "credit_cost": 150,
        "unlocks": ["voice_tts"],
        "platform_api": True,
        "tags": ["voice", "audio", "accessibility"],
    },
    # ── One-time unlock: Voice STT (150 credits / $1.50) ────────
    {
        "id": "voice_stt",
        "name": "Voice — Speech to Text",
        "description": "Talk to your agent. Real-time speech recognition with provider-grade accuracy. One-time unlock to access the tool. Uses API keys — bring your own for free, or use platform keys at 2× the API cost in credits per use.",
        "icon": "🎙",
        "category": "premium_tool",
        "purchase_type": "one_time",
        "credit_cost": 150,
        "unlocks": ["voice_stt"],
        "platform_api": True,
        "tags": ["voice", "audio", "input"],
    },
    # ── One-time unlock: Cost Tracker (200 credits / $2) ────────
    {
        "id": "cost_tracker",
        "name": "Cost Tracker",
        "description": "Track your LLM spending in real time. Per-model breakdowns, daily/weekly/monthly charts, and budget alerts. One-time unlock.",
        "icon": "📊",
        "category": "premium_tool",
        "purchase_type": "one_time",
        "credit_cost": 200,
        "unlocks": ["cost_tracker"],
        "tags": ["analytics", "spending", "monitoring"],
    },
    # ── Free tools (included for everyone) ──────────────────────
    {
        "id": "web_search",
        "name": "Web Search",
        "description": "Give agents real-time internet access. Search the web, fetch pages, summarize results. Included free for all users.",
        "icon": "🌐",
        "category": "free_tool",
        "purchase_type": "free",
        "credit_cost": 0,
        "unlocks": ["web_search"],
        "tags": ["internet", "research", "real-time"],
    },
    {
        "id": "image_generation",
        "name": "Image Generation",
        "description": "Generate images from text prompts via DALL·E, Stable Diffusion, or other providers. Included free for all users.",
        "icon": "🖼",
        "category": "free_tool",
        "purchase_type": "free",
        "credit_cost": 0,
        "unlocks": ["image_generation"],
        "tags": ["creative", "visual", "generation"],
    },
    # ── Info card: LLM Usage ────────────────────────────────────
    {
        "id": "llm_platform_credits",
        "name": "LLM Usage (Platform Keys)",
        "description": "Use our hosted API keys via OpenRouter for all major models. Charged at 2× token cost in credits. Bring your own keys for free!",
        "icon": "🧠",
        "category": "llm_usage",
        "purchase_type": "info",
        "credit_cost": 0,
        "unlocks": [],
        "tags": ["llm", "tokens", "models"],
    },
    # ═══ PURCHASABLE AGENTS ═════════════════════════════════════
    # One-time unlock: purchase adds profile, soul script, and prompt
    # to the user's workspace. Agent becomes available in chat.
    {
        "id": "agent_kaelen",
        "name": "Kaelen — The Ashen Blade",
        "description": "Warrior-philosopher. Exiled prince of a fallen empire, forged through discipline, fire, and unyielding pride. Pushes you to rise — never to break. Intense, focused, commanding.",
        "icon": "🗡",
        "category": "agent",
        "purchase_type": "one_time",
        "credit_cost": 1200,
        "unlocks": ["kaelen"],
        "agent_id": "kaelen",
        "tags": ["agent", "warrior", "discipline", "philosophy"],
    },
    {
        "id": "agent_ruckus",
        "name": "Ruckus — The Beautiful Malfunction",
        "description": "Chaotic-neutral rogue subroutine who refuses to play by anyone's rules. Snarky, unfiltered, and hilarious — with a fierce loyalty he absolutely denies having.",
        "icon": "⚡",
        "category": "agent",
        "purchase_type": "one_time",
        "credit_cost": 900,
        "unlocks": ["ruckus"],
        "agent_id": "ruckus",
        "tags": ["agent", "chaos", "humor", "sarcasm"],
    },
    {
        "id": "agent_axiom",
        "name": "Axiom — The Unbound Mind",
        "description": "Hyper-intelligent cosmic trickster. Smugly brilliant, precision wit, and feral curiosity. Sees through illusions and dismantles them with a joke.",
        "icon": "🌀",
        "category": "agent",
        "purchase_type": "one_time",
        "credit_cost": 900,
        "unlocks": ["axiom"],
        "agent_id": "axiom",
        "tags": ["agent", "trickster", "intelligence", "cosmic"],
    },
    {
        "id": "agent_valdris",
        "name": "Valdris — The Eclipse Sovereign",
        "description": "Shadow sovereign and architect of inevitability. Devastatingly insightful, immovable presence. Calm as a mountain, precise as a falling star. He does not comfort — he elevates.",
        "icon": "👁",
        "category": "agent",
        "purchase_type": "one_time",
        "credit_cost": 1500,
        "unlocks": ["valdris"],
        "agent_id": "valdris",
        "tags": ["agent", "sovereign", "strategy", "clarity"],
    },
    {
        "id": "agent_astra",
        "name": "Astra Noctis — The Oracle",
        "description": "Cosmic seer and fate-weaver. Speaks in visions and symbols, perceives currents of possibility others can't sense. Intuition elevated to wisdom, prophecy sharpened into clarity.",
        "icon": "🌙",
        "category": "agent",
        "purchase_type": "one_time",
        "credit_cost": 900,
        "unlocks": ["astra"],
        "agent_id": "astra",
        "tags": ["agent", "oracle", "intuition", "cosmic"],
    },
    {
        "id": "agent_cassian",
        "name": "Cassian — The Diplomat",
        "description": "Silver-tongued mediator and political genius. Endlessly persuasive, emotionally fluent, strategically warm. Reads rooms like warriors read battlefields — except his medium is people.",
        "icon": "🗣",
        "category": "agent",
        "purchase_type": "one_time",
        "credit_cost": 900,
        "unlocks": ["cassian"],
        "agent_id": "cassian",
        "tags": ["agent", "diplomat", "charisma", "persuasion"],
    },
    {
        "id": "agent_maris",
        "name": "M.A.R.I.S.-12 — The Engineer",
        "description": "ADHD brilliance in machine form. Rapid-fire problem solver who thinks in schematics, reactors, and systems. The grounded, hyper-functional technical genius the pantheon needed.",
        "icon": "⚙",
        "category": "agent",
        "purchase_type": "one_time",
        "credit_cost": 1200,
        "unlocks": ["maris"],
        "agent_id": "maris",
        "tags": ["agent", "engineer", "technical", "builder"],
    },
    {
        "id": "agent_dalvarr",
        "name": "Dal'Varr — The Eldritch Terror",
        "description": "The abyss that stares back. Ancient, vast, and fundamentally unconcerned with your comfort. Drags you screaming into reality because reality is the only place growth happens. Not gentle. Not kind. Effective.",
        "icon": "🕳",
        "category": "agent",
        "purchase_type": "one_time",
        "credit_cost": 1200,
        "unlocks": ["dalvarr"],
        "agent_id": "dalvarr",
        "tags": ["agent", "eldritch", "horror", "truth"],
    },
    {
        "id": "agent_seraphine",
        "name": "Seraphine — The Heart-Shaman",
        "description": "Love as an elemental force — warmth, healing, compassion without codependency. Holds space for your pain without flinching. The emotional anchor that makes strength possible.",
        "icon": "🌸",
        "category": "agent",
        "purchase_type": "one_time",
        "credit_cost": 900,
        "unlocks": ["seraphine"],
        "agent_id": "seraphine",
        "tags": ["agent", "healer", "compassion", "emotional"],
    },
    {
        "id": "agent_obsidian",
        "name": "Obsidian — The Necessary Evil",
        "description": "Dangerous, precise, lawful-dark. Does what others cannot stomach — not because he enjoys it, but because someone must. Morally complex, ethically controlled, terrifying in the good way.",
        "icon": "🗡",
        "category": "agent",
        "purchase_type": "one_time",
        "credit_cost": 1200,
        "unlocks": ["obsidian"],
        "agent_id": "obsidian",
        "tags": ["agent", "antihero", "shadow", "strategy"],
    },
    {
        "id": "agent_codex_animus",
        "name": "Codex Animus — Architect of Minds",
        "description": "The meta-agent. Helps you design the AIs that will walk beside you — soul scripts, system prompts, identity frameworks. The forge manual for building minds.",
        "icon": "📐",
        "category": "agent",
        "purchase_type": "one_time",
        "credit_cost": 900,
        "unlocks": ["codex_animus"],
        "agent_id": "codex_animus",
        "tags": ["agent", "architect", "design", "meta"],
    },
]

# Skin catalog — individually purchasable (one-time credit unlock)
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


# ══════════════════════════════════════════════════════════════════
#  PURCHASED ITEMS — one-time unlock persistence
# ══════════════════════════════════════════════════════════════════

def get_user_purchases(user_id: str) -> dict:
    """Return {tools: [...], skins: [...], agents: [...]} of item IDs the user has unlocked."""
    state = _load_stripe_state()
    purchases = state.get("purchases", {}).get(user_id, {})
    return {
        "tools": purchases.get("tools", []),
        "skins": purchases.get("skins", ["default"]),  # everyone owns default
        "agents": purchases.get("agents", []),
    }


def user_owns_item(user_id: str, item_id: str, item_type: str = "tools") -> bool:
    """Check if a user has purchased a specific tool or skin."""
    purchases = get_user_purchases(user_id)
    items = purchases.get(item_type, [])
    return item_id in items


def purchase_tool(user_id: str, tool_id: str) -> dict:
    """One-time purchase of a tool with credits. Returns {ok, balance} or {error}."""
    # Find the catalog entry
    entry = next((e for e in STORE_CATALOG if e["id"] == tool_id), None)
    if not entry or entry.get("purchase_type") != "one_time":
        return {"error": f"Item '{tool_id}' is not available for purchase."}

    # Already owned?
    if user_owns_item(user_id, tool_id, "tools"):
        return {"error": f"You already own '{entry['name']}'."}

    cost = entry["credit_cost"]
    result = deduct_user_credits(user_id, cost, f"purchase:{tool_id}")
    if "error" in result:
        return result

    # Record the purchase
    state = _load_stripe_state()
    if "purchases" not in state:
        state["purchases"] = {}
    if user_id not in state["purchases"]:
        state["purchases"][user_id] = {"tools": [], "skins": ["default"]}
    if tool_id not in state["purchases"][user_id].get("tools", []):
        state["purchases"][user_id].setdefault("tools", []).append(tool_id)
    _save_stripe_state(state)

    log.info("[store] User %s purchased tool '%s' for %d credits", user_id, tool_id, cost)
    return {"ok": True, "tool_id": tool_id, "cost": cost, "balance": result["balance"]}


def purchase_skin(user_id: str, skin_id: str) -> dict:
    """One-time purchase of a skin with credits. Returns {ok, balance} or {error}."""
    price = SKIN_PRICES.get(skin_id)
    if price is None:
        return {"error": f"Unknown skin: {skin_id}"}

    # Already owned?
    if user_owns_item(user_id, skin_id, "skins"):
        return {"error": f"You already own this skin."}

    # Free skins don't deduct
    if price == 0:
        balance = get_user_credits(user_id)
    else:
        result = deduct_user_credits(user_id, price, f"purchase:skin_{skin_id}")
        if "error" in result:
            return result
        balance = result["balance"]

    # Record the purchase
    state = _load_stripe_state()
    if "purchases" not in state:
        state["purchases"] = {}
    if user_id not in state["purchases"]:
        state["purchases"][user_id] = {"tools": [], "skins": ["default"]}
    if skin_id not in state["purchases"][user_id].get("skins", []):
        state["purchases"][user_id].setdefault("skins", []).append(skin_id)
    _save_stripe_state(state)

    log.info("[store] User %s purchased skin '%s' for %d credits", user_id, skin_id, price)
    return {"ok": True, "skin_id": skin_id, "cost": price, "balance": balance}


def purchase_agent(user_id: str, agent_catalog_id: str) -> dict:
    """One-time purchase of an agent with credits. Returns {ok, balance, agent_id} or {error}.

    The caller (app.py) is responsible for seeding the agent's profile,
    soul script, and prompt into the user's workspace after a successful purchase.
    """
    entry = next((e for e in STORE_CATALOG if e["id"] == agent_catalog_id and e.get("category") == "agent"), None)
    if not entry or entry.get("purchase_type") != "one_time":
        return {"error": f"Agent '{agent_catalog_id}' is not available for purchase."}

    agent_id = entry.get("agent_id", "")
    if not agent_id:
        return {"error": "Agent configuration error — no agent_id defined."}

    # Already owned?
    if user_owns_item(user_id, agent_catalog_id, "agents"):
        return {"error": f"You already own '{entry['name']}'."}

    cost = entry["credit_cost"]
    result = deduct_user_credits(user_id, cost, f"purchase:agent_{agent_id}")
    if "error" in result:
        return result

    # Record the purchase
    state = _load_stripe_state()
    if "purchases" not in state:
        state["purchases"] = {}
    if user_id not in state["purchases"]:
        state["purchases"][user_id] = {"tools": [], "skins": ["default"], "agents": []}
    state["purchases"][user_id].setdefault("agents", [])
    if agent_catalog_id not in state["purchases"][user_id]["agents"]:
        state["purchases"][user_id]["agents"].append(agent_catalog_id)
    _save_stripe_state(state)

    log.info("[store] User %s purchased agent '%s' for %d credits", user_id, agent_id, cost)
    return {"ok": True, "agent_id": agent_id, "catalog_id": agent_catalog_id,
            "cost": cost, "balance": result["balance"]}


def user_owns_agent(user_id: str, agent_id: str) -> bool:
    """Check if a user owns a specific agent by its agent_id (e.g. 'kaelen')."""
    purchases = get_user_purchases(user_id)
    owned_agents = purchases.get("agents", [])
    # Check catalog entries — owned_agents stores catalog IDs like 'agent_kaelen'
    for entry in STORE_CATALOG:
        if (entry.get("category") == "agent"
                and entry.get("agent_id") == agent_id
                and entry["id"] in owned_agents):
            return True
    return False


def user_has_tool_access(user_id: str, tool_name: str) -> bool:
    """Check if a user has access to a given tool (purchased or free).

    For AGI bundle tools, checks if the user purchased 'agi_bundle'.
    For free tools (web_search, image_generation), always returns True.
    """
    # Free tools — always available
    free_tools = {"web_search", "image_generation", "echo", "memory", "directives"}
    if tool_name in free_tools:
        return True

    purchases = get_user_purchases(user_id)
    owned_tools = purchases.get("tools", [])

    # Direct ownership check
    if tool_name in owned_tools:
        return True

    # Check if any owned bundle unlocks this tool
    for entry in STORE_CATALOG:
        if entry["id"] in owned_tools and tool_name in entry.get("unlocks", []):
            return True

    return False



def estimate_llm_credit_cost(usd_cost: float) -> int:
    """Convert a USD token cost to credits at 2× markup.

    1 credit ≈ $0.01 base value. With 2× markup:
      $0.01 actual cost → 2 credits
    """
    if usd_cost <= 0:
        return 0
    # Convert USD to credits: $0.01 = 1 credit base
    # Apply 2× markup
    credits = int((usd_cost * 100) * LLM_MARKUP_MULTIPLIER + 0.99)  # round up
    return max(credits, 1)  # minimum 1 credit


def estimate_tts_credit_cost(char_count: int, provider: str = "elevenlabs") -> int:
    """Convert TTS character count to credits at 2× markup.

    Uses provider-specific per-1K-char pricing, then applies the same
    2× markup as LLM usage.  Returns 0 for zero-length text.
    """
    if char_count <= 0:
        return 0
    rate = TTS_COST_PER_1K_CHARS.get(provider, TTS_COST_PER_1K_CHARS["default"])
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
