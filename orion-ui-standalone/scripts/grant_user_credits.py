#!/usr/bin/env python3
"""
Grant credits to a user by their Supabase user_id.

Usage:
    python scripts/grant_user_credits.py <user_id> <credits_amount> [reason]

Example:
    python scripts/grant_user_credits.py "a2cddfbd-dabb-4d50-b515-b600a08ce8e8" 700 "admin_grant:onboarding_welcome"
"""

import json
import sys
import time
from pathlib import Path


def grant_credits(user_id: str, amount: int, reason: str = "admin_grant:manual"):
    """Add credits to a user's balance."""
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    stripe_state_file = repo_root / "config" / "stripe_state.json"

    if not stripe_state_file.exists():
        print(f"❌ stripe_state.json not found at {stripe_state_file}")
        sys.exit(1)

    # Load current state
    with open(stripe_state_file, "r", encoding="utf-8") as f:
        state = json.load(f)

    # Ensure credits section exists
    if "credits" not in state:
        state["credits"] = {}

    # Create or update user credit record
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

    # Save back
    with open(stripe_state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    new_balance = state["credits"][user_id]["balance"]
    print(f"✓ Granted {amount} credits to user {user_id}")
    print(f"  New balance: {new_balance} credits")
    print(f"  Reason: {reason}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python grant_user_credits.py <user_id> <credits_amount> [reason]")
        print("\nExample for $7 onboarding credit (700 credits):")
        print('  python grant_user_credits.py "a2cddfbd..." 700 "admin_grant:onboarding_$7_welcome"')
        sys.exit(1)

    user_id = sys.argv[1]
    try:
        amount = int(sys.argv[2])
    except ValueError:
        print(f"❌ Invalid credit amount: {sys.argv[2]}")
        sys.exit(1)

    reason = sys.argv[3] if len(sys.argv) > 3 else "admin_grant:manual"

    grant_credits(user_id, amount, reason)
