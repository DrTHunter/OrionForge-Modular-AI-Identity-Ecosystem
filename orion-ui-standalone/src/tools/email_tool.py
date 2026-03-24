"""Email tool — lets agents send emails via SMTP or a FastAPI email service.

Supports multiple email accounts stored in config/settings.json under
"tool_config.email".  Each account can have its own SMTP credentials,
signature, and be designated as:

  - **default**: the fallback account for outgoing mail
  - **user_email**: the account that belongs to the human operator
  - **agent_default**: automatically used when a specific agent sends mail

Sending modes:
  1. Direct SMTP (preferred) — uses account credentials stored locally.
  2. API relay — forwards through an optional FastAPI email-service backend.

Actions:
  - send:     Send an email (requires subject, body, recipients)
  - status:   List configured accounts and check connectivity
  - accounts: List all configured email accounts
"""

import base64
import json
import logging
import os
import re
import smtplib
import uuid
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_API_BASE_URL = os.environ.get(
    "EMAIL_API_URL", "http://127.0.0.1:8000"
)

_DEFAULT_TIMEOUT = 30  # seconds

# Path to settings file (resolved relative to project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SETTINGS_FILE = _PROJECT_ROOT / "config" / "settings.json"
_DATA_DIR = _PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# Password encryption  (Fernet symmetric encryption)
# ---------------------------------------------------------------------------
# The key is sourced from the EMAIL_ENCRYPTION_KEY env var (Fly.io secret).
# If unset, we derive a key from a stable machine-local seed so encryption
# still works in dev — but in production the Fly.io secret should be set.

_fernet_instance = None


def _get_fernet():
    """Return a cached Fernet cipher for password encryption."""
    global _fernet_instance
    if _fernet_instance is not None:
        return _fernet_instance

    try:
        from cryptography.fernet import Fernet
    except ImportError:
        log.warning("[email] cryptography package not installed — passwords stored without encryption")
        return None

    key = os.environ.get("EMAIL_ENCRYPTION_KEY", "")
    if not key:
        # Derive a stable key from a machine-local seed (dev fallback)
        import hashlib
        seed = str(_DATA_DIR.resolve()).encode()
        raw = hashlib.sha256(seed).digest()
        key = base64.urlsafe_b64encode(raw)
    else:
        key = key.encode() if isinstance(key, str) else key

    try:
        _fernet_instance = Fernet(key)
    except Exception as exc:
        log.warning("[email] Invalid Fernet key — passwords stored without encryption: %s", exc)
        return None
    return _fernet_instance


def _encrypt_password(plaintext: str) -> str:
    """Encrypt a password for storage. Returns 'enc:...' or plaintext if crypto unavailable."""
    if not plaintext or plaintext == "••••••••":
        return plaintext
    f = _get_fernet()
    if f is None:
        return plaintext
    try:
        token = f.encrypt(plaintext.encode("utf-8"))
        return "enc:" + token.decode("ascii")
    except Exception:
        return plaintext


def _decrypt_password(stored: str) -> str:
    """Decrypt a stored password. Returns plaintext."""
    if not stored or not stored.startswith("enc:"):
        return stored  # Not encrypted — return as-is
    f = _get_fernet()
    if f is None:
        return stored  # Can't decrypt — return raw
    try:
        token = stored[4:].encode("ascii")
        return f.decrypt(token).decode("utf-8")
    except Exception:
        log.warning("[email] Failed to decrypt password — key may have changed")
        return ""


# ---------------------------------------------------------------------------
# User-id validation (prevent path traversal)
# ---------------------------------------------------------------------------
_SAFE_USER_ID = re.compile(r"^[a-zA-Z0-9_\-]+$")


def _validate_user_id(user_id: str) -> str:
    """Validate user_id to prevent path traversal attacks."""
    if not user_id or not _SAFE_USER_ID.match(user_id):
        raise ValueError(f"Invalid user_id")
    return user_id


# ---------------------------------------------------------------------------
# Settings helpers (global config only — NOT per-user accounts)
# ---------------------------------------------------------------------------

def _load_settings() -> Dict[str, Any]:
    """Load full settings.json."""
    if not _SETTINGS_FILE.exists():
        return {}
    try:
        with open(_SETTINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_settings(settings: Dict[str, Any]) -> None:
    """Persist settings.json."""
    _SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)


def _load_tool_config() -> Dict[str, Any]:
    """Load email config from settings.json (global settings only)."""
    return _load_settings().get("tool_config", {}).get("email", {})


def get_effective_config(user_id: str = "") -> Dict[str, Any]:
    """Return the merged config (defaults + user overrides). Used by the UI."""
    saved = _load_tool_config()
    accounts = _get_accounts_raw(user_id) if user_id else []
    safe_accounts = _mask_accounts(accounts)
    return {
        "api_base_url": saved.get("api_base_url", _DEFAULT_API_BASE_URL),
        "timeout": saved.get("timeout", _DEFAULT_TIMEOUT),
        "require_confirmation": saved.get("require_confirmation", True),
        "accounts": safe_accounts,
    }


# ---------------------------------------------------------------------------
# Per-user account storage — data/users/{user_id}/email_accounts.json
# ---------------------------------------------------------------------------

def _user_accounts_path(user_id: str) -> Path:
    """Return the path to a user's email accounts file."""
    _validate_user_id(user_id)
    return _DATA_DIR / "users" / user_id / "email_accounts.json"


def _load_user_accounts(user_id: str) -> List[Dict[str, Any]]:
    """Load email accounts for a specific user (passwords decrypted)."""
    path = _user_accounts_path(user_id)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        accounts = data if isinstance(data, list) else []
        # Decrypt passwords on load
        for acct in accounts:
            if acct.get("password"):
                acct["password"] = _decrypt_password(acct["password"])
        return accounts
    except Exception:
        return []


def _save_user_accounts(user_id: str, accounts: List[Dict[str, Any]]) -> None:
    """Persist email accounts for a specific user (passwords encrypted)."""
    path = _user_accounts_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Encrypt passwords before writing
    to_save = []
    for acct in accounts:
        a = dict(acct)
        if a.get("password") and a["password"] != "••••••••":
            a["password"] = _encrypt_password(a["password"])
        to_save.append(a)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_save, f, indent=2, ensure_ascii=False)


def _mask_accounts(accounts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return accounts with passwords masked for display."""
    safe = []
    for acct in accounts:
        a = dict(acct)
        if a.get("password"):
            a["password_set"] = True
            a["password"] = "••••••••"
        else:
            a["password_set"] = False
        safe.append(a)
    return safe


# ---------------------------------------------------------------------------
# Account management helpers  (used by API routes and the tool itself)
# All functions require user_id for per-user isolation.
# ---------------------------------------------------------------------------

def get_accounts(user_id: str) -> List[Dict[str, Any]]:
    """Return email accounts for a user (passwords masked)."""
    return _mask_accounts(_load_user_accounts(user_id))


def _get_accounts_raw(user_id: str) -> List[Dict[str, Any]]:
    """Return email accounts WITH real passwords (internal use only)."""
    return _load_user_accounts(user_id)


def get_account_by_id(user_id: str, account_id: str) -> Optional[Dict[str, Any]]:
    """Lookup a single account by id, scoped to user (with real password)."""
    for acct in _get_accounts_raw(user_id):
        if acct.get("id") == account_id:
            return acct
    return None


def get_default_account(user_id: str) -> Optional[Dict[str, Any]]:
    """Return the user's account marked as default, or their first account."""
    accounts = _get_accounts_raw(user_id)
    for acct in accounts:
        if acct.get("is_default"):
            return acct
    return accounts[0] if accounts else None


def get_user_account(user_id: str) -> Optional[Dict[str, Any]]:
    """Return the account marked as user_email for this user."""
    for acct in _get_accounts_raw(user_id):
        if acct.get("is_user_email"):
            return acct
    return None


def get_agent_default_account(user_id: str, agent_name: str) -> Optional[Dict[str, Any]]:
    """Return the account assigned as default for a specific agent (within this user)."""
    for acct in _get_accounts_raw(user_id):
        if acct.get("agent_default") == agent_name:
            return acct
    return None


def save_account(user_id: str, account_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create or update an email account for a user.  Returns the saved account."""
    accounts = _load_user_accounts(user_id)

    acct_id = account_data.get("id")

    # If password field is the masked placeholder, keep the old password
    if account_data.get("password") in ("••••••••", ""):
        for existing in accounts:
            if existing.get("id") == acct_id:
                account_data["password"] = existing.get("password", "")
                break

    if acct_id:
        # Update existing — verify ownership
        for i, existing in enumerate(accounts):
            if existing.get("id") == acct_id:
                accounts[i] = account_data
                break
        else:
            # id provided but not found in this user's accounts — treat as new
            accounts.append(account_data)
    else:
        # New account
        account_data["id"] = f"acct_{uuid.uuid4().hex[:8]}"
        accounts.append(account_data)

    # Enforce uniqueness of is_default within this user's accounts
    if account_data.get("is_default"):
        for acct in accounts:
            if acct["id"] != account_data["id"]:
                acct["is_default"] = False

    # Enforce uniqueness of is_user_email within this user's accounts
    if account_data.get("is_user_email"):
        for acct in accounts:
            if acct["id"] != account_data["id"]:
                acct["is_user_email"] = False

    # Enforce uniqueness of agent_default (one account per agent per user)
    agent = account_data.get("agent_default", "")
    if agent:
        for acct in accounts:
            if acct["id"] != account_data["id"] and acct.get("agent_default") == agent:
                acct["agent_default"] = ""

    _save_user_accounts(user_id, accounts)
    return account_data


def delete_account(user_id: str, account_id: str) -> bool:
    """Delete an email account by id, scoped to user.  Returns True if found and deleted."""
    accounts = _load_user_accounts(user_id)
    new_accounts = [a for a in accounts if a.get("id") != account_id]
    if len(new_accounts) == len(accounts):
        return False
    _save_user_accounts(user_id, new_accounts)
    return True


# ---------------------------------------------------------------------------
# Direct SMTP sending
# ---------------------------------------------------------------------------

def _send_via_smtp(
    account: Dict[str, Any],
    subject: str,
    body: str,
    recipients: List[str],
    append_signature: bool = True,
) -> str:
    """Send email directly via SMTP using account credentials."""
    email_addr = account.get("email", "")
    password = account.get("password", "")
    smtp_server = account.get("smtp_server", "smtp.gmail.com")
    smtp_port = int(account.get("smtp_port", 465))
    signature = account.get("signature", "")
    display_name = account.get("label", "")

    if not email_addr or not password:
        return json.dumps({"error": "Account credentials incomplete (missing email or password)."})

    # Build message body with optional signature
    full_body = body
    if append_signature and signature:
        full_body += f"\n\n--\n{signature}"

    msg = MIMEMultipart()
    if display_name:
        msg["From"] = f"{display_name} <{email_addr}>"
    else:
        msg["From"] = email_addr
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.attach(MIMEText(full_body, "plain"))

    server = None
    try:
        if smtp_port == 587:
            server = smtplib.SMTP(smtp_server, smtp_port, timeout=30)
            server.ehlo()
            server.starttls()
        else:
            server = smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=30)

        server.login(email_addr, password)
        server.sendmail(email_addr, recipients, msg.as_string())

        return json.dumps({
            "status": "sent",
            "message": f"Email sent successfully from {email_addr}.",
            "details": {
                "from": email_addr,
                "subject": subject,
                "recipients": recipients,
            },
        })
    except smtplib.SMTPAuthenticationError as e:
        return json.dumps({"error": f"Authentication failed for {email_addr}: {e}"})
    except smtplib.SMTPException as e:
        return json.dumps({"error": f"SMTP error: {e}"})
    except Exception as e:
        return json.dumps({"error": f"Failed to send email: {e}"})
    finally:
        if server:
            try:
                server.quit()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Tool class (follows runtime tool protocol)
# ---------------------------------------------------------------------------

class EmailTool:
    """Send emails through configured email accounts (SMTP or API relay)."""

    def __init__(self):
        self._session = requests.Session()

    @staticmethod
    def definition() -> Dict[str, Any]:
        return {
            "name": "email",
            "description": (
                "Send emails through configured email accounts via SMTP. "
                "Supports multiple accounts with per-agent defaults.\n\n"
                "ACTIONS:\n"
                "- send: compose and send an email. Requires subject, body, "
                "recipients. The first call returns a PREVIEW for user approval "
                "(confirmation gate). Call again with confirmation='confirmed' "
                "to actually send. Uses SMTP directly with API relay fallback.\n"
                "- status: check configured accounts, server health, and "
                "API relay connectivity.\n"
                "- accounts: list all configured email accounts (passwords masked).\n\n"
                "ACCOUNT RESOLUTION (for send):\n"
                "1. Explicit account_id if provided\n"
                "2. Agent-specific default account\n"
                "3. Global default account\n"
                "4. First configured account\n\n"
                "Always confirm recipient addresses with the user before sending "
                "unless they were explicitly provided."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["send", "status", "accounts"],
                        "description": (
                            "'send' — compose and send an email (2-step: preview then confirm). "
                            "'status' — check email configuration and server health. "
                            "'accounts' — list available email accounts."
                        ),
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject line. Required for 'send'.",
                    },
                    "body": {
                        "type": "string",
                        "description": (
                            "Email body text. Required for 'send'. "
                            "The account's signature is auto-appended if configured."
                        ),
                    },
                    "recipients": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of recipient email addresses. Required for 'send'. "
                            "Must be valid email addresses (contain @ and .)."
                        ),
                    },
                    "account_id": {
                        "type": "string",
                        "description": (
                            "Optional: ID of the email account to send from. "
                            "If omitted, the system auto-selects: agent default → "
                            "global default → first account."
                        ),
                    },
                    "confirmation": {
                        "type": "string",
                        "description": (
                            "Set to 'confirmed' to actually send the email. "
                            "On the first call to 'send', omit this field — you'll "
                            "receive a preview with from/subject/body/recipients. "
                            "Show the preview to the user, and if they approve, "
                            "call send again with confirmation='confirmed'."
                        ),
                    },
                },
                "required": ["action"],
            },
        }

    def execute(self, arguments: Dict[str, Any], agent_name: str = "", user_id: str = "") -> str:
        action = arguments.get("action", "status")

        if action == "status":
            return self._check_status(user_id=user_id)
        elif action == "send":
            return self._send_email(arguments, agent_name=agent_name, user_id=user_id)
        elif action == "accounts":
            return self._list_accounts(user_id=user_id)
        else:
            return json.dumps({"error": f"Unknown action: {action}"})

    # ---- accounts listing ----
    def _list_accounts(self, user_id: str = "") -> str:
        """List all configured email accounts for this user (passwords masked)."""
        accounts = get_accounts(user_id) if user_id else []
        if not accounts:
            return json.dumps({
                "accounts": [],
                "message": "No email accounts configured. Add accounts in the Tools page.",
            })
        return json.dumps({"accounts": accounts, "total": len(accounts)})

    # ---- status ----
    def _check_status(self, user_id: str = "") -> str:
        """Check email accounts and optional API server health."""
        cfg = get_effective_config(user_id=user_id)
        base_url = cfg["api_base_url"]
        timeout = cfg["timeout"]
        accounts = get_accounts(user_id) if user_id else []

        result: Dict[str, Any] = {
            "accounts_configured": len(accounts),
            "accounts": accounts,
        }

        # Check optional API server connectivity
        try:
            resp = self._session.get(f"{base_url}/", timeout=timeout)
            resp.raise_for_status()
            result["api_server_running"] = True
            result["api_server_url"] = base_url
        except Exception:
            result["api_server_running"] = False
            result["api_server_note"] = (
                "API relay server not running (not required — "
                "emails are sent directly via SMTP when accounts are configured)."
            )

        return json.dumps(result)

    # ---- send ----
    def _send_email(self, arguments: Dict[str, Any], agent_name: str = "", user_id: str = "") -> str:
        """Send an email using the best available account for this user."""
        subject = (arguments.get("subject") or "").strip()
        body = (arguments.get("body") or "").strip()
        recipients = arguments.get("recipients", [])
        account_id = (arguments.get("account_id") or "").strip()
        confirmation = (arguments.get("confirmation") or "").strip().lower()

        # Validate required fields
        if not subject:
            return json.dumps({"error": "Email subject is required."})
        if not body:
            return json.dumps({"error": "Email body is required."})
        if not recipients or not isinstance(recipients, list):
            return json.dumps({"error": "At least one recipient email address is required."})

        # Basic email format validation
        invalid = [r for r in recipients if "@" not in r or "." not in r]
        if invalid:
            return json.dumps({
                "error": f"Invalid email address(es): {', '.join(invalid)}",
            })

        # Resolve which account to use (scoped to this user)
        account = None
        if account_id:
            account = get_account_by_id(user_id, account_id) if user_id else None
            if not account:
                return json.dumps({"error": f"Email account '{account_id}' not found."})
        else:
            # Try agent default → global default → first account
            if agent_name and user_id:
                account = get_agent_default_account(user_id, agent_name)
            if not account and user_id:
                account = get_default_account(user_id)

        if not account:
            return json.dumps({
                "error": "No email accounts configured. Add an account in the Tools page.",
            })

        # Confirmation gate
        cfg = get_effective_config(user_id=user_id)
        if cfg.get("require_confirmation", True) and confirmation != "confirmed":
            preview_from = account.get("email", "unknown")
            sig = account.get("signature", "")
            return json.dumps({
                "gate": "awaiting_confirmation",
                "message": (
                    "Email ready to send. Please confirm with the user before sending. "
                    "Show them the details below, and if approved, call this tool again "
                    "with confirmation='confirmed'."
                ),
                "preview": {
                    "from_account": account.get("label", preview_from),
                    "from_email": preview_from,
                    "subject": subject,
                    "body": body,
                    "recipients": recipients,
                    "signature": sig if sig else "(none)",
                },
            })

        # ── Send via direct SMTP (preferred) ──
        smtp_result = _send_via_smtp(account, subject, body, recipients)
        result_data = json.loads(smtp_result)

        # If SMTP fails, try API relay as fallback
        if "error" in result_data:
            api_result = self._send_via_api(subject, body, recipients)
            api_data = json.loads(api_result)
            if api_data.get("status") == "sent":
                return api_result
            # Both failed — return original SMTP error with API note
            result_data["api_fallback_attempted"] = True
            result_data["api_fallback_error"] = api_data.get("error", "API relay also failed")
            return json.dumps(result_data)

        return smtp_result

    # ---- API relay (fallback when SMTP fails) ----
    def _send_via_api(self, subject: str, body: str, recipients: List[str]) -> str:
        """Fallback: send through the FastAPI email service."""
        cfg = get_effective_config()
        base_url = cfg["api_base_url"]
        timeout = cfg["timeout"]

        try:
            self._session.get(f"{base_url}/", timeout=timeout)
        except requests.exceptions.ConnectionError:
            return json.dumps({
                "error": f"Cannot connect to email server at {base_url}.",
            })

        payload = {"subject": subject, "body": body, "recipients": recipients}
        try:
            resp = self._session.post(
                f"{base_url}/send_email",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=timeout,
            )
            if resp.status_code == 200:
                return json.dumps({
                    "status": "sent",
                    "message": "Email sent successfully via API relay.",
                    "details": resp.json(),
                })
            else:
                detail = "Unknown error"
                try:
                    detail = resp.json().get("detail", detail)
                except Exception:
                    detail = resp.text[:200]
                return json.dumps({
                    "status": "failed",
                    "error": f"Failed to send email: {detail}",
                    "http_status": resp.status_code,
                })
        except requests.exceptions.Timeout:
            return json.dumps({"error": f"Email server timed out after {timeout}s."})
        except Exception as exc:
            return json.dumps({"error": f"Error sending email: {str(exc)}"})
