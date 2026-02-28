"""Web search tool — search the web via SearXNG and scrape page content.

Uses a local SearXNG Docker instance (default http://localhost:3000/search)
to perform web searches, then scrapes and extracts the main content from
the top results.  Supports three modes:
  - fast:   quick factual lookups  (2 pages, 1 200 words)
  - normal: typical questions      (5 pages, 1 500 words)
  - deep:   research-level queries (8 pages, 3 000 words)

Also provides a single-page scrape action for fetching a specific URL.
Settings can be overridden in config/settings.json → tool_config.web_search.
"""

import json
import os
import re
import unicodedata
import concurrent.futures
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests

try:
    from bs4 import BeautifulSoup
    _BS4 = True
except ImportError:
    _BS4 = False

try:
    import trafilatura
    _TRAF = True
except ImportError:
    _TRAF = False


# ── Defaults ─────────────────────────────────────────────────────

_DEFAULT_SEARXNG_URL = os.environ.get(
    "SEARXNG_URL", "http://localhost:3000/search"
)

_DEFAULT_IGNORED_SITES = (
    "facebook.com,instagram.com,twitter.com,x.com,pinterest.com,"
    "tiktok.com,linkedin.com,reddit.com,scribd.com,slideshare.net,"
    "docplayer.net,pdfcoffee.com,yumpu.com,issuu.com,"
    "stackprinter.appspot.com,stackovernet.com,stackoverrun.com,"
    "stackoom.com,thinbug.com,reposhub.com,tutorialspoint.com,"
    "javatpoint.com,guru99.com,simplilearn.com,w3resource.com,"
    "codegrepper.com,brainly.com,studocu.com,coursehero.com,"
    "chegg.com,answers.com,answers.yahoo.com,ask.com,"
    "quora.com,investopedia.com,techtarget.com,w3schools.com,"
    "wikihow.com"
)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

_MODE_PRESETS: Dict[str, Dict[str, int]] = {
    "fast":   {"pages": 2, "return_count": 2, "word_limit": 1200},
    "normal": {"pages": 5, "return_count": 3, "word_limit": 1500},
    "deep":   {"pages": 8, "return_count": 5, "word_limit": 3000},
}

_SKIP_SEARCH_SIGNALS = [
    "i already know",
    "from my training",
    "general knowledge",
    "common knowledge",
    "well-known fact",
    "no search needed",
]

_SETTINGS_FILE = Path(__file__).resolve().parent.parent.parent / "config" / "settings.json"


# ── Config helpers ───────────────────────────────────────────────

def _load_tool_config() -> Dict[str, Any]:
    if not _SETTINGS_FILE.exists():
        return {}
    try:
        with open(_SETTINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("tool_config", {}).get("web_search", {})
    except Exception:
        return {}


def get_effective_config() -> Dict[str, Any]:
    """Return merged config (defaults + user overrides from settings.json)."""
    saved = _load_tool_config()
    return {
        "searxng_url": saved.get("searxng_url", _DEFAULT_SEARXNG_URL),
        "ignored_sites": saved.get("ignored_sites", _DEFAULT_IGNORED_SITES),
        "require_justification": saved.get("require_justification", True),
        "modes": {
            mode: {
                k: saved.get("modes", {}).get(mode, {}).get(k, defaults[k])
                for k in ("pages", "return_count", "word_limit")
            }
            for mode, defaults in _MODE_PRESETS.items()
        },
    }


def _get_mode_preset(mode: str) -> tuple:
    cfg = get_effective_config()
    m = cfg["modes"].get(mode, cfg["modes"]["normal"])
    return m["pages"], m["return_count"], m["word_limit"]


# ── Text extraction helpers ──────────────────────────────────────

def _clean_text(text: str) -> str:
    if any(ch in text for ch in ("<", ">", "&lt;", "&gt;")):
        if _BS4:
            text = BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def _remove_emojis(text: str) -> str:
    return "".join(c for c in text if not unicodedata.category(c).startswith("So"))


def _truncate(text: str, word_limit: int) -> str:
    words = text.split()
    return text if len(words) <= word_limit else " ".join(words[:word_limit]) + "..."


def _extract_content(html: str, word_limit: int) -> str:
    """Extract main page content, preferring trafilatura then BS4."""
    if _TRAF:
        try:
            extracted = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=False,
                favor_precision=True,
                include_links=False,
                no_fallback=False,
            )
            if extracted:
                return _truncate(_clean_text(extracted), word_limit)
        except Exception:
            pass

    if not _BS4:
        return _truncate(_clean_text(html), word_limit)

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    main = None
    for sel in ("main", "article", "[role='main']", ".content",
                ".main-content", ".post-content", ".entry-content",
                "#content", "#main"):
        main = soup.select_one(sel)
        if main:
            break
    node = main or soup
    return _truncate(_clean_text(node.get_text(" ", strip=True)), word_limit)


def _fetch_page(url: str, word_limit: int, ignored_sites: str) -> Optional[dict]:
    """Download a URL and return structured result or None on failure."""
    base = urlparse(url).netloc
    if ignored_sites:
        for site in ignored_sites.split(","):
            if site.strip() in base:
                return None
    try:
        resp = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=15)
        resp.raise_for_status()
        content = _extract_content(resp.text, word_limit)
        title = "No title"
        if _BS4:
            soup = BeautifulSoup(resp.text, "html.parser")
            if soup.title and soup.title.string:
                title = _remove_emojis(soup.title.string.strip())
        return {"title": title, "url": url, "content": content}
    except requests.exceptions.RequestException:
        return None


# ── Tool class ───────────────────────────────────────────────────

class WebSearchTool:
    """Search the web via SearXNG and optionally scrape page content."""

    @staticmethod
    def definition() -> Dict[str, Any]:
        return {
            "name": "web_search",
            "description": (
                "Search the web using SearXNG and retrieve relevant page content. "
                "IMPORTANT: Before searching, assess whether you already have "
                "sufficient knowledge to answer. Fill in 'knowledge_check' with "
                "what you already know and 'reason' with why a web search is needed. "
                "If you can confidently answer from training data, do NOT use this tool.\n\n"
                "Use action 'search' with a query and mode (fast/normal/deep).\n"
                "Use action 'scrape' with a url to fetch a single page."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["search", "scrape"],
                        "description": (
                            "'search' — query SearXNG and scrape top results. "
                            "'scrape' — fetch and extract content from a single URL."
                        ),
                    },
                    "knowledge_check": {
                        "type": "string",
                        "description": (
                            "State what you already know about this topic. "
                            "If you can answer confidently, say so and skip searching."
                        ),
                    },
                    "reason": {
                        "type": "string",
                        "description": (
                            "Why the internet is needed: real-time data, beyond "
                            "training cutoff, verify uncertain info, user asked, "
                            "need specific URL."
                        ),
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query (required for 'search').",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["fast", "normal", "deep"],
                        "description": (
                            "'fast' (2 pages), 'normal' (5 pages), "
                            "'deep' (8 pages). Default: normal."
                        ),
                    },
                    "url": {
                        "type": "string",
                        "description": "URL to scrape (required for 'scrape').",
                    },
                },
                "required": ["action"],
            },
        }

    def execute(self, arguments: Dict[str, Any]) -> str:
        action = arguments.get("action", "search")

        if action == "scrape":
            return self._scrape(arguments)

        # Knowledge gate
        cfg = get_effective_config()
        if cfg.get("require_justification", True):
            gate = self._knowledge_gate(arguments)
            if gate is not None:
                return gate

        return self._search(arguments)

    # ── Knowledge gate ──

    def _knowledge_gate(self, arguments: Dict[str, Any]) -> Optional[str]:
        kc = (arguments.get("knowledge_check") or "").strip().lower()
        reason = (arguments.get("reason") or "").strip()

        if kc:
            for signal in _SKIP_SEARCH_SIGNALS:
                if signal in kc:
                    return json.dumps({
                        "gate": "blocked",
                        "message": (
                            "You indicated you already have this knowledge. "
                            "Answer from training data instead."
                        ),
                    })

        if not reason:
            return json.dumps({
                "gate": "missing_justification",
                "message": (
                    "Web search requires a 'reason'. Valid: real-time data, "
                    "beyond training cutoff, verify uncertain info, user asked, "
                    "need specific URL. If you can answer without searching, do so."
                ),
            })

        return None

    # ── Search ──

    def _search(self, arguments: Dict[str, Any]) -> str:
        query = arguments.get("query", "").strip()
        if not query:
            return json.dumps({"error": "No query provided."})

        mode = arguments.get("mode", "normal")
        scrape_count, return_count, word_limit = _get_mode_preset(mode)
        cfg = get_effective_config()
        searxng_url = cfg["searxng_url"]
        ignored_sites = cfg["ignored_sites"]

        # Query SearXNG
        try:
            resp = requests.get(
                searxng_url,
                params={"q": query, "format": "json", "number_of_results": scrape_count},
                headers={"User-Agent": _USER_AGENT},
                timeout=60,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])[:scrape_count]
        except requests.exceptions.RequestException as exc:
            return json.dumps({
                "error": f"SearXNG request failed: {exc}",
                "hint": "Is SearXNG running? Start it with: docker start searxng",
            })

        if not results:
            return json.dumps({"results": [], "message": "No results found."})

        # Scrape pages in parallel
        collected: list[dict] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(_fetch_page, r["url"], word_limit, ignored_sites): r
                for r in results
            }
            for future in concurrent.futures.as_completed(futures):
                page = future.result()
                if page and len(collected) < return_count:
                    snippet = futures[future].get("content", "")
                    if snippet:
                        page["snippet"] = _remove_emojis(snippet)
                    collected.append(page)

        return json.dumps(
            {"query": query, "mode": mode, "results": collected},
            ensure_ascii=False, separators=(",", ":"),
        )

    # ── Scrape ──

    def _scrape(self, arguments: Dict[str, Any]) -> str:
        url = arguments.get("url", "").strip()
        if not url:
            return json.dumps({"error": "No URL provided."})

        _, _, word_limit = _get_mode_preset("normal")
        page = _fetch_page(url, word_limit, "")
        if page is None:
            return json.dumps({"error": f"Failed to fetch {url}"})
        return json.dumps(page, ensure_ascii=False, separators=(",", ":"))
