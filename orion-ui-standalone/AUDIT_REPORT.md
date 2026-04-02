# OrionForge — Comprehensive Audit Report

**Generated:** 2025  
**Scope:** `orion-ui-standalone/tests/` (11 test files + runner) and `orion-ui-standalone/web/templates/` (12 HTML files)  
**Backend reference:** `orion-ui-standalone/web/app.py` (3558 lines)

---

## Table of Contents

1. [Test Suite Audit](#1-test-suite-audit)
   - [1.1 Coverage Analysis](#11-coverage-analysis)
   - [1.2 Test Framework Issues](#12-test-framework-issues)
   - [1.3 Import & Path Issues](#13-import--path-issues)
   - [1.4 Stale References & Empty Tests](#14-stale-references--empty-tests)
   - [1.5 Flaky Test Risks](#15-flaky-test-risks)
2. [Templates Audit](#2-templates-audit)
   - [2.1 Jinja2 Variable Issues](#21-jinja2-variable-issues)
   - [2.2 JavaScript / fetch() Error Handling](#22-javascript--fetch-error-handling)
   - [2.3 CSRF & Security](#23-csrf--security)
   - [2.4 Hardcoded URLs & External Dependencies](#24-hardcoded-urls--external-dependencies)
   - [2.5 Accessibility Issues](#25-accessibility-issues)
   - [2.6 Code Quality & Maintenance](#26-code-quality--maintenance)
3. [Backend Issues (app.py)](#3-backend-issues-apppy)

---

## 1. Test Suite Audit

### 1.1 Coverage Analysis

**32 substantive src modules** (excluding `__init__.py` files) → **27 tested, 5 untested**

| # | Module | Tested By | Status |
|---|--------|-----------|--------|
| 1 | `src/data_paths.py` | test_data_paths, test_tools, test_stress, test_torture | ✅ |
| 2 | `src/runtime_policy.py` | test_tools, test_stress, test_torture | ✅ |
| 3 | `src/directives/parser.py` | test_directives, test_stress, test_torture | ✅ |
| 4 | `src/directives/store.py` | test_directives, test_stress, test_torture | ✅ |
| 5 | `src/directives/injector.py` | test_directives, test_torture | ✅ |
| 6 | `src/directives/manifest.py` | test_directives, test_stress, test_torture | ✅ |
| 7 | `src/governance/active_directives.py` | test_governance, test_stress, test_torture | ✅ |
| 8 | `src/llm_client/base.py` | test_storage_and_llm, test_torture | ✅ |
| 9 | `src/llm_client/factory.py` | test_torture | ✅ |
| 10 | **`src/llm_client/anthropic_client.py`** | — | ❌ NOT TESTED |
| 11 | **`src/llm_client/ollama.py`** | — | ❌ NOT TESTED |
| 12 | **`src/llm_client/openai_compat.py`** | — | ❌ NOT TESTED |
| 13 | `src/memory/vault.py` | test_memory, test_stress, test_torture | ✅ |
| 14 | `src/memory/types.py` | test_memory, test_stress, test_torture | ✅ |
| 15 | `src/memory/pii_guard.py` | test_memory, test_stress, test_torture | ✅ |
| 16 | **`src/memory/notes_faiss.py`** | — | ❌ NOT TESTED |
| 17 | **`src/memory/load_and_index.py`** | — | ❌ NOT TESTED |
| 18 | `src/memory/injector.py` | test_torture | ✅ |
| 19 | `src/memory/faiss_memory.py` | — (mocked in test_torture) | ⚠️ MOCK ONLY |
| 20 | `src/memory/chunker.py` | test_chunker_injector, test_stress, test_torture | ✅ |
| 21 | `src/observability/metering.py` | test_metering, test_stress, test_torture | ✅ |
| 22 | `src/policy/boundary.py` | test_torture | ✅ |
| 23 | `src/storage/user_notes_loader.py` | test_storage_and_llm, test_torture | ✅ |
| 24 | `src/storage/note_collector.py` | test_torture | ✅ |
| 25 | `src/tools/memory_tool.py` | test_torture | ✅ |
| 26 | `src/tools/inbox.py` | test_torture | ✅ |
| 27 | `src/tools/email_tool.py` | test_tools, test_torture | ✅ |
| 28 | `src/tools/echo.py` | test_tools, test_torture | ✅ |
| 29 | `src/tools/registry.py` | test_registry_and_tools, test_stress, test_torture | ✅ |
| 30 | `src/tools/directives_tool.py` | test_directives, test_stress, test_torture | ✅ |
| 31 | `src/tools/web_search.py` | test_registry_and_tools, test_torture | ✅ |
| 32 | `src/tools/cost_tracker.py` | test_registry_and_tools, test_torture | ✅ |

**Effective coverage: ~84% (27/32) of modules have at least one test.**

#### Untested Modules — Detailed Findings

| File | Severity | Finding | Impact |
|------|----------|---------|--------|
| `src/llm_client/anthropic_client.py` | **CRITICAL** | Zero test coverage for Anthropic API client | Regressions in Anthropic chat will go undetected |
| `src/llm_client/ollama.py` | **CRITICAL** | Zero test coverage for Ollama client | Local LLM integration untested |
| `src/llm_client/openai_compat.py` | **CRITICAL** | Zero test coverage for OpenAI-compatible client | Primary cloud LLM path untested |
| `src/memory/notes_faiss.py` | **WARNING** | No direct tests (only `invalidate_notes_faiss` called in test_torture) | Soul Script retrieval pipeline untested |
| `src/memory/load_and_index.py` | **WARNING** | Zero test coverage | FAISS index build/load untested |
| `src/memory/faiss_memory.py` | **WARNING** | Only mocked—never instantiated with real FAISS index in tests | Vector search correctness unverified |

---

### 1.2 Test Framework Issues

| File | Severity | Line | Finding |
|------|----------|------|---------|
| All test files | **WARNING** | — | Custom test framework (global `PASS`/`FAIL` counters + `check()` helper) instead of pytest. No test discovery, no fixtures, no parameterization, no parallel execution. |
| `tests/run_all.py` | **INFO** | ~20-45 | Uses `subprocess.run()` to execute each test file — no shared process, no aggregated reporting to CI, exit code is sum of failures (can overflow). |
| All test files | **WARNING** | — | Tests use `check(label, condition)` which prints but never raises — a crashing import or exception mid-file silently skips remaining tests with no indication. |
| All test files | **INFO** | — | No `setUp`/`tearDown` patterns — temp directories are created ad-hoc and cleaned up inline. |

---

### 1.3 Import & Path Issues

| File | Severity | Line | Finding |
|------|----------|------|---------|
| All 11 test files | **WARNING** | ~10-14 | Every test file uses `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` — fragile and may mask import errors in production. Standard practice is `pip install -e .` or a conftest.py. |
| `test_torture.py` | **INFO** | Various | Has 90+ `from src.*` imports scattered inside individual test functions (lazy imports). These succeed even if the module has breaking changes at the import level. |
| `test_stress.py` | **INFO** | Various | Same lazy import pattern — import errors only surface when that specific test function is called. |

---

### 1.4 Stale References & Empty Tests

| File | Severity | Line | Finding |
|------|----------|------|---------|
| `test_torture.py` | **INFO** | 5144 lines | Massively long file (5144 lines). Should be split into focused test modules for maintainability. |
| `test_stress.py` | **INFO** | 1224 lines | Large stress test file. The `stress_settings_avatar_profile_skin_tools` tests at the end (line ~700+) test web API routes via `fetch`-style URL patterns but don't actually make HTTP requests — they test internal path behavior. |
| All test files | **INFO** | — | No test for `web/app.py` routes, `web/image_gen.py`, or any template rendering. The entire web layer is untested. |

---

### 1.5 Flaky Test Risks

| File | Severity | Line | Finding |
|------|----------|------|---------|
| `test_memory.py` | **WARNING** | Various | Tests write to temp JSONL files and rely on ordering. If two test runs overlap or temp cleanup fails, phantom data may leak. |
| `test_stress.py` | **WARNING** | ~60-145 | Rapid vault CRUD (200 ops) writes/reads in tight loops on JSONL — can fail on slow I/O or concurrent file access. |
| `test_stress.py` | **WARNING** | ~496 | Metering stress test computes 1000 token usage objects in sequence — timing-sensitive assertions on float precision may drift across platforms. |
| `test_torture.py` | **WARNING** | ~595-800 | MemoryTool tests monkey-patch internal module globals (e.g., `mt_mod._get_faiss_memory`) — if test ordering changes, patches may leak. |
| `test_registry_and_tools.py` | **WARNING** | ~116-165 | Tests monkeypatch `src.tools.registry._PROFILES_DIR` — global state mutation may affect subsequent tests. |

---

## 2. Templates Audit

### 2.1 Jinja2 Variable Issues

| File | Severity | Line | Finding |
|------|----------|------|---------|
| `pricing.html` | **CRITICAL** | 186-203 | Template uses `{{ cost_stats }}`, `{{ pricing }}`, `{{ connections }}` — but the `/pricing` route in app.py (line 861) is a **redirect** to `/tools#cost_tracker`. This template is orphaned and cannot be rendered via any active route. If loaded directly, it would crash with undefined variables. |
| `skins.html` | **INFO** | 281 | Uses `{{ active_skin }}` — correctly provided by the `/skins` route (line 869). ✅ |
| `agi_loop.html` | **INFO** | 299 | Uses `{{ config.profile }}`, `{{ config.max_loops }}` — correctly provided by `/agi-loop` route (line 1521). ✅ |
| `about.html` | **INFO** | 297 | Uses `{{ about_text }}`, `{{ wiki_articles | tojson }}` — correctly provided by `/about` route (line 1696). ✅ |
| `knowledge_edit.html` | **INFO** | Various | Uses `{{ note.* }}`, `{{ folders }}` — correctly provided by `/knowledge/{note_id}/edit` route (line 843). ✅ |
| `vault.html` | **INFO** | Various | Uses `{{ stats }}`, `{{ memories }}`, `{{ scopes }}`, etc. — correctly provided by `/vault` route (line 820). ✅ |
| `chat.html` | **INFO** | Various | Uses `{{ agents }}`, `{{ connections }}`, `{{ chat_index }}`, `{{ avatar_map }}`, `{{ user_profile }}`, `{{ pricing }}`, `{{ chat_background }}` — all correctly provided by `/chat` route (line 680). ✅ |
| `profiles.html` | **INFO** | Various | Uses `{{ agents }}`, `{{ agent_data }}`, `{{ all_models }}`, `{{ notes }}`, `{{ avatar_map }}`, `{{ user_profile }}`, `{{ connections }}`, `{{ builtin_notes }}`, `{{ all_tool_names }}`, `{{ tool_display }}` — all correctly provided by `/profiles` route (line 740). ✅ |
| `settings.html` | **INFO** | Various | Uses `{{ settings }}`, `{{ tab }}`, `{{ connections }}` — correctly provided by `/settings` route (line 851). ✅ |
| `tools.html` | **INFO** | Various | Uses 12+ template variables — all correctly provided by `/tools` route (line 1117). ✅ |

---

### 2.2 JavaScript / fetch() Error Handling

**80+ fetch() calls across all templates.** Most `async` functions have `try/catch` blocks, but several critical paths lack error handling:

| File | Severity | Line | Finding |
|------|----------|------|---------|
| `vault.html` | **WARNING** | 468 | `deleteSingle()` — no try/catch around `fetch('/api/vault/delete')`. Network errors will throw uncaught promise rejections. |
| `vault.html` | **WARNING** | 478 | `deleteSelected()` — same issue, no try/catch for bulk delete fetch. |
| `vault.html` | **WARNING** | 484 | `compactVault()` — no try/catch around `fetch('/api/vault/compact')`. |
| `knowledge.html` | **WARNING** | 249 | `createNote()` — no try/catch around `fetch('/api/knowledge', {method:'POST'})`. |
| `knowledge.html` | **WARNING** | 260 | `deleteNote()` — no try/catch around `fetch('/api/knowledge/' + id, {method:'DELETE'})`. |
| `knowledge.html` | **WARNING** | 268-290 | Folder CRUD operations (`createFolder`, `renameFolder`, `deleteFolder`) — no try/catch blocks. |
| `knowledge_edit.html` | **WARNING** | 95 | `saveNote()` — no try/catch around `fetch('/api/knowledge/' + _noteId)`. Save failures silently ignored. |
| `profiles.html` | **WARNING** | 1186 | `uploadAvatar()` — no try/catch; network failure crashes silently. |
| `profiles.html` | **WARNING** | 1243-1267 | `clearImage()`, `setUserColor()`, `saveUserField()` — no try/catch on fetch calls to `/api/profiles/*`. |
| `profiles.html` | **WARNING** | 1409 | `saveConfig()` — fetch to `/api/profiles/${agent}/config` has no error handling. |
| `profiles.html` | **WARNING** | 1441-1550 | Knowledge attach/detach operations — fetch calls without try/catch. |
| `profiles.html` | **WARNING** | 1767 | Photo editor save — no try/catch on avatar position save. |
| `profiles.html` | **WARNING** | 1885 | `createAgent()` — no try/catch on `/api/profiles/create`. |
| `pricing.html` | **WARNING** | 448 | `saveAllPricing()` — no try/catch on fetch to `/api/pricing`. |
| `pricing.html` | **WARNING** | 466 | `deleteModel()` — no try/catch. |
| `chat.html` | **INFO** | Most | Chat.html generally has good try/catch coverage around fetch calls. ✅ |
| `settings.html` | **INFO** | Most | Settings save functions mostly have try/catch. ✅ |
| `tools.html` | **INFO** | Most | Tool config saves have try/catch around critical fetches. ✅ |
| `skins.html` | **INFO** | 619 | Skin apply — has try/catch. ✅ |

---

### 2.3 CSRF & Security

| File | Severity | Line | Finding |
|------|----------|------|---------|
| All templates | **WARNING** | — | No CSRF protection on any API endpoint. All state-changing operations (POST/PUT/DELETE) use JSON fetch() without CSRF tokens. FastAPI does not include CSRF middleware by default. Since this is a local-first app (localhost), risk is lower but still allows CSRF from any website the user visits. |
| `knowledge_edit.html` | **WARNING** | 79 | `contenteditable="true"` div — user HTML input rendered with `{{ note.content_html | safe }}`. If content_html contains `<script>` tags, they execute. This is a stored XSS vector. |
| `about.html` | **INFO** | 297 | `{{ about_text }}` rendered inside a `<textarea>` — safely escaped by Jinja2 in attribute context. ✅ |
| `vault.html` | **INFO** | Various | `{{ search_query }}` rendered in an `<input value="">` — auto-escaped by Jinja2. ✅ |
| `settings.html` | **WARNING** | Various | API keys are rendered as `value="{{ settings.get('tts', {}).get('elevenlabs_api_key', '') }}"` — keys appear in page source. Should use `type="password"` (some already do). |

---

### 2.4 Hardcoded URLs & External Dependencies

| File | Severity | Line | Finding |
|------|----------|------|---------|
| `base.html` | **WARNING** | ~7-12 | Loads Tailwind CSS from `https://cdn.tailwindcss.com` — external CDN dependency. Page will lack styling if CDN is unreachable. Dev mode only (Tailwind CDN should not be used in production). |
| `base.html` | **WARNING** | ~13-16 | Loads `marked.min.js` and `highlight.min.js` from `https://cdnjs.cloudflare.com` — external CDN dependencies. Markdown rendering and syntax highlighting fail if CDN is down. |
| `base.html` | **WARNING** | ~17 | Loads `github-dark.min.css` highlight.js theme from CDN. |
| All templates | **INFO** | — | All API endpoints use relative paths (`/api/...`) — good practice. ✅ |
| `settings.html` | **INFO** | 303 | Hardcoded preset for Ollama URL `http://localhost:11434` — appropriate as a default. ✅ |

---

### 2.5 Accessibility Issues

| File | Severity | Line | Finding |
|------|----------|------|---------|
| All templates | **WARNING** | — | Extensive use of `onclick` handlers on `<button>` and `<div>` elements without keyboard alternatives (`onkeydown`, `role`, `tabindex`). Non-button clickable elements are not keyboard-accessible. |
| `base.html` | **WARNING** | Sidebar | Navigation links in sidebar use `<a>` with `href` — good. But the sidebar toggle button lacks `aria-label`. |
| `skins.html` | **WARNING** | ~280+ | Skin cards use `onclick` on `<div>` elements — not keyboard focusable. Missing `role="button"`, `tabindex="0"`, and `onkeydown` handlers. |
| `profiles.html` | **WARNING** | Various | Color picker swatches are `<div onclick>` — not keyboard accessible. Avatar context menu items use `<button>` — good. |
| `vault.html` | **WARNING** | Various | Filter dropdown, manage toggle, sort controls — mix of buttons (good) and styled elements without proper ARIA roles. |
| `chat.html` | **WARNING** | Various | Chat sidebar items, context menu, tool toggles — `<div>` and `<button>` with proper semantics generally. Some icon-only buttons (sidebar toggle, send) lack `aria-label`. |
| `knowledge.html` | **WARNING** | Various | Folder sidebar items are `<div onclick>` — not keyboard accessible. Note cards use `onclick` on `<div>`. |
| `tools.html` | **WARNING** | Various | Tool cards use `<div onclick>` with `cursor:pointer` — not keyboard accessible. |
| `agi_loop.html` | **WARNING** | Various | Control buttons use proper `<button>` elements. Pipeline stages are non-interactive — OK. |
| All templates | **WARNING** | — | No `skip-to-content` link for screen reader users. |
| All templates | **WARNING** | — | Color contrast may not meet WCAG AA in several places (e.g., `#52525b` text on `#0f0f14` background ≈ 3.5:1 ratio, below 4.5:1 minimum). |
| `profiles.html` | **INFO** | ~693 | `<img src="{{ up.image }}" alt="You">` — has alt text. ✅ |
| `knowledge_edit.html` | **INFO** | — | Emoji button has `title` attribute. ✅ |

---

### 2.6 Code Quality & Maintenance

| File | Severity | Line | Finding |
|------|----------|------|---------|
| `tools.html` | **INFO** | 5645 lines | Extremely large template (5645 lines). Contains complex multi-tab tool configuration UI with embedded JSON data structures. Should be split into partials/includes. |
| `chat.html` | **INFO** | 2434 lines | Very large template. Contains full chat engine, TTS/STT integration, markdown rendering, streaming parser, sidebar management. |
| `profiles.html` | **INFO** | 1907 lines | Very large template. Agent profile editor, photo editor, knowledge attachment, tools toggle — all in one file. |
| `agi_loop.html` | **INFO** | 1493 lines | Large template for AGI loop dashboard, pipeline visualization, tier editor modal, budget management. |
| `base.html` | **INFO** | 522 lines | Contains the full skin system CSS override engine (~300 lines of CSS variable injection). The `applySkinGlobal()` function is a 250+ line template literal with CSS selectors for every component. |
| `skins.html` | **INFO** | 700 lines | Skin catalog with 12+ theme definitions hardcoded in JavaScript. |
| `pricing.html` | **CRITICAL** | All | **Orphaned template.** The `/pricing` route redirects to `/tools#cost_tracker` (app.py line 861). This file is dead code that cannot be rendered. |
| All templates | **INFO** | — | Styling uses a mix of Tailwind utility classes and custom `<style>` blocks — inconsistent approach increases maintenance burden. |
| `base.html` | **INFO** | — | Skin system injects CSS variables at runtime via JavaScript `document.head.append(style)` — works but means initial page load always shows default theme, then flickers to selected skin. |

---

## 3. Backend Issues (app.py)

These issues were discovered while verifying template variable correctness:

| File | Severity | Line | Finding |
|------|----------|------|---------|
| `web/app.py` | **WARNING** | 24, 29 | Duplicate `import uuid` statement. |
| `web/app.py` | **INFO** | 3558 lines | Monolithic file — 3558 lines containing all routes, helpers, data access, prompt assembly, and API endpoints. Should be split into route modules. |
| `web/app.py` | **INFO** | 861 | `/pricing` redirects to `/tools#cost_tracker` but `pricing.html` still exists — dead template should be removed or the redirect removed. |

---

## Summary Statistics

### Test Suite
| Metric | Value |
|--------|-------|
| Total test files | 11 + 1 runner |
| Total test lines | ~10,800 |
| Modules with coverage | 27/32 (84%) |
| Critical gaps | 3 LLM client modules (anthropic, ollama, openai_compat) |
| Warning gaps | 3 memory/FAISS modules (notes_faiss, load_and_index, faiss_memory) |
| Web layer tests | 0 |

### Templates
| Metric | Value |
|--------|-------|
| Total template files | 12 |
| Total template lines | ~14,800 |
| Orphaned templates | 1 (pricing.html) |
| fetch() calls without error handling | ~25+ |
| CSRF protection | None |
| XSS vectors | 1 (knowledge_edit.html `| safe`) |
| Accessibility issues | Pervasive (keyboard nav, ARIA, contrast) |
| External CDN dependencies | 3 (Tailwind, marked.js, highlight.js) |

---

## 4. AGI Loop Tab — Recent Fixes (2025)

A comprehensive audit and fix pass addressed the following issues in the AGI Loop tab (`agi_loop.html`, `app.py`, `src/tools/agi_loop.py`):

### 4.1 Agent Switching (Fixed)

| File | Finding | Fix |
|------|---------|-----|
| `agi_loop.html` | `switchTab()` relied on `event.currentTarget` — broke on programmatic calls from `switchTabById()` | Refactored to accept explicit `(id, btnEl)` params; falls back to querySelector if no button passed |
| `agi_loop.html` | `switchTabById()` used fragile `.textContent` matching to find tab buttons | Replaced with index-based lookup map for reliable programmatic tab switching |

### 4.2 Config Persistence (Fixed)

| File | Finding | Fix |
|------|---------|-----|
| `agi_loop.html` | `saveConfig()` gathered config without tier data — save/reload wiped all tiers | `saveConfig()` now always includes `_tiers` via `JSON.parse(JSON.stringify(_tiers))` |
| `app.py` | `stale_streak_limit` missing from `_AGI_LOOP_DEFAULTS` and `AGILoopConfigUpdate` Pydantic model | Added `stale_streak_limit: int = 2` to both defaults dict and Pydantic model |

### 4.3 Storage Display (Fixed)

| File | Finding | Fix |
|------|---------|-----|
| `agi_loop.html` | Storage card showed `data/orion/` paths with "Local" badge — misleading since Fly.io VM uses `/persist/` | Updated all file path references to `/persist/orion/…` with green "VM Volume" badge |

### 4.4 Journal Popup Modal (New Feature)

| File | Finding | Fix |
|------|---------|-----|
| `agi_loop.html` | Journal entries only displayed truncated text in a cramped side panel | Added full popup modal (`jnlOpenModal()`, `jnlOpenMultiModal()`, `jnlCloseModal()`) with scrollable narrative, tool call details, and metadata grid |
| `agi_loop.html` | New CSS: `.jnl-modal-overlay`, `.jnl-modal`, `.jnl-modal-head`, `.jnl-modal-body`, `.jnl-modal-narrative`, `.jnl-modal-section`, `.jnl-modal-meta-grid/card` | Escape key closes modal; click-outside dismisses |

### 4.5 Expandable Loop Log (New Feature)

| File | Finding | Fix |
|------|---------|-----|
| `agi_loop.html` | Loop log entries truncated at 300 chars with no way to see full content | `_renderTickLog()` now creates expandable `.loop-entry` elements with full response text and tool call details on click |

### 4.6 Tick History Disk Persistence (Fixed)

| File | Finding | Fix |
|------|---------|-----|
| `src/tools/agi_loop.py` | `tick_history` only lived in memory — lost on restart | Added `load_history_from_disk()` method (mirrors `load_journal_from_disk()`); called at module init alongside journal load |
| `app.py` | `/api/agi-loop/history` returned empty array after restart | Added disk-load fallback: if in-memory history is empty, calls `state.load_history_from_disk()` before responding |

### 4.7 Test Coverage Added

New torture tests cover:
- `load_history_from_disk()` — persist, reload, cap at 200, corrupt JSONL recovery, empty file handling
- `stale_streak_limit` in config defaults and Pydantic model
- Journal popup modal HTML elements in template
- Expandable loop log entry elements in template
- VM storage path references in template

---

### Priority Recommendations

1. **CRITICAL:** Add tests for LLM client modules (`anthropic_client.py`, `ollama.py`, `openai_compat.py`)
2. **CRITICAL:** Remove or reactivate orphaned `pricing.html` template
3. **CRITICAL:** Sanitize `content_html` in `knowledge_edit.html` to prevent stored XSS
4. **HIGH:** Add try/catch error handling to all fetch() calls (25+ unprotected)
5. **HIGH:** Add tests for `notes_faiss.py`, `load_and_index.py`, and real FAISS integration
6. **HIGH:** Consider migrating to pytest for proper test discovery, fixtures, and CI integration
7. **MEDIUM:** Bundle CDN dependencies locally for offline reliability
8. **MEDIUM:** Add CSRF middleware (e.g., `starlette-csrf` or token-based approach)
9. **MEDIUM:** Add keyboard accessibility to interactive `<div>` elements
10. **LOW:** Split large templates into Jinja2 includes/partials
11. **LOW:** Split `app.py` into route modules
12. **LOW:** Split `test_torture.py` into focused test files
