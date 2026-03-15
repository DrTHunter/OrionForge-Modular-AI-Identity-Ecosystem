# tests/

Comprehensive test suite for the OrionForge agent runtime. **246 test functions, ~3,600 assertions** across 11 test files.

## Test Files

| File | Functions | Checks | What It Tests |
|------|-----------|--------|---------------|
| `test_torture.py` | 99 | ~2,550 | Deep torture of every code path — memory tool (13 actions), vault sort (8 modes, dict & object), max memory limits, utilization calc, template rendering (vault, tools, profiles, skins, about), boundary policy, PII guard, runtime policy, manifest system, directive parser/store/injector, tool registry, EmailTool, WebSearchTool, InboxTool, cost tracker, metering, LLM client factory, dynamic scopes, category policy, saved profiles, 6-tier model router, coding tiers, escalation chains, budget tracking, **sidecar service wiring** (SearXNG, TTS, Whisper — env-var override, URL normalization, fallback behavior, timeout config), **soul script helpers** (_load/_save round-trip, dir creation, unicode), **soul script API** (config endpoint save/update/empty/combined), **soul script FAISS indexing** (rebuild, doc_id format, agent discovery), **note collector soul script injection**, **profiles template collapsible sections** (toggleCollapse, FAISS badge, soul-script textarea) |
| `test_memory.py` | 23 | 155 | VaultStore CRUD, scoping, PII guard, bulk delete, versioning, resolve_latest, compact, stats, Memory dataclass, taxonomy constants, tiers & topics, tags & source, JSONL format |
| `test_stress.py` | 29 | ~398 | Rapid-fire operations, concurrent access, boundary conditions, cross-module integration, router presets, coding tier routing |
| `test_directives.py` | 14 | 108 | Parser, store search, store list/get, scoping, injector, directives tool, scoring, manifest generation, save/load, helpers, diff, audit, changes action |
| `test_registry_and_tools.py` | 17 | 86 | Tool registry dispatch, resolution, listing, error paths, cost tracker, web search tool |
| `test_governance.py` | 16 | 74 | ActiveDirectives (record/list/ids/summary/reset), validate_manifest (schema/enums/duplicates/SHA-256 drift) |
| `test_chunker_injector.py` | 14 | 51 | Pure chunking logic, merge/split, formatting helpers |
| `test_metering.py` | 11 | 92 | Token accounting, cost computation, log persistence, aggregation |
| `test_storage_and_llm.py` | 14 | 45 | HTML stripping, note loading, LLMResponse dataclass |
| `test_data_paths.py` | 5 | 31 | Canonical data directory layout, auto-creation, isolation, edge cases |
| `test_tools.py` | 4 | 38 | EchoTool, ContinuationUpdateTool, EmailTool, RuntimePolicy |
| `run_all.py` | — | — | Master runner — executes all suites in dependency order, consolidates results |

**Total: 246 functions, ~3,600 checks across 11 test suites**

## Running Tests

```powershell
# Run the comprehensive torture test (fastest way to verify everything)
cd orion-ui-standalone
$env:PYTHONIOENCODING="utf-8"; python tests/test_torture.py

# Run ALL test suites via the master runner
python tests/run_all.py

# Run individual suites
python tests/test_memory.py
python tests/test_directives.py
python tests/test_governance.py
python tests/test_stress.py
```

## Syntax Check (all Python files)

```bash
python -c "
import py_compile, glob
files = glob.glob('src/**/*.py', recursive=True) + glob.glob('tests/**/*.py', recursive=True)
for f in sorted(files):
    py_compile.compile(f, doraise=True)
print('All files OK')
"
```

## Test Framework

Tests use a lightweight manual framework (no pytest dependency). Each test function calls `check(label, condition)` which prints `[PASS]`/`[FAIL]` and tracks global counters. Exit code 1 if any failures.

## Coverage Notes

The `test_torture.py` suite alone covers the most code paths and is the best single test to run for regression. It exercises:
- All 11 tool implementations (memory, directives, echo, continuation, email, web search, inbox, cost tracker, model router, agi loop, runtime info) + registry
- 6-tier model router (LOCAL_CHEAP, LOCAL_STRONG, CHEAP_CLOUD, EXPENSIVE_CLOUD, CODE_LIGHT, CODE_HEAVY)
- Task classification, escalation chains, force tier, budget tracking
- Vault sort logic (8 modes × dict & object forms × edge cases)
- Max memory limit & utilization calculation
- Template rendering (vault.html sort dropdown, metadata, unlimited display; tools.html max memory dropdown; profiles.html collapsible sections + soul script; skins.html; about.html)
- Memory profile configuration & saved profile upgrade
- Dynamic scopes & category policy
- Boundary policy, PII guard, runtime policy clamping
- Manifest validation, audit, diff
- LLM client factory, metering helpers, data paths
- **Soul script helpers** — `_load_soul_script` / `_save_soul_script` round-trip, directory auto-creation, unicode, overwrite, empty save
- **Soul script API** — `PUT /api/profiles/{name}/config` with `soul_script_text`: save, update, empty, combined with system_prompt, preservation when omitted
- **Soul script FAISS indexing** — `_rebuild_notes_faiss()` indexes soul scripts with `__soul_script__{agent}` doc_ids, empty scripts skipped, agent discovery
- **Note collector soul script injection** — `collect_notes()` auto-adds soul script doc_id for active agent
- **Profiles template collapsible** — toggleCollapse JS, collapsible-header/body, soul-script textarea, FAISS badge, soul_script_text in saveAll
- **Sidecar service wiring** — SearXNG, openedai-speech TTS, faster-whisper STT: env-var override (`SEARXNG_URL`, `TTS_URL`, `WHISPER_URL`), URL normalization (trailing slash stripping), fallback to `connections.json`, timeout configuration
- **Auth paywall template checks** — login, plans, store, admin_keys pages render without crash
- Profile API torture — CRUD, trash cycle, config patching, avatar uploads, user profile, soul script save
- Skin API and saved-profile CRUD endpoints


