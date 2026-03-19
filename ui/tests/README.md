# tests/

Comprehensive test suite for the OrionForge agent runtime. **265 test functions, ~4,061 assertions** across 12 files.

## Test Files

| File | Functions | Checks | What It Tests |
|------|-----------|--------|---------------|
| `test_torture.py` | 118 | ~2,961 | Deep torture of all code paths — memory, vault, sort, policy, tools, templates, model router, presets, 6-tier routing, sidecar wiring, soul script helpers, soul script API, soul script FAISS indexing, note collector soul script injection, profiles template collapsible, admin keys, chat 3-mode selector, user model catalog, `__userkey_` dynamic connections, Stripe state persistence, store catalog structure, tier & trial system, credit system, credit cost estimators, purchase flows (tool/skin/agent), agent ownership, user activity tracking, wipe user data, purge inactive, list all users, auth helpers, tier info structure |
| `test_memory.py` | 23 | 155 | VaultStore CRUD, scoping, PII guard, bulk delete, versioning, resolve_latest, compact, stats, Memory dataclass, taxonomy constants, tiers & topics, tags & source, JSONL format |
| `test_stress.py` | 15 | 94 | Rapid-fire operations, concurrent access, boundary conditions, cross-module integration |
| `test_directives.py` | 14 | 108 | Parser, store search, store list/get, scoping, injector, directives tool, scoring, manifest generation, save/load, helpers, diff, audit, changes action |
| `test_registry_and_tools.py` | 17 | 86 | Tool registry dispatch, resolution, listing, error paths, cost tracker, web search tool |
| `test_governance.py` | 16 | 74 | ActiveDirectives (record/list/ids/summary/reset), validate_manifest (schema/enums/duplicates/SHA-256 drift) |
| `test_chunker_injector.py` | 14 | 51 | Pure chunking logic, merge/split, formatting helpers |
| `test_metering.py` | 11 | 92 | Token accounting, cost computation, log persistence, aggregation |
| `test_storage_and_llm.py` | 14 | 45 | HTML stripping, note loading, LLMResponse dataclass |
| `test_data_paths.py` | 5 | 31 | Canonical data directory layout, auto-creation, isolation, edge cases |
| `test_tools.py` | 4 | 34 | EchoTool, ContinuationUpdateTool, EmailTool, RuntimePolicy |
| `run_all.py` | — | — | Master runner — executes all suites in dependency order, consolidates results |

**Total: 265 functions, ~4,061 checks across 12 test suites**

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
- All 10 tool implementations (memory, directives, echo, continuation, email, web search, inbox, cost tracker + registry)
- Vault sort logic (8 modes × dict & object forms × edge cases)
- Max memory limit & utilization calculation
- Template rendering (vault.html sort dropdown, metadata, unlimited display; tools.html max memory dropdown)
- Memory profile configuration & saved profile upgrade
- Dynamic scopes & category policy
- Boundary policy, PII guard, runtime policy clamping
- Manifest validation, audit, diff
- LLM client factory, metering helpers, data paths
- Store catalog structure, tier & trial system, credit system
- Credit cost estimators, purchase flows (tool/skin/agent), agent ownership
- User activity tracking, wipe user data, purge inactive, list all users
- Auth helpers (public paths, config, token extraction), tier info structure


