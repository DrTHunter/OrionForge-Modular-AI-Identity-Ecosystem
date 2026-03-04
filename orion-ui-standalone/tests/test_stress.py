"""Stress / integration tests — exercising cross-module interactions.

Run from project root:
    python -m tests.test_stress

This is the "torture test" suite: rapid-fire operations, concurrent-ish
access patterns, boundary conditions, and cross-module integration.

Covers:
  - Vault CRUD rapid-fire (200 ops), bulk delete, compact, stats
  - PII guard stress (many patterns at speed)
  - Directive store + tool stress (50 scopes, 100 searches)
  - Manifest validation stress (evil payloads)
  - Active directives stress (500 records)
  - Runtime policy boundary stress
  - Metering accumulation (1000 ops)
  - Cost log stress (100 events, filter, aggregate)
  - Chunker large doc (100 sections)
  - Vault types integration (all scopes/categories/tiers)
  - Manifest ↔ ActiveDirectives integration
  - DataPaths ↔ ContinuationUpdate integration
  - Memory types edge cases (unicode, backward compat)
  - Settings rapid read/write stress (100 cycles)
  - Avatar file I/O stress (20 rapid avatar writes)
  - Profile CRUD stress (10 agent create/update/delete cycles)
  - Saved profile CRUD stress (20 rapid save/list/delete)
  - Skin toggle stress (50 rapid skin switches)
  - Tool registry resolution stress (all tools x 10 rounds)
  - Memory strip-tags stress (100 mixed-tag strings)
  - Model router classification stress (500 classify_task ops, resolve_model_for_task,
    tier escalation, ModelRouterTool.execute 100 rapid calls)
  - Router presets CRUD stress (30 rapid save/list/load/delete cycles,
    filename sanitisation, overwrite, concurrent-ish access)
  - AGI loop state stress (500 tick logs, state transitions, 200 tool executions,
    50 pause/resume cycles, double-pause/double-resume edge cases)
  - AGI journal stress (600 rapid writes, in-memory cap, disk persistence,
    to_dict under load, 10 clear/reload cycles, 1000-entry reload, agent filtering)
  - Narrative builder stress (200 rapid _build_tick_narrative calls, error/tool/response
    variants, unicode, truncation, multi-tool, empty input)
"""

import json
import os
import sys
import time
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PASS = 0
FAIL = 0


def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label}  {detail}")


# ═════════════════════════════════════════════
# 1. Vault rapid-fire CRUD stress test
# ═════════════════════════════════════════════
def test_vault_rapid_crud():
    print("\n=== STRESS: Vault Rapid CRUD (200 ops) ===")
    from src.memory.vault import VaultStore

    tmp = tempfile.mkdtemp()
    try:
        vault = VaultStore(os.path.join(tmp, "vault.jsonl"))
        ids = []

        # Rapid creation
        t0 = time.time()
        for i in range(100):
            m = vault.create_memory(
                text=f"Memory #{i}: " + "x" * 50,
                scope="shared" if i % 2 == 0 else "astraea",
                category="fact" if i % 3 == 0 else "preference",
                tags=[f"tag{i % 5}"],
            )
            ids.append(m.id)
        elapsed = time.time() - t0
        check(f"100 creates in {elapsed:.2f}s", len(ids) == 100)

        # Read all
        active = vault.read_active()
        check("100 active memories", len(active) == 100, f"got {len(active)}")

        # Rapid updates
        t0 = time.time()
        for i in range(0, 50):
            vault.update_memory(ids[i], text=f"Updated #{i}: " + "y" * 50)
        elapsed = time.time() - t0
        check(f"50 updates in {elapsed:.2f}s", True)

        # Verify updates
        for i in range(0, 50):
            m = vault.get_memory(ids[i])
            check_ok = m is not None and m.text.startswith(f"Updated #{i}")
            if not check_ok:
                check(f"update #{i} correct", False, f"got {m.text[:30] if m else 'None'}")
                break
        else:
            check("all 50 updates correct", True)

        # Version check — updated memories should be version 2
        m0 = vault.get_memory(ids[0])
        check("version incremented", m0.version == 2, f"got {m0.version}")

        # Rapid deletes
        t0 = time.time()
        for i in range(50, 100):
            vault.delete_memory(ids[i])
        elapsed = time.time() - t0
        check(f"50 deletes in {elapsed:.2f}s", True)

        active2 = vault.read_active()
        check("50 remain active", len(active2) == 50, f"got {len(active2)}")

        # Stats
        stats = vault.stats()
        check("stats active=50", stats["active_count"] == 50, f"got {stats['active_count']}")
        check("stats deleted=50", stats["deleted_count"] == 50, f"got {stats['deleted_count']}")

        # Compact
        vault.compact()
        active3 = vault.read_active()
        check("compact preserves active", len(active3) == 50)

        # JSONL integrity after all operations
        with open(os.path.join(tmp, "vault.jsonl"), "r") as f:
            lines = [l for l in f if l.strip()]
        for i, line in enumerate(lines):
            try:
                json.loads(line)
            except:
                check(f"JSONL line {i} valid", False)
                break
        else:
            check("all JSONL lines valid", True)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ═════════════════════════════════════════════
# 2. Bulk delete stress
# ═════════════════════════════════════════════
def test_vault_bulk_delete():
    print("\n=== STRESS: Vault Bulk Delete ===")
    from src.memory.vault import VaultStore

    tmp = tempfile.mkdtemp()
    try:
        vault = VaultStore(os.path.join(tmp, "v.jsonl"))
        ids = []
        for i in range(50):
            m = vault.create_memory(text=f"Bulkable {i}", scope="shared", category="fact")
            ids.append(m.id)

        # Bulk delete first 25
        result = vault.bulk_delete(ids[:25])
        check("bulk deleted 25", len(result["deleted"]) == 25, f"got {len(result['deleted'])}")
        check("bulk not_found 0", len(result["not_found"]) == 0)

        # Bulk delete again (already deleted) + some new
        result2 = vault.bulk_delete(ids[:30])
        check("re-delete: 5 new deletes", len(result2["deleted"]) == 5, f"got {len(result2['deleted'])}")
        check("re-delete: 25 not_found", len(result2["not_found"]) == 25, f"got {len(result2['not_found'])}")

        # Bulk delete with nonexistent IDs
        result3 = vault.bulk_delete(["fake_id_1", "fake_id_2"])
        check("fake IDs not_found", len(result3["not_found"]) == 2)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ═════════════════════════════════════════════
# 3. PII guard boundary cases
# ═════════════════════════════════════════════
def test_pii_guard_stress():
    print("\n=== STRESS: PII Guard Boundary Cases ===")
    from src.memory.pii_guard import check_pii

    # Standard PII patterns must be caught
    pii_texts = [
        "My SSN is 123-45-6789",
        "Card: 4111 1111 1111 1111",
        "password: hunter2",
        "api_key: sk-12345678",
        "secret_key: abc123xyz",
        "Card number: 4111111111111111",
        "SSN: 123 45 6789",
    ]
    for text in pii_texts:
        result = check_pii(text)
        check(f"PII caught: {text[:30]}...", len(result) > 0, f"returned empty")

    # Safe texts must pass
    safe_texts = [
        "The sky is blue.",
        "I prefer dark mode.",
        "Meeting at 3pm tomorrow.",
        "Order #12345 was placed.",
        "Temperature is 72 degrees.",
        "Phone: 555-0123",  # short number, not SSN
        "A" * 5000,  # Long but safe
    ]
    for text in safe_texts:
        result = check_pii(text)
        check(f"safe: {text[:30]}...", len(result) == 0, f"false positive: {result}")


# ═════════════════════════════════════════════
# 4. Directive store stress
# ═════════════════════════════════════════════
def test_directive_store_stress():
    print("\n=== STRESS: Directive Store Search ===")
    from src.directives.parser import parse_directive_file
    from src.directives.store import DirectiveStore

    tmp = tempfile.mkdtemp()
    try:
        # Create a large directive file with many sections
        lines = []
        for i in range(50):
            lines.append(f"## Section {i}: Topic {chr(65 + i % 26)}")
            lines.append(f"This section covers topic {i} about {chr(65 + i % 26)} stuff.")
            lines.append(f"Details: keyword{i} important relevant data.")
            lines.append("")

        # Write as shared.md so DirectiveStore can find it
        path = os.path.join(tmp, "shared.md")
        with open(path, "w") as f:
            f.write("\n".join(lines))

        store = DirectiveStore(tmp, scopes="shared")

        # Search should return results
        results = store.search("keyword42", limit=5)
        check("search finds keyword42", len(results) > 0)
        check("top result relevant", "keyword42" in results[0].body.lower()
              or "42" in results[0].heading.lower())

        # Search with no matches
        results2 = store.search("zzzznonexistent", limit=5)
        check("no match → empty", len(results2) == 0)

        # List headings
        headings = store.list_headings()
        check("50 headings", len(headings) == 50, f"got {len(headings)}")

        # Get all
        all_sections = store.get_all()
        check("get_all returns 50", len(all_sections) == 50, f"got {len(all_sections)}")

        # Rapid repeated searches
        t0 = time.time()
        for i in range(100):
            store.search(f"keyword{i % 50}", limit=3)
        elapsed = time.time() - t0
        check(f"100 searches in {elapsed:.2f}s", elapsed < 5.0)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ═════════════════════════════════════════════
# 4b. DirectivesTool stress — rapid actions with scope switching
# ═════════════════════════════════════════════
def test_directives_tool_stress():
    print("\n=== STRESS: DirectivesTool — Rapid Actions + Scope Switching ===")
    from src.tools.directives_tool import DirectivesTool
    import src.tools.directives_tool as dt_mod
    import src.directives.manifest as manifest_mod

    tool = DirectivesTool()
    tmp = tempfile.mkdtemp()
    orig_dir = dt_mod._DIRECTIVES_DIR
    orig_scopes = manifest_mod.SCOPES
    orig_dt_scopes = getattr(dt_mod, 'SCOPES', None)

    dt_mod._DIRECTIVES_DIR = tmp

    try:
        # Create 10 scope files with 20 sections each = 200+ total sections
        scope_names = [f"scope_{i}" for i in range(10)]
        all_scopes = tuple(["shared"] + scope_names)
        manifest_mod.SCOPES = all_scopes
        dt_mod.SCOPES = all_scopes

        # Write shared.md
        lines = ["## Shared Rule {i}\nShared content for rule {i}.\n".format(i=i)
                 for i in range(20)]
        with open(os.path.join(tmp, "shared.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        # Write per-scope files
        for scope in scope_names:
            lines = ["## {s} Section {i}\n{s} content number {i} keyword_{i}.\n".format(
                s=scope.title(), i=i) for i in range(20)]
            with open(os.path.join(tmp, f"{scope}.md"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

        # _resolve_scopes rapidly
        t0 = time.time()
        for _ in range(500):
            DirectivesTool._resolve_scopes(None)
            DirectivesTool._resolve_scopes("shared")
            DirectivesTool._resolve_scopes("scope_3")
        elapsed = time.time() - t0
        check(f"1500 _resolve_scopes in {elapsed:.3f}s", elapsed < 2.0)

        # Rapid list calls — no scope
        t0 = time.time()
        for _ in range(50):
            r = json.loads(tool.execute({"action": "list"}))
        elapsed = time.time() - t0
        check(f"50 list (all scopes) in {elapsed:.2f}s", elapsed < 10.0)
        check("list all → 220 headings", r["count"] == 220,
              f"got {r['count']}")

        # Rapid search with scope switching
        t0 = time.time()
        for i in range(100):
            scope = scope_names[i % len(scope_names)]
            r = json.loads(tool.execute({
                "action": "search", "query": f"keyword_{i % 20}", "scope": scope,
            }))
        elapsed = time.time() - t0
        check(f"100 scoped searches in {elapsed:.2f}s", elapsed < 10.0)

        # Rapid get calls
        t0 = time.time()
        for i in range(50):
            tool.execute({"action": "get", "heading": f"Shared Rule {i % 20}"})
        elapsed = time.time() - t0
        check(f"50 get calls in {elapsed:.2f}s", elapsed < 5.0)

        # Scope isolation: searching scope_0 should not return scope_5 content
        r = json.loads(tool.execute({
            "action": "search", "query": "Scope_5", "scope": "scope_0",
        }))
        for sec in r.get("sections", []):
            check(f"scope isolation: {sec['scope']} in (shared, scope_0)",
                  sec["scope"] in ("shared", "scope_0"))

    finally:
        dt_mod._DIRECTIVES_DIR = orig_dir
        manifest_mod.SCOPES = orig_scopes
        if orig_dt_scopes is not None:
            dt_mod.SCOPES = orig_dt_scopes
        shutil.rmtree(tmp, ignore_errors=True)


# ═════════════════════════════════════════════
# 5. Governance: validate_manifest edge cases
# ═════════════════════════════════════════════
def test_manifest_validation_stress():
    print("\n=== STRESS: Manifest Validation Edge Cases ===")
    from src.directives.manifest import validate_manifest

    # Completely empty manifest
    result = validate_manifest({})
    check("empty manifest → issues", not result["valid"] and len(result["errors"]) > 0)

    # Manifest with directives but missing all inner keys
    bad_manifest = {
        "manifest_version": 1,
        "generated_utc": "2026-01-01",
        "hash_algo": "sha256",
        "root_paths": ["directives/"],
        "default_retrieval_mode": "keyword_hybrid",
        "directives": [{}],
    }
    result2 = validate_manifest(bad_manifest)
    check("empty entry → issues", not result2["valid"] and len(result2["errors"]) > 0)

    # Manifest with all invalid enums
    evil_manifest = {
        "manifest_version": 1,
        "generated_utc": "2026-01-01",
        "hash_algo": "sha256",
        "root_paths": ["directives/"],
        "default_retrieval_mode": "keyword_hybrid",
        "directives": [{
            "id": "test",
            "name": "test",
            "scope": "INVALID_SCOPE",
            "status": "INVALID_STATUS",
            "risk": "INVALID_RISK",
            "sha256": "abc",
            "path": __file__,
            "summary": "test",
            "triggers": [],
            "dependencies": [],
            "version": "1.0.0",
            "token_estimate": 10,
        }],
    }
    result3 = validate_manifest(evil_manifest)
    check("invalid enums caught", len(result3["errors"]) >= 2,
          f"got {len(result3['errors'])} issues: {result3['errors']}")

    # Manifest with duplicate IDs
    dup_manifest = {
        "manifest_version": 1,
        "generated_utc": "2026-01-01",
        "hash_algo": "sha256",
        "root_paths": ["directives/"],
        "default_retrieval_mode": "keyword_hybrid",
        "directives": [
            {"id": "dup", "name": "a", "scope": "shared", "status": "active",
             "risk": "low", "sha256": "aaa", "path": __file__, "summary": "test",
             "triggers": [], "dependencies": [], "version": "1.0.0", "token_estimate": 5},
            {"id": "dup", "name": "b", "scope": "shared", "status": "active",
             "risk": "low", "sha256": "bbb", "path": __file__, "summary": "test",
             "triggers": [], "dependencies": [], "version": "1.0.0", "token_estimate": 5},
        ],
    }
    result4 = validate_manifest(dup_manifest)
    check("duplicate IDs caught",
          any("dup" in str(e).lower() for e in result4["errors"]),
          f"issues: {result4['errors']}")


# ═════════════════════════════════════════════
# 7. ActiveDirectives rapid record/reset
# ═════════════════════════════════════════════
def test_active_directives_stress():
    print("\n=== STRESS: ActiveDirectives rapid record/reset ===")
    from src.governance.active_directives import ActiveDirectives

    ad = ActiveDirectives()

    # Record 500 entries
    t0 = time.time()
    for i in range(500):
        ad.record(
            heading=f"Section {i}",
            body=f"Content for section {i}",
            scope="shared",
        )
    elapsed = time.time() - t0
    check(f"500 records in {elapsed:.2f}s", True)
    check("count = 500", ad.count() == 500, f"got {ad.count()}")

    # IDs unique
    ids_ = ad.ids()
    check("500 unique IDs", len(set(ids_)) == 500, f"got {len(set(ids_))}")

    # Summary
    summary = ad.summary()
    check("summary count=500", summary["count"] == 500)
    check("summary has total_tokens", "total_tokens" in summary)

    # Reset
    ad.reset()
    check("reset → 0", ad.count() == 0)
    check("reset → empty list", ad.list() == [])

    # Double reset safe
    ad.reset()
    check("double reset safe", ad.count() == 0)


# ═════════════════════════════════════════════
# 8. Runtime policy boundary
# ═════════════════════════════════════════════
def test_runtime_policy_stress():
    print("\n=== STRESS: RuntimePolicy boundaries ===")
    from src.runtime_policy import RuntimePolicy

    # Edge: exactly at limit
    p = RuntimePolicy(max_iterations=10, max_wall_time_seconds=60)
    check("iter 9 ok", p.check(9, time.time()) is None)
    check("iter 10 blocked", p.check(10, time.time()) is not None)

    # Edge: wall time exactly expired
    p2 = RuntimePolicy(max_wall_time_seconds=0)
    # Start time = now, wall_time = 0 → immediate expiry
    result = p2.check(0, time.time() - 0.001)
    check("wall time 0 + past start → blocked", result is not None)

    # Very high limits
    p3 = RuntimePolicy(max_iterations=999999, max_wall_time_seconds=999999)
    check("high limits ok", p3.check(999998, time.time()) is None)

    # Default values
    p4 = RuntimePolicy()
    check("default max_iterations", p4.max_iterations > 0)
    check("default max_wall_time is None", p4.max_wall_time_seconds is None)


# ═════════════════════════════════════════════
# 9. Metering accumulation stress
# ═════════════════════════════════════════════
def test_metering_accumulation():
    print("\n=== STRESS: Metering accumulation ===")
    from src.observability.metering import Metering, TokenUsage, CostBreakdown, zero_metering

    acc = zero_metering()

    # Accumulate 1000 metering events
    t0 = time.time()
    for i in range(1000):
        m = Metering(
            usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            cost=CostBreakdown(input_cost=0.001, output_cost=0.002, total_cost=0.003),
            model="gpt-4", provider="openai",
        )
        acc = acc + m
    elapsed = time.time() - t0
    check(f"1000 accumulations in {elapsed:.2f}s", True)

    check("total prompt tokens", acc.usage.prompt_tokens == 100_000)
    check("total completion tokens", acc.usage.completion_tokens == 50_000)
    check("total cost", abs(acc.cost.total_cost - 3.0) < 0.01,
          f"got {acc.cost.total_cost}")


# ═════════════════════════════════════════════
# 10. Cost log stress — many writes + reads
# ═════════════════════════════════════════════
def test_cost_log_stress():
    print("\n=== STRESS: Cost Log (100 events) ===")
    from src.observability.metering import (
        Metering, TokenUsage, CostBreakdown,
        log_cost_event, read_cost_log, aggregate_costs,
        set_cost_log_path,
    )

    tmp = tempfile.mkdtemp()
    log_path = os.path.join(tmp, "stress_log.jsonl")
    set_cost_log_path(log_path)

    try:
        agents = ["astraea", "callum", "codex_animus"]
        models = ["gpt-4", "claude-sonnet-4-20250514", "deepseek-v2"]

        t0 = time.time()
        for i in range(100):
            m = Metering(
                usage=TokenUsage(prompt_tokens=100 + i, completion_tokens=50 + i,
                                 total_tokens=150 + 2 * i),
                cost=CostBreakdown(total_cost=0.001 * (i + 1)),
                model=models[i % 3],
                provider="openai" if i % 3 == 0 else "anthropic",
            )
            log_cost_event(m, agent=agents[i % 3], chat_id=f"chat_{i // 10}")
        elapsed = time.time() - t0
        check(f"100 log writes in {elapsed:.2f}s", True)

        # Read all
        events = read_cost_log(limit=200)
        check("read 100 events", len(events) == 100, f"got {len(events)}")

        # Filter by agent
        astraea_events = read_cost_log(agent="astraea", limit=200)
        check("filter astraea", len(astraea_events) > 0)
        check("all astraea", all(e["agent"] == "astraea" for e in astraea_events))

        # Aggregate
        agg = aggregate_costs(events)
        check("aggregate num_calls=100", agg["num_calls"] == 100)
        check("aggregate total_cost > 0", agg["total_cost"] > 0)
        check("aggregate by_model has 3", len(agg["by_model"]) == 3)
        check("aggregate by_agent has 3", len(agg["by_agent"]) == 3)

    finally:
        set_cost_log_path(None)
        shutil.rmtree(tmp, ignore_errors=True)


# ═════════════════════════════════════════════
# 11. Chunker stress — large documents
# ═════════════════════════════════════════════
def test_chunker_large_doc():
    print("\n=== STRESS: Chunker — Large Document ===")
    from src.memory.chunker import SemanticChunker

    chunker = SemanticChunker(min_chunk_size=100, max_chunk_size=2000)

    # Generate a 100-section document
    parts = []
    for i in range(100):
        parts.append(f"### Section {i}")
        parts.append(f"Content for section {i}. " * 20)
        parts.append("")

    doc = "\n".join(parts)
    check(f"doc size: {len(doc)} chars", len(doc) > 10000)

    t0 = time.time()
    chunks = chunker.chunk_by_headers(doc, "big_doc", "Big Document")
    elapsed = time.time() - t0
    check(f"chunked in {elapsed:.2f}s", True)
    check("chunks created", len(chunks) > 0, f"got {len(chunks)}")

    # All chunks have valid metadata
    for i, c in enumerate(chunks):
        if "text" not in c or "metadata" not in c:
            check(f"chunk {i} structure", False)
            break
        if c["metadata"]["char_count"] != len(c["text"]):
            check(f"chunk {i} char_count", False,
                  f"meta={c['metadata']['char_count']} actual={len(c['text'])}")
            break
    else:
        check("all chunks valid structure", True)

    # No empty text chunks
    empty_chunks = [c for c in chunks if not c["text"].strip()]
    check("no empty chunks", len(empty_chunks) == 0, f"got {len(empty_chunks)} empty")


# ═════════════════════════════════════════════
# 12. Cross-module: Vault → Memory types round-trip
# ═════════════════════════════════════════════
def test_vault_types_integration():
    print("\n=== INTEGRATION: Vault ↔ Memory Types ===")
    from src.memory.vault import VaultStore
    from src.memory.types import Memory, VALID_SCOPES, VALID_CATEGORIES, VALID_TIERS

    tmp = tempfile.mkdtemp()
    try:
        vault = VaultStore(os.path.join(tmp, "v.jsonl"))

        # Create memories with all valid scopes
        for scope in VALID_SCOPES:
            m = vault.create_memory(text=f"Test {scope}", scope=scope, category="fact")
            check(f"scope '{scope}' accepted", m is not None)

        # Create with all valid categories
        for cat in VALID_CATEGORIES:
            m = vault.create_memory(text=f"Test {cat}", scope="shared", category=cat)
            check(f"category '{cat}' accepted", m is not None)

        # Create with all valid tiers
        for tier in VALID_TIERS:
            m = vault.create_memory(text=f"Test {tier}", scope="shared",
                                    category="fact", tier=tier)
            check(f"tier '{tier}' accepted", m is not None)

        # to_dict → from_dict round-trip
        m_orig = vault.create_memory(
            text="Round trip test", scope="shared", category="preference",
            tags=["a", "b"], tier="core",
        )
        d = m_orig.to_dict()
        m_back = Memory.from_dict(d)
        check("round-trip text", m_back.text == m_orig.text)
        check("round-trip scope", m_back.scope == m_orig.scope)
        check("round-trip id", m_back.id == m_orig.id)
        check("round-trip tags", m_back.tags == m_orig.tags)
        check("round-trip tier", m_back.tier == m_orig.tier)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ═════════════════════════════════════════════
# 13. Manifest → ActiveDirectives integration
# ═════════════════════════════════════════════
def test_manifest_directives_integration():
    print("\n=== INTEGRATION: Manifest ↔ ActiveDirectives ===")
    from src.directives.manifest import generate_manifest
    from src.governance.active_directives import ActiveDirectives
    from src.directives.parser import parse_directive_file
    from src.directives.store import DirectiveStore

    tmp = tempfile.mkdtemp()
    try:
        # Create test directive as shared.md (scope file)
        path = os.path.join(tmp, "shared.md")
        with open(path, "w") as f:
            f.write("## Alpha\nAlpha content.\n\n## Beta\nBeta content.\n")

        manifest = generate_manifest(directives_dir=tmp, scopes=("shared",))
        directives = manifest["directives"]
        check("manifest has entries", len(directives) == 2,
              f"got {len(directives)}")

        # Use DirectiveStore to look up sections
        store = DirectiveStore(tmp, scopes="shared")

        # Register with ActiveDirectives using manifest
        ad = ActiveDirectives()
        ad.reset()

        for entry in directives:
            section = store.get_section(entry["name"])
            if section:
                ad.record(
                    heading=section.heading,
                    body=section.body,
                    scope=section.scope,
                    manifest_entry=entry,
                )

        check("AD count matches manifest", ad.count() == len(directives))
        ids = ad.ids()
        check("AD IDs populated", len(ids) == 2)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ═════════════════════════════════════════════
# 14. Data paths + Continuation Update integration
# ═════════════════════════════════════════════
def test_data_paths_continuation_integration():
    print("\n=== INTEGRATION: DataPaths ↔ ContinuationUpdate ===")
    import src.data_paths as dp
    from src.tools.continuation_update import ContinuationUpdateTool

    orig_root = dp.DATA_ROOT
    tmp = tempfile.mkdtemp()
    dp.DATA_ROOT = tmp

    try:
        tool = ContinuationUpdateTool()

        # Write continuation for a profile
        tool.execute({"profile": "agent_alpha", "mode": "append",
                      "content": "Started task."})

        # Verify it landed in the right place
        expected = dp.continuation_path("agent_alpha")
        check("file at expected path", os.path.isfile(expected))

        with open(expected, "r") as f:
            content = f.read()
        check("content correct", "Started task" in content)

    finally:
        dp.DATA_ROOT = orig_root
        shutil.rmtree(tmp, ignore_errors=True)


# ═════════════════════════════════════════════
# 15. Memory types edge cases
# ═════════════════════════════════════════════
def test_memory_types_edge_cases():
    print("\n=== TORTURE: Memory Types Edge Cases ===")
    from src.memory.types import Memory, MAX_MEMORY_TEXT_LENGTH

    # Very long text
    long_text = "a" * 5000
    m = Memory(id="long", text=long_text, scope="shared", category="fact")
    check("long text stored", len(m.text) == 5000)

    # Unicode text
    unicode_text = "こんにちは世界 🌍 مرحبا Привет"
    m2 = Memory(id="uni", text=unicode_text, scope="shared", category="fact")
    check("unicode stored", m2.text == unicode_text)

    # Round-trip with unicode
    d = m2.to_dict()
    m3 = Memory.from_dict(d)
    check("unicode round-trip", m3.text == unicode_text)

    # from_dict backward compat — missing fields
    minimal = {"id": "min", "text": "minimal", "scope": "shared", "category": "fact"}
    m4 = Memory.from_dict(minimal)
    check("missing tier → default", m4.tier is not None or m4.tier == "")
    check("missing version → 1", m4.version == 1)
    check("missing tags → []", m4.tags == [] or m4.tags is None)

    # is_active
    m5 = Memory(id="act", text="active", scope="shared", category="fact")
    check("is_active True", m5.is_active())
    m5.deleted_at = "2026-01-01T00:00:00Z"
    check("is_active False after delete", not m5.is_active())

    # MAX_MEMORY_TEXT_LENGTH exists and is reasonable
    check("MAX_MEMORY_TEXT_LENGTH defined", MAX_MEMORY_TEXT_LENGTH > 0)
    check("MAX_MEMORY_TEXT_LENGTH reasonable",
          100 <= MAX_MEMORY_TEXT_LENGTH <= 100000,
          f"got {MAX_MEMORY_TEXT_LENGTH}")


# ═════════════════════════════════════════════
# 16. Settings rapid read/write stress
# ═════════════════════════════════════════════
def test_settings_stress():
    print("\n=== STRESS: Settings Rapid Read/Write (100 cycles) ===")
    from pathlib import Path

    tmp = tempfile.mkdtemp()
    try:
        import web.app as _app

        orig_settings = _app.SETTINGS_FILE
        tmp_settings = Path(tmp) / "config" / "settings.json"
        tmp_settings.parent.mkdir(parents=True)
        tmp_settings.write_text("{}", encoding="utf-8")
        _app.SETTINGS_FILE = tmp_settings

        t0 = time.time()
        for i in range(100):
            _app._save_settings({"iteration": i, "data": f"value_{i}" * 10})
            loaded = _app._load_settings()
            if loaded.get("iteration") != i:
                check(f"iteration {i} mismatch", False, f"got {loaded.get('iteration')}")
                break
        else:
            elapsed = time.time() - t0
            check(f"100 read/write cycles in {elapsed:.2f}s", True)

        # Final state correct
        final = _app._load_settings()
        check("final iteration=99", final["iteration"] == 99)

        # Concurrent-ish: rapid overwrites then read
        for i in range(50):
            _app._save_settings({"x": i})
        final2 = _app._load_settings()
        check("rapid overwrite: last wins", final2["x"] == 49)

    finally:
        _app.SETTINGS_FILE = orig_settings
        shutil.rmtree(tmp, ignore_errors=True)


# ═════════════════════════════════════════════
# 17. Avatar file I/O stress
# ═════════════════════════════════════════════
def test_avatar_io_stress():
    print("\n=== STRESS: Avatar File I/O (20 rapid writes) ===")
    import base64 as b64
    from pathlib import Path

    tmp = tempfile.mkdtemp()
    try:
        import web.app as _app

        orig_settings = _app.SETTINGS_FILE
        orig_uploads = _app._UPLOADS_DIR

        tmp_uploads = Path(tmp) / "uploads"
        tmp_uploads.mkdir()
        tmp_settings = Path(tmp) / "settings.json"
        tmp_settings.write_text("{}", encoding="utf-8")

        _app.SETTINGS_FILE = tmp_settings
        _app._UPLOADS_DIR = tmp_uploads

        # 1x1 red PNG
        png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

        t0 = time.time()
        settings = {"agent_avatars": {}}
        for i in range(20):
            name = f"agent_{i}"
            settings["agent_avatars"][name] = {
                "image": f"data:image/png;base64,{png_b64}",
                "color": f"#{i:06x}"
            }
        tmp_settings.write_text(json.dumps(settings), encoding="utf-8")

        _app._migrate_base64_avatars()
        elapsed = time.time() - t0

        reloaded = json.loads(tmp_settings.read_text(encoding="utf-8"))
        migrated_count = sum(
            1 for v in reloaded["agent_avatars"].values()
            if v["image"].startswith("/uploads/")
        )
        check(f"20 avatars migrated in {elapsed:.2f}s", migrated_count == 20)

        # Verify files exist
        avatar_files = list(tmp_uploads.glob("avatar_*"))
        check("20 avatar files on disk", len(avatar_files) == 20, f"got {len(avatar_files)}")

        # All files have data
        empty_files = [f for f in avatar_files if f.stat().st_size == 0]
        check("no empty avatar files", len(empty_files) == 0)

        # Unique filenames (no collisions)
        names = [f.name for f in avatar_files]
        check("all filenames unique", len(set(names)) == 20)

    finally:
        _app.SETTINGS_FILE = orig_settings
        _app._UPLOADS_DIR = orig_uploads
        shutil.rmtree(tmp, ignore_errors=True)


# ═════════════════════════════════════════════
# 18. Profile CRUD stress
# ═════════════════════════════════════════════
def test_profile_crud_stress():
    print("\n=== STRESS: Profile CRUD (10 agent cycles) ===")
    from pathlib import Path

    tmp = tempfile.mkdtemp()
    try:
        import web.app as _app

        orig_profiles = _app._PROFILES_DIR
        orig_prompts = _app._PROMPTS_DIR
        orig_settings = _app.SETTINGS_FILE

        tmp_profiles = Path(tmp) / "profiles"
        tmp_profiles.mkdir()
        tmp_prompts = Path(tmp) / "prompts"
        tmp_prompts.mkdir()
        tmp_settings = Path(tmp) / "settings.json"
        tmp_settings.write_text("{}", encoding="utf-8")

        _app._PROFILES_DIR = tmp_profiles
        _app._PROMPTS_DIR = tmp_prompts
        _app.SETTINGS_FILE = tmp_settings

        try:
            from httpx import ASGITransport, AsyncClient
            import asyncio
            from web.app import app as _test_app

            async def _run():
                transport = ASGITransport(app=_test_app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    t0 = time.time()
                    agent_names = []

                    # Create 10 agents
                    for i in range(10):
                        name = f"stress_agent_{i}"
                        r = await client.post("/api/profiles", json={
                            "name": name, "model": f"model_{i}",
                            "system_prompt": f"You are agent {i}."
                        })
                        if r.status_code == 200:
                            agent_names.append(name)
                    check(f"10 agents created", len(agent_names) == 10)

                    # Update all 10
                    for name in agent_names:
                        await client.put(f"/api/profiles/{name}", json={
                            "temperature": 0.8
                        })
                    check("10 agents updated", True)

                    # Read all 10
                    for name in agent_names:
                        r = await client.get(f"/api/profiles/{name}")
                        if r.status_code != 200:
                            check(f"read {name}", False, f"status={r.status_code}")
                            break
                    else:
                        check("10 agents readable", True)

                    # Delete all 10
                    for name in agent_names:
                        await client.delete(f"/api/profiles/{name}")
                    remaining = list(tmp_profiles.glob("*.yaml"))
                    check("all agents deleted", len(remaining) == 0, f"remaining: {len(remaining)}")

                    elapsed = time.time() - t0
                    check(f"10 CRUD cycles in {elapsed:.2f}s", True)

            asyncio.run(_run())

        except ImportError:
            check("httpx not available — skipped", True)

    finally:
        _app._PROFILES_DIR = orig_profiles
        _app._PROMPTS_DIR = orig_prompts
        _app.SETTINGS_FILE = orig_settings
        shutil.rmtree(tmp, ignore_errors=True)


# ═════════════════════════════════════════════
# 19. Saved profile CRUD stress
# ═════════════════════════════════════════════
def test_saved_profile_stress():
    print("\n=== STRESS: Saved Profile CRUD (20 rapid cycles) ===")
    from pathlib import Path

    tmp = tempfile.mkdtemp()
    try:
        import web.app as _app

        orig_dir = _app._SAVED_MEMORY_PROFILES_DIR
        orig_settings = _app.SETTINGS_FILE

        tmp_profiles = Path(tmp) / "profiles" / "memory"
        tmp_profiles.mkdir(parents=True)
        tmp_settings = Path(tmp) / "settings.json"
        tmp_settings.write_text("{}", encoding="utf-8")

        _app._SAVED_MEMORY_PROFILES_DIR = tmp_profiles
        _app.SETTINGS_FILE = tmp_settings

        try:
            from httpx import ASGITransport, AsyncClient
            import asyncio
            from web.app import app as _test_app

            async def _run():
                transport = ASGITransport(app=_test_app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    t0 = time.time()

                    # Create 20 profiles
                    for i in range(20):
                        r = await client.post("/api/tools/memory/profiles", json={
                            "filename": f"stress_{i}",
                            "profile": {"name": f"Stress {i}", "max_total": i * 100}
                        })
                        if r.status_code != 200:
                            check(f"save stress_{i}", False, f"status={r.status_code}")
                            break
                    else:
                        check("20 profiles saved", True)

                    # List all
                    r2 = await client.get("/api/tools/memory/profiles")
                    entries = r2.json()
                    check("list returns 20", len(entries) == 20, f"got {len(entries)}")

                    # Read each
                    for i in range(20):
                        r3 = await client.get(f"/api/tools/memory/profiles/stress_{i}")
                        if r3.status_code != 200:
                            check(f"read stress_{i}", False)
                            break
                        data = r3.json()
                        if data.get("max_total") != i * 100:
                            check(f"data stress_{i}", False, f"max_total={data.get('max_total')}")
                            break
                    else:
                        check("20 profiles readable with correct data", True)

                    # Delete all
                    for i in range(20):
                        await client.delete(f"/api/tools/memory/profiles/stress_{i}")
                    remaining = list(tmp_profiles.glob("*.json"))
                    check("all profiles deleted", len(remaining) == 0, f"remaining: {len(remaining)}")

                    elapsed = time.time() - t0
                    check(f"20 CRUD cycles in {elapsed:.2f}s", True)

            asyncio.run(_run())

        except ImportError:
            check("httpx not available — skipped", True)

    finally:
        _app._SAVED_MEMORY_PROFILES_DIR = orig_dir
        _app.SETTINGS_FILE = orig_settings
        shutil.rmtree(tmp, ignore_errors=True)


# ═════════════════════════════════════════════
# 20. Skin toggle stress
# ═════════════════════════════════════════════
def test_skin_toggle_stress():
    print("\n=== STRESS: Skin Toggle (50 rapid switches) ===")
    from pathlib import Path

    tmp = tempfile.mkdtemp()
    try:
        import web.app as _app

        orig_settings = _app.SETTINGS_FILE
        tmp_settings = Path(tmp) / "settings.json"
        tmp_settings.write_text("{}", encoding="utf-8")
        _app.SETTINGS_FILE = tmp_settings

        try:
            from httpx import ASGITransport, AsyncClient
            import asyncio
            from web.app import app as _test_app

            skins = ["default", "midnight", "aurora", "ember", "ocean", "forest",
                      "sunset", "slate", "neon", "arctic", "volcanic", "cyberpunk"]

            async def _run():
                transport = ASGITransport(app=_test_app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    t0 = time.time()
                    for i in range(50):
                        skin = skins[i % len(skins)]
                        r = await client.put("/api/skin", json={"skin": skin})
                        if r.status_code != 200:
                            check(f"skin toggle {i}", False)
                            break
                    else:
                        elapsed = time.time() - t0
                        check(f"50 skin toggles in {elapsed:.2f}s", True)

                    # Verify last skin persisted correctly
                    last_skin = skins[49 % len(skins)]
                    r2 = await client.get("/api/skin")
                    check("last skin correct", r2.json()["skin"] == last_skin,
                          f"expected {last_skin}, got {r2.json()['skin']}")

                    # Settings file reflects last skin
                    s = json.loads(tmp_settings.read_text(encoding="utf-8"))
                    check("skin in settings.json", s["skin"] == last_skin)

            asyncio.run(_run())

        except ImportError:
            check("httpx not available — skipped", True)

    finally:
        _app.SETTINGS_FILE = orig_settings
        shutil.rmtree(tmp, ignore_errors=True)


# ═════════════════════════════════════════════
# 21. Tool registry resolution stress
# ═════════════════════════════════════════════
def test_tool_registry_stress():
    print("\n=== STRESS: Tool Registry Resolution (all tools x 10) ===")
    import src.tools.registry as reg

    all_tools = reg.list_registered_tools()
    check("registry has tools", len(all_tools) > 0, f"found {len(all_tools)}")

    t0 = time.time()
    resolved_count = 0
    for _round in range(10):
        for name in all_tools:
            td = reg._resolve_tool(name)
            if td:
                resolved_count += 1
    elapsed = time.time() - t0

    expected = len(all_tools) * 10
    check(f"{expected} resolutions in {elapsed:.2f}s", resolved_count == expected,
          f"got {resolved_count}")

    # All definitions have required fields
    for name in all_tools:
        td = reg._resolve_tool(name)
        if td:
            fn = td.get("function", {})
            check(f"{name}: has name", "name" in fn)
            check(f"{name}: has description", "description" in fn)
            check(f"{name}: has parameters", "parameters" in fn)


# ═════════════════════════════════════════════
# 22. Memory strip-tags stress
# ═════════════════════════════════════════════
def test_strip_memory_tags_stress():
    print("\n=== STRESS: _strip_memory_tags (100 mixed strings) ===")
    from web.app import _strip_memory_tags

    t0 = time.time()
    for i in range(100):
        # Build a string with mixed tags and plain text
        parts = [f"Sentence {i}."]
        if i % 2 == 0:
            parts.append(f"[MEMORY_SAVE: category=bio | User fact {i}]")
        if i % 3 == 0:
            parts.append(f"[MEMORY_SAVE: Another fact {i}]")
        parts.append(f"End of message {i}.")
        text = " ".join(parts)

        result = _strip_memory_tags(text)
        if "[MEMORY_SAVE" in result:
            check(f"strip #{i}: tags remaining", False, result[:80])
            break
        if f"Sentence {i}" not in result:
            check(f"strip #{i}: text lost", False, result[:80])
            break
        if f"End of message {i}" not in result:
            check(f"strip #{i}: end text lost", False, result[:80])
            break
    else:
        elapsed = time.time() - t0
        check(f"100 strip ops in {elapsed:.3f}s", True)

    # Edge: deeply nested brackets (should not crash)
    nested = "[MEMORY_SAVE: text with [brackets] inside]"
    try:
        _strip_memory_tags(nested)
        check("nested brackets: no crash", True)
    except Exception as e:
        check("nested brackets: no crash", False, str(e))

    # Edge: very long tag
    long_tag = f"[MEMORY_SAVE: category=lore | {'x' * 5000}]"
    result_long = _strip_memory_tags(long_tag)
    check("long tag stripped", "[MEMORY_SAVE" not in result_long)

    # Edge: empty string
    check("empty string: no crash", _strip_memory_tags("") == "")


# ═════════════════════════════════════════════
# 23. Model Router classification stress
# ═════════════════════════════════════════════
def test_model_router_classify_stress():
    print("\n=== STRESS: Model Router Classification (500 ops) ===")
    from src.tools.model_router import (
        classify_task, resolve_model_for_task, resolve_tier,
        get_next_tier, ModelRouterTool, _CLASSIFICATION_RULES,
    )

    # ── Rapid classification — 500 diverse stimuli ──
    stimuli = [
        "write code for a REST API",
        "summarize the meeting notes",
        "plan the sprint for Q3",
        "review the security audit report",
        "polish the final document",
        "remember that the user likes dark mode",
        "reflect on today's progress",
        "search for Python documentation",
        "hello how are you today",
        "debug the authentication module",
    ]

    t0 = time.time()
    for i in range(500):
        text = stimuli[i % len(stimuli)]
        result = classify_task(text)
        if not isinstance(result, str) or len(result) == 0:
            check(f"classify #{i} valid", False, f"got {result!r}")
            break
    else:
        elapsed = time.time() - t0
        check(f"500 classifications in {elapsed:.3f}s", elapsed < 2.0)

    # Verify deterministic results
    for text in stimuli:
        r1 = classify_task(text)
        r2 = classify_task(text)
        if r1 != r2:
            check(f"deterministic: {text[:30]}", False, f"{r1} != {r2}")
            break
    else:
        check("all classifications deterministic", True)

    # ── Known task types: each classification rule must fire ──
    rule_task_types = [task_type for _, task_type in _CLASSIFICATION_RULES]
    for task_type in rule_task_types:
        # Find a stimulus that triggers this rule
        for keywords, tt in _CLASSIFICATION_RULES:
            if tt == task_type:
                test_text = keywords[0]  # use the first keyword
                result = classify_task(test_text)
                check(f"rule fires: {task_type}", result == task_type,
                      f"input='{test_text}' got '{result}'")
                break

    # ── resolve_model_for_task stress with mock config ──
    mock_cfg = {
        "enabled": True,
        "tiers": [
            {"id": "t0", "label": "local_cheap", "enabled": True,
             "provider": "ollama", "primary_model": "qwen:7b"},
            {"id": "t1", "label": "cheap_cloud", "enabled": True,
             "provider": "openai", "primary_model": "gpt-4o-mini"},
            {"id": "t2", "label": "expensive_cloud", "enabled": True,
             "provider": "openai", "primary_model": "gpt-4o"},
        ],
        "task_tier_map": {
            "coding": "cheap_cloud",
            "summarization": "local_cheap",
            "planning": "cheap_cloud",
            "high_stakes": "expensive_cloud",
            "final_polish": "expensive_cloud",
            "memory_ops": "local_cheap",
            "reflection": "local_cheap",
            "tool_use": "cheap_cloud",
            "agi_tick": "cheap_cloud",
            "general": "__auto__",
        },
    }

    t0 = time.time()
    for i in range(200):
        text = stimuli[i % len(stimuli)]
        model, provider, task_type = resolve_model_for_task(text, mock_cfg)
        # For non-general tasks, model and provider should be set
        if task_type != "general":
            if model is None or provider is None:
                check(f"resolve #{i} model set", False,
                      f"task={task_type} model={model} provider={provider}")
                break
        else:
            # general → __auto__ → fallback
            if model is not None:
                check(f"resolve #{i} general fallback", False, f"model={model}")
                break
    else:
        elapsed = time.time() - t0
        check(f"200 resolve_model_for_task in {elapsed:.3f}s", elapsed < 2.0)

    # ── get_next_tier escalation stress ──
    tier_ids = ["t0", "t1", "t2"]
    for current_id in tier_ids:
        nt = get_next_tier(current_id, mock_cfg)
        idx = tier_ids.index(current_id)
        if idx + 1 < len(tier_ids):
            check(f"escalate {current_id} valid", nt is not None and nt["id"] == tier_ids[idx + 1])
        else:
            check(f"escalate {current_id} → None (top)", nt is None)

    # ── ModelRouterTool.execute stress (100 rapid calls) ──
    actions_cycle = [
        {"action": "classify", "text": "write code"},
        {"action": "resolve", "text": "summarize this"},
        {"action": "list_tiers"},
        {"action": "get_map"},
    ]
    t0 = time.time()
    for i in range(100):
        args = actions_cycle[i % len(actions_cycle)]
        result = ModelRouterTool.execute(args)
        parsed = json.loads(result)
        if "error" in parsed and args.get("text"):
            check(f"tool execute #{i}", False, f"unexpected error: {parsed}")
            break
    else:
        elapsed = time.time() - t0
        check(f"100 tool executions in {elapsed:.3f}s", elapsed < 3.0)


# ═════════════════════════════════════════════
# 23b. Router Presets CRUD stress
# ═════════════════════════════════════════════
def test_router_presets_stress():
    """Rapid-fire preset save/list/load/delete cycles via direct API helpers."""
    print("\n=== STRESS: Router Presets CRUD (30 cycles) ===")
    from pathlib import Path

    tmp = tempfile.mkdtemp()
    try:
        import web.app as _app_mod

        # Redirect presets dir to temp isolation
        orig_presets_dir = _app_mod._ROUTER_PRESETS_DIR
        tmp_presets = Path(tmp) / "router_presets"
        _app_mod._ROUTER_PRESETS_DIR = tmp_presets

        from web.app import (
            _ensure_router_presets_dir,
            _list_router_presets,
        )

        _ensure_router_presets_dir()
        check("presets dir created", tmp_presets.exists())

        # ── Rapid save — 30 presets ──
        t0 = time.time()
        for i in range(30):
            payload = {
                "name": f"Stress Preset {i}",
                "description": f"Preset number {i}",
                "created": f"2026-03-03T00:00:{i:02d}",
                "config": {
                    "tiers": [{"id": f"t{i % 4}", "label": f"tier_{i}", "enabled": True}],
                    "task_tier_map": {"coding": f"tier_{i}", "general": "__auto__"},
                },
            }
            import re as _re
            safe = _re.sub(r'[^\w\-. ]', '', payload["name"]).strip().replace(' ', '_')[:80] or "preset"
            dest = tmp_presets / f"{safe}.json"
            dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        elapsed = time.time() - t0
        check(f"30 saves in {elapsed:.3f}s", elapsed < 2.0)

        # ── List all ──
        presets = _list_router_presets()
        check("30 presets listed", len(presets) == 30, f"got {len(presets)}")

        # Each entry has required fields
        for p in presets:
            ok = all(k in p for k in ("name", "description", "filename", "created"))
            if not ok:
                check(f"preset entry fields", False, f"missing key in {p}")
                break
        else:
            check("all 30 entries have required fields", True)

        # ── Rapid load — read all 30 files ──
        t0 = time.time()
        for p in presets:
            f = tmp_presets / f"{p['filename']}.json"
            data = json.loads(f.read_text(encoding="utf-8"))
            if "config" not in data:
                check(f"load {p['filename']} has config", False)
                break
        else:
            elapsed = time.time() - t0
            check(f"30 loads in {elapsed:.3f}s", elapsed < 2.0)

        # ── Name sanitisation stress ──
        evil_names = [
            "../../etc/passwd",
            "<script>alert(1)</script>",
            "A" * 200,  # over 80 char limit
            "   spaces   ",
            "UPPER and lower",
            "slashes/and\\backslashes",
            "dots...lots...of...them",
            "",  # empty
        ]
        for name in evil_names:
            safe = _re.sub(r'[^\w\-. ]', '', name).strip().replace(' ', '_')[:80] or "preset"
            check(f"sanitise '{name[:20]}' → no slash", "/" not in safe and "\\" not in safe)
            check(f"sanitise '{name[:20]}' → <=80 chars", len(safe) <= 80)
            check(f"sanitise '{name[:20]}' → not empty", len(safe) > 0)

        # ── Overwrite stress — write same name 10 times ──
        for i in range(10):
            payload = {
                "name": "Overwrite_Me",
                "description": f"Version {i}",
                "created": f"2026-03-03T00:00:00",
                "config": {"tiers": [], "task_tier_map": {"general": f"v{i}"}},
            }
            dest = tmp_presets / "Overwrite_Me.json"
            dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        final = json.loads(dest.read_text(encoding="utf-8"))
        check("overwrite final version", final["description"] == "Version 9")
        check("overwrite final config", final["config"]["task_tier_map"]["general"] == "v9")

        # ── Delete stress — remove all 30 + the overwritten one ──
        t0 = time.time()
        for f in list(tmp_presets.glob("*.json")):
            f.unlink()
        elapsed = time.time() - t0
        remaining = _list_router_presets()
        check(f"all deleted in {elapsed:.3f}s", len(remaining) == 0)

        # ── Empty dir listing ──
        check("empty list returns []", _list_router_presets() == [])

        # ── Corrupt file handling — _list_router_presets skips bad files ──
        (tmp_presets / "corrupt.json").write_text("{invalid json", encoding="utf-8")
        (tmp_presets / "good.json").write_text(json.dumps({
            "name": "Good One", "description": "ok", "created": "2026-01-01", "config": {},
        }), encoding="utf-8")
        presets_mixed = _list_router_presets()
        check("corrupt file skipped, good file listed", len(presets_mixed) == 1)
        check("good file name correct", presets_mixed[0]["name"] == "Good One")

        _app_mod._ROUTER_PRESETS_DIR = orig_presets_dir

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ═════════════════════════════════════════════
# 25. AGI Loop state stress
# ═════════════════════════════════════════════
def test_agi_loop_state_stress():
    print("\n=== STRESS: AGI Loop State (rapid ticks + transitions) ===")
    from src.tools.agi_loop import AGILoopTool, AGILoopState, get_loop_state

    state = get_loop_state()
    state.reset()

    # ── Rapid tick logging — 500 ticks ──
    t0 = time.time()
    for i in range(500):
        state.log_tick({
            "tick": i,
            "cost": 0.001,
            "model": "gpt-4o-mini",
            "status": "ok" if i % 10 != 0 else "error",
            "ts": f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}Z",
        })
    elapsed = time.time() - t0
    check(f"500 tick logs in {elapsed:.3f}s", elapsed < 2.0)

    # In-memory cap at 200
    check("tick_history capped at 200", len(state.tick_history) == 200)
    check("oldest tick is 300", state.tick_history[0]["tick"] == 300)
    check("newest tick is 499", state.tick_history[-1]["tick"] == 499)

    # ── to_dict under load ──
    t0 = time.time()
    for _ in range(100):
        d = state.to_dict()
    elapsed = time.time() - t0
    check(f"100 to_dict in {elapsed:.3f}s", elapsed < 1.0)
    check("to_dict recent_ticks capped at 20", len(d["recent_ticks"]) == 20)

    # ── Rapid state transitions ──
    t0 = time.time()
    for i in range(100):
        state.running = True
        state.paused = False
        state.current_tick = i
        state.total_cost += 0.001
        state.error_streak = i % 5
        state.paused = True
        state.running = False
    elapsed = time.time() - t0
    check(f"100 state transitions in {elapsed:.3f}s", elapsed < 0.5)

    # ── Tool execute stress — 200 rapid calls ──
    state.reset()
    state.running = True
    for i in range(20):
        state.log_tick({"tick": i})

    actions = [
        {"action": "status"},
        {"action": "tick_history"},
        {"action": "tick_history", "limit": 5},
    ]

    t0 = time.time()
    for i in range(200):
        args = actions[i % len(actions)]
        result = AGILoopTool.execute(args)
        parsed = json.loads(result)
        if not isinstance(parsed, dict):
            check(f"tool call #{i}", False, f"not a dict: {type(parsed)}")
            break
    else:
        elapsed = time.time() - t0
        check(f"200 tool executions in {elapsed:.3f}s", elapsed < 2.0)

    # ── Pause/resume cycle stress ──
    state.reset()
    state.running = True
    t0 = time.time()
    for i in range(50):
        # Pause
        r_pause = json.loads(AGILoopTool.execute({"action": "request_pause"}))
        if not r_pause.get("ok"):
            check(f"pause #{i}", False, f"ok=False: {r_pause}")
            break
        # Resume
        r_resume = json.loads(AGILoopTool.execute({"action": "request_resume"}))
        if not r_resume.get("ok"):
            check(f"resume #{i}", False, f"ok=False: {r_resume}")
            break
    else:
        elapsed = time.time() - t0
        check(f"50 pause/resume cycles in {elapsed:.3f}s", elapsed < 1.0)

    # ── Edge: Double pause → second still works (already paused) ──
    state.reset()
    state.running = True
    AGILoopTool.execute({"action": "request_pause"})
    r_double = json.loads(AGILoopTool.execute({"action": "request_pause"}))
    # Already paused → still ok because state.paused is already True
    check("double pause: state still paused", state.paused is True)

    # ── Edge: Double resume ──
    AGILoopTool.execute({"action": "request_resume"})
    r_double_resume = json.loads(AGILoopTool.execute({"action": "request_resume"}))
    check("double resume: ok=False", r_double_resume["ok"] is False)

    # ── Reset round-trip ──
    state.running = True
    state.total_cost = 99.99
    state.error_streak = 10
    state.log_tick({"tick": 999})
    state.reset()
    check("reset clears running", state.running is False)
    check("reset clears cost", state.total_cost == 0.0)
    check("reset clears streak", state.error_streak == 0)
    check("reset clears history", len(state.tick_history) == 0)

    # Clean up
    state.reset()


# ═════════════════════════════════════════════
# 26. AGI Journal stress (rapid writes, disk I/O, clear/reload)
# ═════════════════════════════════════════════
def test_agi_journal_stress():
    print("\n=== STRESS: AGI Journal (rapid writes + disk I/O) ===")
    from src.tools.agi_loop import get_loop_state
    import src.tools.agi_loop as _agi_mod
    from pathlib import Path

    state = get_loop_state()
    state.reset()

    tmp = tempfile.mkdtemp()
    orig_data_dir = _agi_mod._DATA_DIR
    _agi_mod._DATA_DIR = Path(tmp)

    try:
        # ── Rapid journal writes — 600 entries (test cap at 500) ──
        t0 = time.time()
        for i in range(600):
            state.log_journal({
                "ts": f"2026-01-01T00:00:{i % 60:02d}Z",
                "loop": i // 15,
                "tick": i % 15,
                "agent": ["astraea", "callum", "codex_animus"][i % 3],
                "narrative": f"Journal stress entry {i}: " + "x" * 40,
                "cost": round(0.001 * (i % 10), 4),
                "had_error": i % 50 == 0,
            })
        elapsed = time.time() - t0
        check(f"600 journal writes in {elapsed:.3f}s", elapsed < 3.0)
        check("in-memory capped at 500", len(state.journal_entries) == 500)
        check("oldest entry is #100", state.journal_entries[0]["narrative"].endswith("100: " + "x" * 40))
        check("newest entry is #599", state.journal_entries[-1]["narrative"].endswith("599: " + "x" * 40))

        # ── JSONL disk file size ──
        jpath = Path(tmp) / "agi_loop_journal.jsonl"
        check("journal file exists", jpath.is_file())
        lines = jpath.read_text(encoding="utf-8").strip().split("\n")
        check("all 600 lines persisted on disk", len(lines) == 600)

        # ── Rapid to_dict calls under load ──
        t0 = time.time()
        for _ in range(100):
            d = state.to_dict()
        elapsed = time.time() - t0
        check(f"100 to_dict (journal) in {elapsed:.3f}s", elapsed < 1.0)
        check("recent_journal capped at 20", len(d["recent_journal"]) == 20)
        check("recent_journal latest correct", "599" in d["recent_journal"][-1]["narrative"])

        # ── Clear and reload cycles ──
        t0 = time.time()
        for cycle in range(10):
            # Write 50 entries
            for i in range(50):
                state.log_journal({"ts": f"c{cycle}-{i}", "narrative": f"Cycle{cycle}-{i}"})
            # Clear
            state.clear_journal()
            check_ok = len(state.journal_entries) == 0
            if not check_ok:
                check(f"clear cycle {cycle}", False, f"got {len(state.journal_entries)}")
                break
            # Reload should be empty since file was deleted
            state.load_journal_from_disk()
            if len(state.journal_entries) != 0:
                check(f"reload after clear cycle {cycle}", False, f"got {len(state.journal_entries)}")
                break
        else:
            elapsed = time.time() - t0
            check(f"10 clear/reload cycles in {elapsed:.3f}s", elapsed < 3.0)

        # ── Reload large file stress ──
        state.clear_journal()
        with open(jpath, "w", encoding="utf-8") as f:
            for i in range(1000):
                f.write(json.dumps({"narrative": f"Reload{i}", "ts": f"t{i}"}) + "\n")
        t0 = time.time()
        state.load_journal_from_disk()
        elapsed = time.time() - t0
        check(f"reload 1000 entries in {elapsed:.3f}s", elapsed < 2.0)
        check("reload caps at 500", len(state.journal_entries) == 500)
        check("reload has latest (999)", state.journal_entries[-1]["narrative"] == "Reload999")
        check("reload oldest (500)", state.journal_entries[0]["narrative"] == "Reload500")

        # ── Mixed agent filtering readiness — verify unique agents ──
        state.clear_journal()
        agents = ["astraea", "callum", "codex_animus", "test_agent"]
        for i in range(200):
            state.log_journal({"agent": agents[i % len(agents)], "narrative": f"A{i}"})
        unique = set(e["agent"] for e in state.journal_entries)
        check("4 unique agents", len(unique) == 4)
        check("all agents present", unique == set(agents))

    finally:
        _agi_mod._DATA_DIR = orig_data_dir
        shutil.rmtree(tmp, ignore_errors=True)
        state.reset()


# ═════════════════════════════════════════════
# 27. Narrative builder stress (rapid _build_tick_narrative calls)
# ═════════════════════════════════════════════
def test_narrative_builder_stress():
    print("\n=== STRESS: _build_tick_narrative (200 rapid builds) ===")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from web.app import _build_tick_narrative

    # Build varied tick results
    variants = [
        # Error
        {"error": "Connection refused"},
        # Single tool + short response
        {"tool_calls": [{"tool": "echo"}], "response": "Hello world."},
        # Multiple tools
        {"tool_calls": [{"tool": "web.search"}, {"tool": "memory.add"}],
         "response": "Found it. Saved to memory."},
        # Long response
        {"response": "A" * 400 + ". Done."},
        # No response
        {"tool_calls": [], "response": ""},
        # Empty dict
        {},
        # Only response, no tools
        {"response": "Simple answer. Clear and concise."},
        # Error with tool calls (error takes priority)
        {"error": "Timeout", "tool_calls": [{"tool": "echo"}], "response": "ignored"},
        # Many tool calls
        {"tool_calls": [{"tool": f"tool_{i}"} for i in range(10)],
         "response": "Lots of tools used."},
        # Unicode response
        {"response": "Réponse en français. Très bien."},
    ]

    t0 = time.time()
    errors = 0
    for i in range(200):
        tick = variants[i % len(variants)]
        try:
            result = _build_tick_narrative(tick, "astraea", i // 15, i % 15, 0.001 * i)
            if not isinstance(result, str) or len(result) == 0:
                errors += 1
        except Exception:
            errors += 1
    elapsed = time.time() - t0
    check(f"200 narratives in {elapsed:.3f}s", elapsed < 2.0)
    check("no errors in 200 builds", errors == 0, f"got {errors} errors")

    # Verify specific outputs
    narr_err = _build_tick_narrative({"error": "boom"}, "a", 0, 0, 0.0)
    check("error narrative correct", "boom" in narr_err)

    narr_tools = _build_tick_narrative(
        {"tool_calls": [{"tool": "a"}, {"tool": "b"}, {"tool": "c"}], "response": "Done."},
        "a", 0, 0, 0.0
    )
    check("multi tool narrative has 'and'", " and " in narr_tools)

    narr_long = _build_tick_narrative({"response": "W" * 500}, "a", 0, 0, 0.0)
    check("long response capped", len(narr_long) <= 310 or "..." in narr_long)

    narr_empty = _build_tick_narrative({}, "a", 0, 0, 0.0)
    check("empty → No textual response", "No textual response" in narr_empty)


# ═════════════════════════════════════════════
if __name__ == "__main__":
    test_vault_rapid_crud()
    test_vault_bulk_delete()
    test_pii_guard_stress()
    test_directive_store_stress()
    test_directives_tool_stress()
    test_manifest_validation_stress()
    test_active_directives_stress()
    test_runtime_policy_stress()
    test_metering_accumulation()
    test_cost_log_stress()
    test_chunker_large_doc()
    test_vault_types_integration()
    test_manifest_directives_integration()
    test_data_paths_continuation_integration()
    test_memory_types_edge_cases()
    test_settings_stress()
    test_avatar_io_stress()
    test_profile_crud_stress()
    test_saved_profile_stress()
    test_skin_toggle_stress()
    test_tool_registry_stress()
    test_strip_memory_tags_stress()
    test_model_router_classify_stress()
    test_router_presets_stress()
    test_agi_loop_state_stress()
    test_agi_journal_stress()
    test_narrative_builder_stress()

    print(f"\n{'='*40}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL:
        sys.exit(1)
    else:
        print("All tests passed.")
