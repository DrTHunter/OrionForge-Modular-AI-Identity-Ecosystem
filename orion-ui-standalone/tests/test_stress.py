"""Stress / integration tests — exercising cross-module interactions.

Run from project root:
    python -m tests.test_stress

This is the "torture test" suite: rapid-fire operations, concurrent-ish
access patterns, boundary conditions, and cross-module integration.
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
if __name__ == "__main__":
    test_vault_rapid_crud()
    test_vault_bulk_delete()
    test_pii_guard_stress()
    test_directive_store_stress()
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

    print(f"\n{'='*40}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL:
        sys.exit(1)
    else:
        print("All tests passed.")
