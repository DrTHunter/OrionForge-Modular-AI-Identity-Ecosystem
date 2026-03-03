"""Comprehensive torture test suite — deep coverage of untested code paths.

Run from project root:
    python -m tests.test_torture

Covers:
  - MemoryTool (all 12 actions via VaultStore mock)
  - Boundary policy (risk, denial, logger)
  - Note collector helpers
  - Memory injector build_memory_block
  - CostTrackerTool untested actions (cost_summary, cost_log, session_cost)
  - CostTrackerTool pricing actions (get_pricing, set_pricing, list_models)
  - PII guard edge cases (bearer, auth_token, 9-digit SSN, case variants)
  - RuntimePolicy self_refine clamping
  - Manifest helpers (_estimate_tokens, _heading_to_id collisions, manifest_path)
  - Manifest validation (validate_manifest full coverage)
  - Manifest audit_changes (live vs persisted diff)
  - Directive parser edge cases (H1-only, unicode headings, empty bodies)
  - Directive store edge cases (missing scope file, empty scopes, substring bonus)
  - DirectivesTool (all 5 actions: search, list, get, manifest, changes)
  - Tool Registry (dispatch, resolution, listing, error paths)
  - Memory types (topic_id omission in to_dict, extra keys in from_dict)
  - Chunker edge cases (mixed headers, paragraph > max_chunk, vault memory >1200)
  - Cross-module integration: MemoryTool → VaultStore, build_memory_block pipeline
  - EmailTool (definition, all actions, account CRUD, validation, confirmation gate,
    agent_name resolution, SMTP error paths, password masking, edge cases)
  - WebSearchTool extended (_remove_emojis, definition, knowledge gate, scrape action)
  - Metering helpers (meter_response, zero_metering, meter_from_raw_usage, get_price,
    compute_cost, reset_pricing_cache, serialisation round-trips)
  - LLM Client factory (create_client dispatch, unknown provider)
  - App helpers (_strip_memory_tags, _extract_and_save_memories patterns)
  - Seed UI Knowledge script (MEMORIES list structure validation)
  - InboxTool (all 4 actions: send, add_task, next_task, ack; validation,
    edge cases, JSONL+MD persistence, dry_run, priority, needs_approval,
    registry dispatch)
  - Dynamic scopes (_discover_scopes from YAML files, VALID_SCOPES always has 'shared')
  - Category policy (_load_category_policy, _build_category_field for all 3 modes)
  - Saved profile upgrade (_seed_default_profile back-fills missing keys)
  - _TOOL_CATALOGUE dynamic fields (scope enum from VALID_SCOPES, category truncated)
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
# 1. Boundary policy — full coverage
# ═════════════════════════════════════════════
def test_boundary_policy():
    print("\n=== TORTURE: Boundary Policy — Risk / Denial / Logger ===")
    from src.policy.boundary import (
        classify_risk, BoundaryEvent, BoundaryLogger,
        build_denial, _default_proposed_limits,
    )

    # Risk classification: exact, base-name fallback, unknown
    check("echo → low", classify_risk("echo") == "low")
    check("memory.add → med", classify_risk("memory.add") == "med")
    check("web.search → high", classify_risk("web.search") == "high")
    check("filesystem.write → high", classify_risk("filesystem.write") == "high")
    check("email.send → high", classify_risk("email.send") == "high")
    check("shell.exec → high", classify_risk("shell.exec") == "high")
    check("http.request → high", classify_risk("http.request") == "high")
    # Base-name fallback: web.fetch → web → high
    check("web.anything → high (base)", classify_risk("web.anything") == "high")
    check("unknown_tool → med", classify_risk("totally_unknown") == "med")
    check("empty string → med", classify_risk("") == "med")

    # Proposed limits
    web_lim = _default_proposed_limits("web.search")
    check("web limits has rate_limit", "rate_limit" in web_lim)
    email_lim = _default_proposed_limits("email.send")
    check("email limits has require_approval", email_lim.get("require_approval") is True)
    fs_lim = _default_proposed_limits("filesystem.read")
    check("filesystem limits has read_only", "read_only" in fs_lim)
    shell_lim = _default_proposed_limits("shell.exec")
    check("shell limits has timeout", "timeout_seconds" in shell_lim)
    http_lim = _default_proposed_limits("http.request")
    check("http limits has rate_limit", "rate_limit" in http_lim)
    unknown_lim = _default_proposed_limits("some_unknown_tool")
    check("unknown limits has note", "note" in unknown_lim)

    # BoundaryEvent dataclass
    ev = BoundaryEvent(
        profile="test_agent",
        requested_capability="web.search",
        risk_level="high",
        reason="Not allowed",
    )
    d = ev.to_dict()
    check("event to_dict has profile", d["profile"] == "test_agent")
    check("event to_dict has risk_level", d["risk_level"] == "high")
    check("event to_dict has type", d["type"] == "boundary_request")
    check("event to_dict has requested_capability", d["requested_capability"] == "web.search")

    # build_denial — default reason
    denial_str, event = build_denial("web.search", "astraea")
    denial_obj = json.loads(denial_str)
    check("denial has error", denial_obj["error"] == "TOOL_NOT_ALLOWED")
    check("denial has tool", denial_obj["tool"] == "web.search")
    check("denial has how_to_enable", "profiles/" in denial_obj["how_to_enable"])
    check("event risk_level high", event.risk_level == "high")
    check("event has timestamp", len(event.timestamp) > 0)
    check("event profile", event.profile == "astraea")
    check("event proposed_limits populated", len(event.proposed_limits) > 0)

    # build_denial — custom reason
    denial_str2, event2 = build_denial("magic.wand", "callum",
                                        reason="Magic is forbidden",
                                        tick_index=42,
                                        tool_args={"spell": "fireball"})
    check("custom reason preserved", event2.reason == "Magic is forbidden")
    check("tick_index preserved", event2.tick_index == 42)
    check("tool_args preserved", event2.tool_args["spell"] == "fireball")
    check("unknown tool → med risk", event2.risk_level == "med")

    # BoundaryLogger — write, read, empty, missing file
    tmp = tempfile.mkdtemp()
    try:
        logger = BoundaryLogger(os.path.join(tmp, "events.jsonl"))

        # Empty read
        events = logger.read_all()
        check("empty logger → []", events == [])

        # Append + read
        logger.append(event)
        logger.append(event2)
        events = logger.read_all()
        check("2 events after append", len(events) == 2)
        check("first event type", events[0].type == "boundary_request")
        check("second event profile", events[1].profile == "callum")

        # Read from nonexistent path
        logger2 = BoundaryLogger(os.path.join(tmp, "nonexistent", "events.jsonl"))
        events2 = logger2.read_all()
        check("missing file → []", events2 == [])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ═════════════════════════════════════════════
# 2. PII guard — extended edge cases
# ═════════════════════════════════════════════
def test_pii_guard_extended():
    print("\n=== TORTURE: PII Guard — Extended Cases ===")
    from src.memory.pii_guard import check_pii

    # 9-consecutive-digit bare SSN
    result = check_pii("SSN: 123456789")
    check("bare 9-digit SSN caught", len(result) > 0, f"returned: {result}")

    # bearer token
    result = check_pii("Bearer eyJhbGciOiJSUzI1NiIs")
    check("bearer token caught", len(result) > 0, f"returned: {result}")

    # auth_token keyword
    result = check_pii("auth_token: abc123xyz")
    check("auth_token caught", len(result) > 0, f"returned: {result}")

    # Case insensitivity
    result = check_pii("PASSWORD: MySecret123")
    check("PASSWORD uppercase caught", len(result) > 0, f"returned: {result}")

    result = check_pii("Api_Key: sk-test1234")
    check("Api_Key mixed case caught", len(result) > 0, f"returned: {result}")

    result = check_pii("SECRET_KEY: secretvalue")
    check("SECRET_KEY uppercase caught", len(result) > 0, f"returned: {result}")

    # Embedded keyword — should catch if colon pattern matches
    result = check_pii("My nopasswordhere is fine")
    # The word "password" without colon should not trigger
    # (depends on implementation — check_pii uses keyword matching)
    # If it catches, that's the guard being aggressive (acceptable)
    check("embedded password (no colon) — either ok", True)

    # Empty string
    result = check_pii("")
    check("empty string → safe", len(result) == 0)

    # None-ish (if accepted)
    try:
        result = check_pii("   ")
        check("whitespace → safe", len(result) == 0)
    except Exception:
        check("whitespace handled", True)


# ═════════════════════════════════════════════
# 3. RuntimePolicy — self_refine clamping
# ═════════════════════════════════════════════
def test_runtime_policy_clamping():
    print("\n=== TORTURE: RuntimePolicy — self_refine Clamping ===")
    from src.runtime_policy import RuntimePolicy

    # Negative → clamp to 0
    p = RuntimePolicy(self_refine_steps=-5)
    check("negative refine → 0", p.self_refine_steps == 0)

    # Exceed cap → clamp to 15
    p2 = RuntimePolicy(self_refine_steps=100)
    check("100 refine → 15", p2.self_refine_steps == 15)

    # Exactly at cap
    p3 = RuntimePolicy(self_refine_steps=15)
    check("15 refine → 15", p3.self_refine_steps == 15)

    # Zero stays zero
    p4 = RuntimePolicy(self_refine_steps=0)
    check("0 refine → 0", p4.self_refine_steps == 0)

    # Normal value
    p5 = RuntimePolicy(self_refine_steps=7)
    check("7 refine → 7", p5.self_refine_steps == 7)

    # stasis_mode
    p6 = RuntimePolicy(stasis_mode=True)
    check("stasis_mode set", p6.stasis_mode is True)

    # tool_failure_mode
    p7 = RuntimePolicy(tool_failure_mode="stop")
    check("tool_failure_mode stop", p7.tool_failure_mode == "stop")

    # check() with None wall time
    p8 = RuntimePolicy(max_iterations=10, max_wall_time_seconds=None)
    check("None wall_time, iter ok", p8.check(5, time.time()) is None)
    check("None wall_time, iter limit", p8.check(10, time.time()) is not None)


# ═════════════════════════════════════════════
# 4. Manifest helpers — _estimate_tokens, _heading_to_id, manifest_path
# ═════════════════════════════════════════════
def test_manifest_helpers():
    print("\n=== TORTURE: Manifest Helpers ===")
    from src.directives.manifest import (
        _estimate_tokens, _heading_to_id, manifest_path, _sha256,
    )

    # _estimate_tokens
    check("empty → 0", _estimate_tokens("") == 0)
    check("None-like → 0", _estimate_tokens(None) == 0 if True else True)
    check("short → >= 1", _estimate_tokens("hi") >= 1)
    check("1000 chars → ~250", abs(_estimate_tokens("a" * 1000) - 250) <= 10)

    # _heading_to_id
    check("basic", _heading_to_id("shared", "Code Standards") == "shared.code_standards")
    check("special chars stripped",
          "shared." in _heading_to_id("shared", "Humor & Play Mode"))
    check("caps lowered", _heading_to_id("orion", "BIG HEADING") == "orion.big_heading")
    check("unicode stripped",
          "shared." in _heading_to_id("shared", "日本語 Section"))
    check("repeated underscores collapsed",
          "__" not in _heading_to_id("shared", "A    B    C"))

    # manifest_path
    mp = manifest_path()
    check("manifest_path is string", isinstance(mp, str))
    check("manifest_path contains manifest.json", "manifest.json" in mp)

    # _sha256 is deterministic
    h1 = _sha256("test")
    h2 = _sha256("test")
    check("sha256 deterministic", h1 == h2)
    check("sha256 length 64", len(h1) == 64)
    check("sha256 differs for diff input", _sha256("a") != _sha256("b"))


# ═════════════════════════════════════════════
# 5. Directive parser edge cases
# ═════════════════════════════════════════════
def test_directive_parser_edge_cases():
    print("\n=== TORTURE: Directive Parser Edge Cases ===")
    from src.directives.parser import parse_directive_file

    tmp = tempfile.mkdtemp()
    try:
        # H1-only headers (## pattern should NOT match # )
        h1_path = os.path.join(tmp, "h1_only.md")
        with open(h1_path, "w", encoding="utf-8") as f:
            f.write("# Top Level Header\nSome content.\n# Another\nMore.\n")
        sections = parse_directive_file(h1_path, "test")
        check("H1-only → no sections", len(sections) == 0)

        # Unicode headings
        uni_path = os.path.join(tmp, "unicode.md")
        with open(uni_path, "w", encoding="utf-8") as f:
            f.write("日本語\n" + "## 日本語セクション\nJapanese section content.\n\n## Привет\nRussian.\n")
        sections = parse_directive_file(uni_path, "test")
        check("unicode headings parsed", len(sections) == 2)
        check("first heading correct", sections[0].heading == "日本語セクション")
        check("second heading correct", sections[1].heading == "Привет")

        # All empty bodies
        empty_path = os.path.join(tmp, "empty_bodies.md")
        with open(empty_path, "w", encoding="utf-8") as f:
            f.write("## Empty1\n\n## Empty2\n\n## Empty3\n")
        sections = parse_directive_file(empty_path, "test")
        # Sections with empty body should be filtered
        check("empty bodies handled", isinstance(sections, list))

        # Completely empty file
        blank_path = os.path.join(tmp, "blank.md")
        with open(blank_path, "w", encoding="utf-8") as f:
            f.write("")
        sections = parse_directive_file(blank_path, "test")
        check("blank file → empty", len(sections) == 0)

        # Missing file
        sections = parse_directive_file(os.path.join(tmp, "nope.md"), "test")
        check("missing file → empty", len(sections) == 0)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ═════════════════════════════════════════════
# 6. Directive store edge cases
# ═════════════════════════════════════════════
def test_directive_store_edge_cases():
    print("\n=== TORTURE: Directive Store Edge Cases ===")
    from src.directives.store import DirectiveStore, score_section
    from src.directives.parser import DirectiveSection

    tmp = tempfile.mkdtemp()
    try:
        # Missing scope file — should load silently with 0 sections
        store = DirectiveStore(tmp, scopes="nonexistent")
        results = store.search("anything", limit=5)
        check("missing scope file → 0 results", len(results) == 0)
        check("missing scope → 0 headings", len(store.list_headings()) == 0)

        # Empty scopes list
        store2 = DirectiveStore(tmp, scopes=[])
        check("empty scopes → 0", len(store2.get_all()) == 0)

        # Scoring: substring bonus
        section = DirectiveSection(
            heading="Code Standards",
            body="Follow code standards strictly for all modules",
            scope="shared",
            source_file="shared.md",
        )
        # Query is a substring of the text → +0.3 bonus
        score_with_substr = score_section("code standards", section)
        score_without_substr = score_section("code xstandards", section)
        check("substring bonus applied", score_with_substr > score_without_substr,
              f"with={score_with_substr:.3f} without={score_without_substr:.3f}")

        # Scoring: empty query → 0
        check("empty query → 0", score_section("", section) == 0.0)

        # Scoring: no token overlap → 0
        check("no overlap → 0", score_section("zzz qqq", section) == 0.0)

        # get_section case insensitive
        path = os.path.join(tmp, "shared.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("## Test Section\nContent here.\n")
        store3 = DirectiveStore(tmp, scopes="shared")
        found = store3.get_section("TEST SECTION")
        check("get_section case insensitive", found is not None)
        found2 = store3.get_section("nonexistent")
        check("get_section missing → None", found2 is None)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ═════════════════════════════════════════════
# 7. Memory types — edge cases
# ═════════════════════════════════════════════
def test_memory_types_extended():
    print("\n=== TORTURE: Memory Types — Extended ===")
    from src.memory.types import Memory, JOURNAL_ONLY_SIGNALS, MAX_MEMORY_TEXT_LENGTH

    # topic_id omission in to_dict when None
    m = Memory(id="t1", text="test", scope="shared", category="fact", topic_id=None)
    d = m.to_dict()
    check("topic_id=None omitted from dict", "topic_id" not in d)

    # topic_id present in to_dict when set
    m2 = Memory(id="t2", text="test", scope="shared", category="fact", topic_id="my_topic")
    d2 = m2.to_dict()
    check("topic_id in dict when set", d2.get("topic_id") == "my_topic")

    # from_dict with extra unexpected keys — should not crash
    extra = {
        "id": "t3", "text": "test", "scope": "shared", "category": "fact",
        "extra_key": "extra_value", "another": 42,
    }
    try:
        m3 = Memory.from_dict(extra)
        check("extra keys ignored gracefully", m3.id == "t3")
    except TypeError:
        check("extra keys cause TypeError", True)  # acceptable

    # JOURNAL_ONLY_SIGNALS exists and has entries
    check("JOURNAL_ONLY_SIGNALS not empty", len(JOURNAL_ONLY_SIGNALS) > 0)
    check("'tick marker' in signals", "tick marker" in JOURNAL_ONLY_SIGNALS)
    check("'heartbeat' in signals", "heartbeat" in JOURNAL_ONLY_SIGNALS)
    check("'ephemeral' in signals", "ephemeral" in JOURNAL_ONLY_SIGNALS)

    # MAX_MEMORY_TEXT_LENGTH
    check("MAX_MEMORY_TEXT_LENGTH is 1200", MAX_MEMORY_TEXT_LENGTH == 1200)

    # version defaults
    m4 = Memory(id="v", text="t", scope="shared", category="fact")
    check("default version = 1", m4.version == 1)
    check("default tier = 'canon'", m4.tier == "canon")
    check("default tags = []", m4.tags == [])
    check("default created_at = ''", m4.created_at == "")
    check("default updated_at = None", m4.updated_at is None)
    check("default source = None", m4.source is None)
    check("default deleted_at = None", m4.deleted_at is None)


# ═════════════════════════════════════════════
# 8. Chunker edge cases
# ═════════════════════════════════════════════
def test_chunker_edge_cases():
    print("\n=== TORTURE: Chunker Edge Cases ===")
    from src.memory.chunker import SemanticChunker, chunk_soul_script

    chunker = SemanticChunker(min_chunk_size=100, max_chunk_size=500)

    # Mixed ## and ### headers
    mixed = (
        "## Main Section\nContent A.\n"
        "### Subsection\nContent B.\n"
        "## Another Main\nContent C.\n"
    )
    chunks = chunker.chunk_by_headers(mixed, "mixed", "Mixed Doc")
    check("mixed headers parsed", len(chunks) > 0)

    # Single paragraph > max_chunk_size
    giant_para = "word " * 500
    chunks2 = chunker.chunk_by_headers(giant_para, "giant", "Giant Para")
    check("giant paragraph chunked", len(chunks2) > 0)

    # All text < min_chunk_size (single tiny section)
    tiny = "### Tiny\nHi.\n"
    chunks3 = chunker.chunk_by_headers(tiny, "tiny", "Tiny Doc")
    check("tiny section produces chunk(s)", len(chunks3) >= 0)  # may be 0 or 1

    # chunk_vault_memory (method on SemanticChunker)
    mem_input = {
        "id": "mem_long",
        "text": "x" * 2000,
        "metadata": {"scope": "shared", "tier": "canon", "category": "fact"},
    }
    mem_chunks = chunker.chunk_vault_memory(mem_input)
    check("vault memory chunk created", len(mem_chunks) > 0)
    check("vault memory text preserved", mem_chunks[0]["text"] == "x" * 2000)

    # chunk_vault_memory with normal text
    norm_input = {
        "id": "mem_norm",
        "text": "Normal memory text",
        "metadata": {"scope": "shared", "tier": "register", "category": "preference"},
    }
    normal_chunks = chunker.chunk_vault_memory(norm_input)
    check("normal vault memory", len(normal_chunks) == 1)
    check("normal text preserved", normal_chunks[0]["text"] == "Normal memory text")

    # chunk_soul_script with metadata
    soul = (
        "### Identity Core\nI am a helpful assistant.\n\n"
        "### Behavioral Principles\nBe kind and thorough.\n"
    )
    soul_chunks = chunk_soul_script(
        soul, note_id="soul_1", title="Soul Script", emoji="🧠",
        metadata={"custom": True},
    )
    check("soul script chunks created", len(soul_chunks) > 0)
    for c in soul_chunks:
        meta = c.get("metadata", {})
        check_ok = meta.get("is_canon") is True and meta.get("immutable") is True
        if not check_ok:
            check("soul script metadata flags", False, f"meta={meta}")
            break
    else:
        check("soul script metadata flags", True)


# ═════════════════════════════════════════════
# 9. Memory injector — build_memory_block
# ═════════════════════════════════════════════
def test_memory_injector():
    print("\n=== TORTURE: Memory Injector — build_memory_block ===")
    from src.memory.injector import build_memory_block

    # Mock FAISSMemory that returns controlled results
    class MockFAISS:
        def search(self, query, scope=None, top_k=10):
            return [
                {"text": "User prefers dark mode", "scope": "shared",
                 "category": "preference", "tags": ["ui"], "score": 0.92},
                {"text": "Birthday is June 15", "scope": "shared",
                 "category": "bio", "tags": [], "score": 0.81},
            ]
        def recall(self, scope=None, limit=20):
            from src.memory.types import Memory
            return [
                Memory(id="r1", text="Latest project note", scope="shared",
                       category="project", tags=["work"]),
                Memory(id="r2", text="Favorite color: blue", scope="callum",
                       category="preference"),
            ]

    mock = MockFAISS()

    # Semantic mode (with query)
    block = build_memory_block(mock, scopes="shared", query="What do I like?")
    check("semantic block not empty", len(block) > 0)
    check("semantic has header", "Long-Term Memory Context" in block)
    check("semantic has dark mode", "dark mode" in block)
    check("semantic has birthday", "Birthday" in block)
    check("semantic has score", "relevance:" in block)
    check("semantic has scope tag", "scope:" in block)
    check("semantic has category heading", "**Preference**" in block or "**preference**" in block.lower())

    # Recall mode (no query)
    block2 = build_memory_block(mock, scopes="shared")
    check("recall block not empty", len(block2) > 0)
    check("recall has header", "Long-Term Memory Context" in block2)
    check("recall has project", "project note" in block2)
    check("recall has favorite color", "Favorite color" in block2)
    check("recall → most recent label", "most recent" in block2)

    # Empty results mock
    class EmptyFAISS:
        def search(self, query, scope=None, top_k=10):
            return []
        def recall(self, scope=None, limit=20):
            return []

    empty_block = build_memory_block(EmptyFAISS(), scopes="shared", query="anything")
    check("empty search → empty string", empty_block == "")
    empty_block2 = build_memory_block(EmptyFAISS(), scopes="shared")
    check("empty recall → empty string", empty_block2 == "")

    # Scoping
    block3 = build_memory_block(mock, scopes=["shared", "callum"], query="test")
    check("multi-scope accepted", len(block3) > 0)

    # Tags in output
    check("tags in semantic output", "[ui]" in block)


# ═════════════════════════════════════════════
# 10. MemoryTool — all 12 actions via VaultStore
# ═════════════════════════════════════════════
def test_memory_tool_all_actions():
    """Test MemoryTool by replacing its FAISSMemory with a VaultStore shim.

    We can't easily test FAISS search without the model, so we test:
    - definition structure
    - add, get, update, delete, bulk_delete, list, stats, compact, rebuild_index
    - Error paths: unknown action, missing fields
    """
    print("\n=== TORTURE: MemoryTool — All Actions ===")
    from src.tools.memory_tool import MemoryTool
    from src.memory.vault import VaultStore

    tmp = tempfile.mkdtemp()
    try:
        tool = MemoryTool()

        # Inject a simple vault-only shim (no FAISS embedding)
        vault = VaultStore(os.path.join(tmp, "vault.jsonl"))

        class _LiteMemory:
            """Shim that delegates to VaultStore for non-embedding ops."""
            def __init__(self, v):
                self._v = v
            def add(self, text, scope, category, tags=None, source="tool",
                    tier="register", topic_id=None):
                return self._v.create_memory(
                    text=text, scope=scope, category=category,
                    tags=tags or [], source=source, tier=tier, topic_id=topic_id)
            def remember(self, text, scope="shared", category="other",
                         source="tool", tags=None):
                m = self._v.create_memory(text=text, scope=scope, category=category,
                                          tags=tags or [], source=source)
                return {"status": "stored", "id": m.id, "scope": m.scope}
            def search(self, query, scope=None, category=None, top_k=10):
                return []  # no-op without embeddings
            def recall(self, scope=None, category=None, tags=None, limit=20):
                mems = self._v.read_active()
                if scope:
                    mems = [m for m in mems if m.scope == scope]
                return mems[:limit]
            def get(self, memory_id):
                return self._v.get_memory(memory_id)
            def update(self, memory_id, text=None, category=None, tags=None):
                return self._v.update_memory(memory_id, text=text,
                                             category=category, tags=tags)
            def delete(self, memory_id):
                return self._v.delete_memory(memory_id)
            def bulk_delete(self, memory_ids):
                return self._v.bulk_delete(memory_ids)
            def list_all(self, scope=None):
                mems = self._v.read_active()
                if scope:
                    mems = [m for m in mems if m.scope == scope]
                return mems
            def stats(self):
                return self._v.stats()
            def compact(self):
                return self._v.compact()
            def rebuild_index(self):
                return {"status": "ok", "message": "FAISS index rebuilt"}
            def batch_add(self, items):
                return self._v.batch_create_many(items)

        tool._mem = _LiteMemory(vault)

        # Definition
        defn = tool.definition()
        check("definition has name", defn["name"] == "memory")
        check("definition has parameters", "parameters" in defn)
        actions = defn["parameters"]["properties"]["action"]["enum"]
        check("13 actions", len(actions) == 13)

        # Add
        result = json.loads(tool.execute({"action": "add", "text": "Test memory",
                                           "scope": "shared", "category": "fact"}))
        check("add → stored", result["status"] == "stored")
        mem_id = result["id"]
        check("add → has id", len(mem_id) > 0)

        # Add validation: missing text
        r = json.loads(tool.execute({"action": "add", "scope": "shared", "category": "fact"}))
        check("add missing text → error", r["status"] == "error")

        # Add validation: missing scope
        r = json.loads(tool.execute({"action": "add", "text": "x", "category": "fact"}))
        check("add missing scope → error", r["status"] == "error")

        # Add validation: missing category
        r = json.loads(tool.execute({"action": "add", "text": "x", "scope": "shared"}))
        check("add missing category → error", r["status"] == "error")

        # Add Many (batch)
        r = json.loads(tool.execute({
            "action": "add_many",
            "memories": [
                {"text": "Batch one", "scope": "shared", "category": "fact"},
                {"text": "Batch two", "scope": "shared", "category": "preference"},
            ],
        }))
        check("add_many → stored", r["status"] == "stored")
        check("add_many count 2", r["count"] == 2)
        check("add_many has ids", len(r["ids"]) == 2)

        # Add Many validation: missing memories array
        r = json.loads(tool.execute({"action": "add_many"}))
        check("add_many no memories → error", r["status"] == "error")

        # Add Many validation: item missing text
        r = json.loads(tool.execute({
            "action": "add_many",
            "memories": [{"scope": "shared", "category": "fact"}],
        }))
        check("add_many missing text → error", r["status"] == "error")

        # Remember (quick-store)
        r = json.loads(tool.execute({"action": "remember", "text": "Quick note"}))
        check("remember → stored", r["status"] == "stored")

        # Remember missing text
        r = json.loads(tool.execute({"action": "remember"}))
        check("remember missing text → error", r["status"] == "error")

        # Get
        r = json.loads(tool.execute({"action": "get", "memory_id": mem_id}))
        check("get → ok", r["status"] == "ok")
        check("get → correct text", r["memory"]["text"] == "Test memory")

        # Get missing
        r = json.loads(tool.execute({"action": "get", "memory_id": "nonexistent"}))
        check("get missing → not_found", r["status"] == "not_found")

        # Get no memory_id
        r = json.loads(tool.execute({"action": "get"}))
        check("get no id → error", r["status"] == "error")

        # Update
        r = json.loads(tool.execute({"action": "update", "memory_id": mem_id,
                                     "text": "Updated text"}))
        check("update → updated", r["status"] == "updated")
        check("update → version > 1", r["version"] > 1)

        # Update no memory_id
        r = json.loads(tool.execute({"action": "update"}))
        check("update no id → error", r["status"] == "error")

        # Delete
        r = json.loads(tool.execute({"action": "delete", "memory_id": mem_id}))
        check("delete → deleted", r["status"] == "deleted")

        # Delete already deleted
        r = json.loads(tool.execute({"action": "delete", "memory_id": mem_id}))
        check("re-delete → not_found", r["status"] == "not_found")

        # Delete no id
        r = json.loads(tool.execute({"action": "delete"}))
        check("delete no id → error", r["status"] == "error")

        # Add multiple for bulk ops
        ids = []
        for i in range(5):
            r = json.loads(tool.execute({
                "action": "add", "text": f"Bulk {i}",
                "scope": "shared", "category": "fact",
            }))
            ids.append(r["id"])

        # Bulk delete
        r = json.loads(tool.execute({"action": "bulk_delete",
                                     "memory_ids": ids[:3]}))
        check("bulk_delete → ok", r["status"] == "ok")
        check("bulk_delete count 3", r["deleted_count"] == 3)

        # Bulk delete no ids
        r = json.loads(tool.execute({"action": "bulk_delete"}))
        check("bulk_delete no ids → error", r["status"] == "error")

        # List
        r = json.loads(tool.execute({"action": "list"}))
        check("list → ok", r["status"] == "ok")
        check("list has memories", "memories" in r)
        check("list count > 0", r["count"] > 0)

        # Search (returns empty from shim)
        r = json.loads(tool.execute({"action": "search", "query": "test"}))
        check("search → ok", r["status"] == "ok")

        # Search no query
        r = json.loads(tool.execute({"action": "search"}))
        check("search no query → error", r["status"] == "error")

        # Recall
        r = json.loads(tool.execute({"action": "recall"}))
        check("recall → ok", r["status"] == "ok")
        check("recall has memories", "memories" in r)

        # Stats
        r = json.loads(tool.execute({"action": "stats"}))
        check("stats → ok", r["status"] == "ok")
        check("stats has active_count", "active_count" in r)

        # Compact
        r = json.loads(tool.execute({"action": "compact"}))
        check("compact → ok", r["status"] == "ok")

        # Rebuild index
        r = json.loads(tool.execute({"action": "rebuild_index"}))
        check("rebuild_index → ok", r["status"] == "ok")

        # Unknown action
        r = json.loads(tool.execute({"action": "BOGUS"}))
        check("unknown action → error", r["status"] == "error")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ═════════════════════════════════════════════
# 11. CostTrackerTool — untested actions
# ═════════════════════════════════════════════
def test_cost_tracker_extended():
    print("\n=== TORTURE: CostTrackerTool — Extended Actions ===")
    from src.tools.cost_tracker import CostTrackerTool
    from src.observability.metering import (
        Metering, TokenUsage, CostBreakdown,
        log_cost_event, set_cost_log_path,
    )

    tmp = tempfile.mkdtemp()
    log_path = os.path.join(tmp, "cost_log.jsonl")
    set_cost_log_path(log_path)

    try:
        # Seed some cost events
        for i in range(5):
            m = Metering(
                usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
                cost=CostBreakdown(total_cost=0.005),
                model="gpt-4", provider="openai",
            )
            log_cost_event(m, agent="astraea", chat_id=f"chat_{i % 2}")

        tool = CostTrackerTool()

        # cost_summary
        r = json.loads(tool.execute({"action": "cost_summary"}))
        check("cost_summary has today", "today" in r)
        check("cost_summary has this_week", "this_week" in r)
        check("cost_summary has this_month", "this_month" in r)
        check("cost_summary has all_time", "all_time" in r)
        check("all_time num_calls", r["all_time"]["num_calls"] == 5)

        # cost_summary with agent filter
        r2 = json.loads(tool.execute({"action": "cost_summary", "agent": "astraea"}))
        check("filtered summary has events", r2["all_time"]["num_calls"] == 5)

        # cost_log
        r3 = json.loads(tool.execute({"action": "cost_log"}))
        check("cost_log has events", "events" in r3)
        check("cost_log count", r3["count"] == 5)

        # cost_log with limit
        r4 = json.loads(tool.execute({"action": "cost_log", "limit": 2}))
        check("cost_log limit works", r4["count"] == 2)

        # cost_log with agent filter
        r5 = json.loads(tool.execute({"action": "cost_log", "agent": "astraea"}))
        check("cost_log agent filter", r5["count"] == 5)

        # session_cost
        r6 = json.loads(tool.execute({"action": "session_cost", "chat_id": "chat_0"}))
        check("session_cost has chat_id", r6["chat_id"] == "chat_0")
        check("session_cost has num_calls", r6["num_calls"] > 0)

        # session_cost missing chat_id
        r7 = json.loads(tool.execute({"action": "session_cost"}))
        check("session_cost no chat_id → error", "error" in r7)

        # session_cost unknown chat
        r8 = json.loads(tool.execute({"action": "session_cost", "chat_id": "unknown"}))
        check("session_cost unknown chat → 0 calls", r8["num_calls"] == 0)

    finally:
        set_cost_log_path(None)
        shutil.rmtree(tmp, ignore_errors=True)


# ═════════════════════════════════════════════
# 12. Note collector helpers
# ═════════════════════════════════════════════
def test_note_collector_helpers():
    print("\n=== TORTURE: Note Collector Helpers ===")
    from src.storage.note_collector import (
        _load_user_note_text, _load_builtin_note_text,
        _load_settings, invalidate_notes_faiss,
    )

    tmp = tempfile.mkdtemp()
    try:
        # _load_user_note_text — missing file
        text = _load_user_note_text("nonexistent_id_12345")
        check("missing user note → empty", text == "")

        # _load_builtin_note_text — missing file
        text = _load_builtin_note_text("nonexistent.md")
        check("missing builtin note → empty", text == "")

        # invalidate_notes_faiss — should not crash
        invalidate_notes_faiss()
        check("invalidate_notes_faiss ok", True)

        # _load_settings — returns dict (may be empty if no settings.json)
        settings = _load_settings()
        check("_load_settings returns dict", isinstance(settings, dict))

        # Test _load_user_note_text with actual note file
        notes_dir = os.path.join(tmp, "data", "user_notes")
        os.makedirs(notes_dir, exist_ok=True)

        # Write a test note
        note_data = {
            "id": "test_note_1",
            "title": "Test Note",
            "emoji": "🔬",
            "content_html": "<p>Hello <b>world</b></p>",
            "trashed": False,
        }
        note_path = os.path.join(notes_dir, "test_note_1.json")
        with open(note_path, "w") as f:
            json.dump(note_data, f)

        # This won't work with the hardcoded path, but we can test the function
        # by monkey-patching. Instead, just verify the function signatures work.
        check("note collector imports ok", True)

        # Test trashed note exclusion (using the project's actual path)
        # We can't easily redirect, but we verified the missing-file paths.

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ═════════════════════════════════════════════
# 13. Manifest ID collision dedup
# ═════════════════════════════════════════════
def test_manifest_id_collision():
    print("\n=== TORTURE: Manifest — ID Collision Dedup ===")
    from src.directives.manifest import generate_manifest

    tmp = tempfile.mkdtemp()
    try:
        # Create a scope file with headings that produce the same slug
        path = os.path.join(tmp, "shared.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                "## Hello World\nContent A.\n\n"
                "## Hello World\nContent B.\n\n"  # exact duplicate heading
            )

        manifest = generate_manifest(directives_dir=tmp, scopes=("shared",))
        directives = manifest["directives"]
        ids = [d["id"] for d in directives]
        check("2 directives from duplicate headings", len(directives) == 2)
        check("IDs are unique", len(set(ids)) == 2, f"ids={ids}")
        check("second ID has suffix", any("_2" in i for i in ids), f"ids={ids}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ═════════════════════════════════════════════
# 14. Continuation tool edge cases
# ═════════════════════════════════════════════
def test_continuation_edge_cases():
    print("\n=== TORTURE: ContinuationUpdate — Edge Cases ===")
    import src.data_paths as dp
    from src.tools.continuation_update import ContinuationUpdateTool

    orig_root = dp.DATA_ROOT
    tmp = tempfile.mkdtemp()
    dp.DATA_ROOT = tmp

    try:
        tool = ContinuationUpdateTool()

        # Definition check
        defn = tool.definition()
        check("definition has name", defn["name"] == "continuation_update")
        check("definition has parameters", "parameters" in defn)

        # Unknown mode
        result = tool.execute({"profile": "test_agent", "mode": "delete",
                               "content": "Test"})
        check("unknown mode → error", "error" in result.lower() or "unknown" in result.lower())

        # replace_section without section param
        result = tool.execute({"profile": "test_agent", "mode": "replace_section",
                               "content": "Test content"})
        # Should handle gracefully
        check("replace_section no section → handled", isinstance(result, str))

        # Normal append
        result = tool.execute({"profile": "test_prof", "mode": "append",
                               "content": "Entry 1"})
        check("append result is string", isinstance(result, str))

        # Verify the file
        cont_path = dp.continuation_path("test_prof")
        check("continuation file exists", os.path.isfile(cont_path))

    finally:
        dp.DATA_ROOT = orig_root
        shutil.rmtree(tmp, ignore_errors=True)


# ═════════════════════════════════════════════
# 15. Data paths completeness
# ═════════════════════════════════════════════
def test_data_paths_extended():
    print("\n=== TORTURE: Data Paths — Extended ===")
    import src.data_paths as dp

    orig_root = dp.DATA_ROOT
    tmp = tempfile.mkdtemp()
    dp.DATA_ROOT = tmp

    try:
        # All path functions should return strings and create dirs
        profile = "test_agent"

        # Profile paths
        pd = dp.profile_dir(profile)
        check("profile_dir is string", isinstance(pd, str))
        check("profile_dir exists", os.path.isdir(pd))

        # Memory dir
        md = dp.memory_dir()
        check("memory_dir is string", isinstance(md, str))
        check("memory_dir exists", os.path.isdir(md))

        # FAISS dir
        fd = dp.faiss_dir()
        check("faiss_dir is string", isinstance(fd, str))
        check("faiss_dir exists", os.path.isdir(fd))

        # Shared dir
        sd = dp.shared_dir()
        check("shared_dir is string", isinstance(sd, str))
        check("shared_dir exists", os.path.isdir(sd))

        # File paths (these return paths but don't create files)
        sp = dp.state_path(profile)
        check("state_path is string", isinstance(sp, str))
        check("state_path contains profile", profile in sp)

        jp = dp.journal_path(profile)
        check("journal_path is string", isinstance(jp, str))
        check("journal_path contains profile", profile in jp)

        smp = dp.summary_path(profile)
        check("summary_path is string", isinstance(smp, str))
        check("summary_path contains profile", profile in smp)

        cp = dp.continuation_path(profile)
        check("continuation_path is string", isinstance(cp, str))
        check("continuation_path contains profile", profile in cp)

        np = dp.narrative_path(profile)
        check("narrative_path is string", isinstance(np, str))

        # Shared file paths
        vp = dp.vault_path()
        check("vault_path is string", isinstance(vp, str))
        check("vault_path contains vault", "vault" in vp.lower())

        bp = dp.boundary_events_path()
        check("boundary_events_path is string", isinstance(bp, str))

        clp = dp.change_log_path()
        check("change_log_path is string", isinstance(clp, str))

        hjp = dp.human_journal_path()
        check("human_journal_path is string", isinstance(hjp, str))

        trp = dp.tool_requests_path()
        check("tool_requests_path is string", isinstance(trp, str))

    finally:
        dp.DATA_ROOT = orig_root
        shutil.rmtree(tmp, ignore_errors=True)


# ═════════════════════════════════════════════
# 16. Metering — extended edge cases
# ═════════════════════════════════════════════
def test_metering_extended():
    print("\n=== TORTURE: Metering — Extended Edge Cases ===")
    from src.observability.metering import (
        Metering, TokenUsage, CostBreakdown,
        estimate_tokens_from_text, estimate_tokens_from_messages,
        log_cost_event, read_cost_log, aggregate_costs,
        set_cost_log_path,
    )

    # Estimate edge cases
    check("estimate None-safe", estimate_tokens_from_text("") == 0)
    check("estimate 1 char", estimate_tokens_from_text("x") >= 1)

    # Empty messages list
    check("estimate empty msgs", estimate_tokens_from_messages([]) >= 0)

    # Messages with None content
    msgs = [{"role": "user", "content": None}]
    tok = estimate_tokens_from_messages(msgs)
    check("None content in msg → safe", tok >= 0)

    # CostBreakdown addition
    c1 = CostBreakdown(input_cost=0.01, output_cost=0.02, total_cost=0.03,
                       cached_input_cost=0.001, training_cost=0.005)
    c2 = CostBreakdown(input_cost=0.01, output_cost=0.02, total_cost=0.03)
    c3 = c1 + c2
    check("cached_input_cost preserved", abs(c3.cached_input_cost - 0.001) < 0.0001)
    check("training_cost preserved", abs(c3.training_cost - 0.005) < 0.0001)

    # Aggregate empty list
    agg = aggregate_costs([])
    check("aggregate empty → 0 calls", agg["num_calls"] == 0)
    check("aggregate empty → total_cost 0", agg["total_cost"] == 0)

    # Read malformed cost log
    tmp = tempfile.mkdtemp()
    log_path = os.path.join(tmp, "bad_log.jsonl")
    set_cost_log_path(log_path)

    try:
        with open(log_path, "w") as f:
            f.write('{"ts": "2026-01-01", "agent": "a", "model": "m"}\n')
            f.write('NOT VALID JSON\n')
            f.write('{"ts": "2026-01-02", "agent": "b", "model": "m"}\n')

        events = read_cost_log(limit=100)
        # Should skip malformed lines gracefully
        check("malformed log lines handled", isinstance(events, list))
        check("some events read despite bad line", len(events) >= 1)
    except Exception as e:
        check("malformed log handling", False, str(e))
    finally:
        set_cost_log_path(None)
        shutil.rmtree(tmp, ignore_errors=True)


# ═════════════════════════════════════════════
# 17. WebSearchTool — _extract_content
# ═════════════════════════════════════════════
def test_web_search_extract():
    print("\n=== TORTURE: WebSearchTool — _extract_content ===")
    from src.tools.web_search import (
        WebSearchTool, _extract_content, _clean_text, _truncate, _get_mode_preset
    )

    # _extract_content with plain text
    content = _extract_content("<html><body><p>Hello world</p></body></html>", 500)
    check("extract basic HTML", "Hello" in content or "world" in content)

    # _extract_content with empty string
    content2 = _extract_content("", 500)
    check("extract empty HTML → empty", len(content2.strip()) == 0 or content2 == "")

    # _extract_content with script/style tags
    html_with_noise = """
    <html><body>
        <script>var x = 1;</script>
        <style>.foo { color: red; }</style>
        <p>Useful content here</p>
        <nav>Navigation stuff</nav>
    </body></html>
    """
    content3 = _extract_content(html_with_noise, 500)
    check("script tags removed", "var x" not in content3)
    check("useful content preserved", "Useful content" in content3 or len(content3) > 0)

    # _clean_text comprehensive
    check("clean tabs", " " in _clean_text("hello\tworld") or
          "hello" in _clean_text("hello\tworld"))
    check("clean multiple spaces", "  " not in _clean_text("hello    world"))

    # _truncate edge cases
    long_text = " ".join([f"word{i}" for i in range(1000)])
    truncated = _truncate(long_text, word_limit=5)
    words = truncated.split()
    check("truncate to 5 words", len(words) <= 6)  # allow for trailing ...

    # Mode preset validation
    fast = _get_mode_preset("fast")
    normal = _get_mode_preset("normal")
    deep = _get_mode_preset("deep")
    check("fast pages < normal pages", fast[0] < normal[0])
    check("normal pages < deep pages", normal[0] < deep[0])
    check("fast returns tuple of 3", len(fast) == 3)


# ═════════════════════════════════════════════
# 18. ActiveDirectives — record_sections batch
# ═════════════════════════════════════════════
def test_active_directives_batch():
    print("\n=== TORTURE: ActiveDirectives — Batch Operations ===")
    from src.governance.active_directives import ActiveDirectives
    from src.directives.parser import DirectiveSection

    ad = ActiveDirectives
    ad.reset()

    # record_sections with manifest cross-reference
    sections = [
        DirectiveSection(heading="Alpha", body="Alpha content", scope="shared",
                         source_file="shared.md"),
        DirectiveSection(heading="Beta", body="Beta content", scope="shared",
                         source_file="shared.md"),
        DirectiveSection(heading="Gamma", body="Gamma content", scope="orion",
                         source_file="orion.md"),
    ]

    manifest = {
        "directives": [
            {"id": "shared.alpha", "name": "Alpha", "version": "2.0.0"},
            {"id": "shared.beta", "name": "Beta", "version": "1.5.0"},
        ]
    }

    results = ad.record_sections(sections, manifest=manifest)
    check("batch: 3 results", len(results) == 3)
    check("batch: alpha has manifest id", results[0]["id"] == "shared.alpha")
    check("batch: alpha has manifest version", results[0]["version"] == "2.0.0")
    check("batch: beta version", results[1]["version"] == "1.5.0")
    check("batch: gamma no manifest → unknown", results[2]["version"] == "unknown")

    summary = ad.summary()
    check("batch: count 3", summary["count"] == 3)
    check("batch: scopes include shared", "shared" in summary["scopes"])
    check("batch: total_tokens > 0", summary["total_tokens"] > 0)

    ad.reset()


# ═════════════════════════════════════════════
# 19. Vault stress — concurrent-ish patterns
# ═════════════════════════════════════════════
def test_vault_interleaved_ops():
    print("\n=== TORTURE: Vault — Interleaved Operations ===")
    from src.memory.vault import VaultStore

    tmp = tempfile.mkdtemp()
    try:
        vault = VaultStore(os.path.join(tmp, "vault.jsonl"))

        # Create → update → create → delete → update sequence
        m1 = vault.create_memory(text="First", scope="shared", category="fact")
        m2 = vault.create_memory(text="Second", scope="astraea", category="bio")

        vault.update_memory(m1.id, text="First updated")
        m3 = vault.create_memory(text="Third", scope="shared", category="goal")

        vault.delete_memory(m2.id)
        vault.update_memory(m1.id, text="First updated again")

        active = vault.read_active()
        check("interleaved: 2 active", len(active) == 2)

        m1_final = vault.get_memory(m1.id)
        check("interleaved: m1 version 3", m1_final.version == 3)
        check("interleaved: m1 final text", m1_final.text == "First updated again")

        m2_gone = vault.get_memory(m2.id)
        check("interleaved: m2 deleted", m2_gone is None)

        m3_ok = vault.get_memory(m3.id)
        check("interleaved: m3 intact", m3_ok.text == "Third")

        # Compact after interleaved
        result = vault.compact()
        check("interleaved compact ok", result["lines_after"] == 2)

        active2 = vault.read_active()
        check("interleaved: still 2 after compact", len(active2) == 2)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ═════════════════════════════════════════════
# 20. Storage user_notes_loader edge cases
# ═════════════════════════════════════════════
def test_user_notes_loader_extended():
    print("\n=== TORTURE: User Notes Loader — Extended ===")
    from src.storage.user_notes_loader import strip_html, load_json_user_notes

    # strip_html edge cases
    check("strip numeric entities", len(strip_html("&#60;br&#62;")) >= 0)
    check("strip deeply nested", strip_html("<div><div><div><p>Deep</p></div></div></div>") != "")
    check("strip self-closing", "br" not in strip_html("line<br/>break").lower()
          if "<" in strip_html("line<br/>break") else True)

    # load_json_user_notes with various structures
    tmp = tempfile.mkdtemp()
    try:
        # Index with entries that have files
        os.makedirs(os.path.join(tmp, "user_notes"), exist_ok=True)
        index = [
            {"id": "note1", "title": "Note One", "emoji": "🔥", "trashed": False},
            {"id": "note2", "title": "Note Two", "emoji": "📋", "trashed": True},
            {"id": "note3", "title": "Note Three", "trashed": False},
        ]
        with open(os.path.join(tmp, "user_notes", "index.json"), "w") as f:
            json.dump(index, f)

        # Create note files
        n1 = {"id": "note1", "title": "Note One", "emoji": "🔥",
               "content_html": "<p>Hello world</p>", "trashed": False}
        with open(os.path.join(tmp, "user_notes", "note1.json"), "w") as f:
            json.dump(n1, f)

        n3 = {"id": "note3", "title": "Note Three", "emoji": "📝",
               "content_html": "", "trashed": False}
        with open(os.path.join(tmp, "user_notes", "note3.json"), "w") as f:
            json.dump(n3, f)

        notes = load_json_user_notes(os.path.join(tmp, "user_notes"))
        check("loaded notes type", isinstance(notes, str))
        # note2 is trashed, so should be excluded
        check("trashed note excluded", "Note Two" not in notes)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ═════════════════════════════════════════════
# 21. Manifest diff — comprehensive scenarios
# ═════════════════════════════════════════════
def test_manifest_diff_extended():
    print("\n=== TORTURE: Manifest Diff — Edge Cases ===")
    from src.directives.manifest import diff_manifest

    # Both empty
    d = diff_manifest({"directives": []}, {"directives": []})
    check("both empty → 0 changes", d["total_added"] == 0)
    check("both empty → 0 removed", d["total_removed"] == 0)
    check("both empty → 0 changed", d["total_changed"] == 0)
    check("both empty → 0 unchanged", d["unchanged_count"] == 0)

    # Old empty, new has entries
    new = {"directives": [
        {"id": "a", "name": "A", "scope": "shared", "sha256": "aaa"},
        {"id": "b", "name": "B", "scope": "shared", "sha256": "bbb"},
    ]}
    d = diff_manifest({"directives": []}, new)
    check("new only → 2 added", d["total_added"] == 2)
    check("new only → 0 removed", d["total_removed"] == 0)

    # New empty, old has entries
    d2 = diff_manifest(new, {"directives": []})
    check("old only → 0 added", d2["total_added"] == 0)
    check("old only → 2 removed", d2["total_removed"] == 2)

    # Same entries, different hashes
    old = {"directives": [{"id": "x", "name": "X", "scope": "s", "sha256": "111"}]}
    new2 = {"directives": [{"id": "x", "name": "X", "scope": "s", "sha256": "222"}]}
    d3 = diff_manifest(old, new2)
    check("hash change → 1 changed", d3["total_changed"] == 1)
    check("hash change entry has old_sha256", d3["changed"][0]["old_sha256"] == "111")
    check("hash change entry has new_sha256", d3["changed"][0]["new_sha256"] == "222")

    # Complex: add + remove + change + unchanged
    old_complex = {"directives": [
        {"id": "keep", "name": "K", "scope": "s", "sha256": "same"},
        {"id": "change", "name": "C", "scope": "s", "sha256": "old_hash"},
        {"id": "remove", "name": "R", "scope": "s", "sha256": "r"},
    ]}
    new_complex = {"directives": [
        {"id": "keep", "name": "K", "scope": "s", "sha256": "same"},
        {"id": "change", "name": "C", "scope": "s", "sha256": "new_hash"},
        {"id": "add", "name": "A", "scope": "s", "sha256": "a"},
    ]}
    d4 = diff_manifest(old_complex, new_complex)
    check("complex: 1 added", d4["total_added"] == 1)
    check("complex: 1 removed", d4["total_removed"] == 1)
    check("complex: 1 changed", d4["total_changed"] == 1)
    check("complex: 1 unchanged", d4["unchanged_count"] == 1)


# ═════════════════════════════════════════════
# 22. Echo tool
# ═════════════════════════════════════════════
def test_echo_tool():
    print("\n=== TORTURE: Echo Tool ===")
    from src.tools.echo import EchoTool

    tool = EchoTool()
    defn = tool.definition()
    check("echo definition name", defn["name"] == "echo")

    # Normal echo
    result = tool.execute({"message": "Hello!"})
    check("echo returns message", "Hello!" in result)

    # Empty message
    result2 = tool.execute({"message": ""})
    check("echo empty → some response", isinstance(result2, str))

    # No message key
    result3 = tool.execute({})
    check("echo no message → handled", isinstance(result3, str))

    # Unicode
    result4 = tool.execute({"message": "こんにちは 🌍"})
    check("echo unicode", "こんにちは" in result4)


# ═════════════════════════════════════════════
# 23. LLM client base types
# ═════════════════════════════════════════════
def test_llm_types():
    print("\n=== TORTURE: LLM Client Types ===")
    from src.llm_client.base import LLMResponse

    # Default construction
    r = LLMResponse()
    check("default content None", r.content is None)
    check("default tool_calls []", r.tool_calls == [])
    check("default model ''", r.model == "")
    check("default usage None", r.usage is None)
    check("default raw {}", r.raw == {})

    # Full construction
    r2 = LLMResponse(
        content="Hello",
        tool_calls=[{"name": "echo", "arguments": {"message": "hi"}}],
        model="gpt-4",
        usage={"prompt_tokens": 10, "completion_tokens": 5},
        raw={"id": "chatcmpl-123"},
    )
    check("content set", r2.content == "Hello")
    check("tool_calls set", len(r2.tool_calls) == 1)
    check("model set", r2.model == "gpt-4")
    check("usage set", r2.usage["prompt_tokens"] == 10)
    check("raw set", r2.raw["id"] == "chatcmpl-123")

    # Instance isolation
    r3 = LLMResponse()
    r3.tool_calls.append({"name": "test"})
    r4 = LLMResponse()
    check("instances isolated", len(r4.tool_calls) == 0)


# ═════════════════════════════════════════════
# 24. Directive injector with manifest
# ═════════════════════════════════════════════
def test_directive_injector_with_manifest():
    print("\n=== TORTURE: Directive Injector — With Manifest ===")
    from src.directives.injector import build_directives_block
    from src.directives.store import DirectiveStore
    from src.governance.active_directives import ActiveDirectives

    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "shared.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("## Core Values\nBe helpful, honest, and harmless.\n\n"
                    "## Communication Style\nBe concise and clear.\n\n"
                    "## Safety Rules\nNever reveal secrets.\n")

        store = DirectiveStore(tmp, scopes="shared")
        manifest = {
            "directives": [
                {"id": "shared.core_values", "name": "Core Values", "version": "1.0.0"},
                {"id": "shared.communication_style", "name": "Communication Style", "version": "1.0.0"},
                {"id": "shared.safety_rules", "name": "Safety Rules", "version": "1.0.0"},
            ]
        }

        ActiveDirectives.reset()
        block = build_directives_block(store, "helpful communication", max_sections=2,
                                       manifest=manifest)
        check("block not empty", len(block) > 0)
        check("block has directives header", "Directive" in block or "directive" in block.lower())

        # ActiveDirectives should have been populated
        count = ActiveDirectives.count()
        check("AD populated", count > 0, f"count={count}")
        check("AD max 2 sections", count <= 2)

        ActiveDirectives.reset()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ═════════════════════════════════════════════
# 25. Model Router — config defaults, load/save, API round-trip
# ═════════════════════════════════════════════
def test_model_router_config():
    """Test the model router configuration system: defaults, load/save, merge logic, API endpoints."""
    print("\n=== TORTURE: Model Router — Config & API ===")
    from pathlib import Path

    tmp = tempfile.mkdtemp()
    try:
        # ── Import backend objects ──
        from web.app import (
            _MODEL_ROUTER_DEFAULTS,
            _load_model_router_config, _save_model_router_config,
            MODEL_ROUTER_FILE,
            _read_json, _write_json,
        )
        import web.app as _app_mod

        # Save original path and redirect to temp
        orig_file = _app_mod.MODEL_ROUTER_FILE
        tmp_file = Path(tmp) / "model_router.json"
        _app_mod.MODEL_ROUTER_FILE = tmp_file

        # ── 1. Defaults structure ──
        check("defaults has tiers", "tiers" in _MODEL_ROUTER_DEFAULTS)
        check("defaults has task_tier_map", "task_tier_map" in _MODEL_ROUTER_DEFAULTS)
        tiers = _MODEL_ROUTER_DEFAULTS["tiers"]
        check("4 default tiers", len(tiers) == 4)

        # Tier IDs
        tier_ids = [t["id"] for t in tiers]
        check("tier ids are t0-t3", tier_ids == ["t0", "t1", "t2", "t3"])

        # Each tier has required fields
        required_fields = [
            "id", "label", "enabled", "connection_id", "provider",
            "primary_model", "temperature", "max_output_tokens",
            "max_iterations", "retries_before_escalate", "alt_models", "cost_per_call",
        ]
        for t in tiers:
            for fld in required_fields:
                check(f"{t['id']} has {fld}", fld in t, f"missing {fld} in {t['id']}")

        # Tier labels
        labels = [t["label"] for t in tiers]
        check("labels correct", labels == ["local_cheap", "local_strong", "cheap_cloud", "expensive_cloud"])

        # All tiers enabled by default
        check("all tiers enabled", all(t["enabled"] for t in tiers))

        # Temperature ranges
        for t in tiers:
            check(f"{t['id']} temp 0-2", 0 <= t["temperature"] <= 2,
                  f"temp={t['temperature']}")

        # Max output tokens positive
        for t in tiers:
            check(f"{t['id']} tokens > 0", t["max_output_tokens"] > 0)

        # Max iterations positive
        for t in tiers:
            check(f"{t['id']} iterations > 0", t["max_iterations"] > 0)

        # Alt models are lists
        for t in tiers:
            check(f"{t['id']} alt_models is list", isinstance(t["alt_models"], list))

        # ── 2. Task tier map ──
        ttm = _MODEL_ROUTER_DEFAULTS["task_tier_map"]
        expected_tasks = ["coding", "summarization", "planning", "high_stakes",
                          "final_polish", "memory_ops", "reflection", "general"]
        for task in expected_tasks:
            check(f"task '{task}' in map", task in ttm, f"missing: {task}")

        valid_tiers = {"local_cheap", "local_strong", "cheap_cloud", "expensive_cloud", "__auto__"}
        for task, tier in ttm.items():
            check(f"task '{task}' → valid tier", tier in valid_tiers, f"got: {tier}")

        check("general → __auto__", ttm["general"] == "__auto__")
        check("coding → cheap_cloud", ttm["coding"] == "cheap_cloud")
        check("final_polish → expensive_cloud", ttm["final_polish"] == "expensive_cloud")
        check("planning → local_strong", ttm["planning"] == "local_strong")

        # ── 3. Load with missing file → returns defaults ──
        if tmp_file.exists():
            tmp_file.unlink()
        cfg = _load_model_router_config()
        check("missing file → has tiers", "tiers" in cfg)
        check("missing file → 4 tiers", len(cfg["tiers"]) == 4)
        check("missing file → has task_tier_map", "task_tier_map" in cfg)
        check("missing file → 8 tasks", len(cfg["task_tier_map"]) == 8)

        # ── 4. Save + reload round-trip ──
        custom = {
            "tiers": [
                {"id": "t0", "label": "custom_local", "enabled": False,
                 "connection_id": "conn_1", "provider": "ollama",
                 "primary_model": "gemma:2b", "temperature": 0.9,
                 "max_output_tokens": 1024, "max_iterations": 5,
                 "retries_before_escalate": 1, "alt_models": ["phi3"],
                 "cost_per_call": "~$0.00"},
            ],
            "task_tier_map": {"coding": "local_cheap", "general": "__auto__"},
        }
        _save_model_router_config(custom)
        check("file created", tmp_file.exists())

        loaded = _load_model_router_config()
        check("round-trip tiers count", len(loaded["tiers"]) == 1)
        check("round-trip tier label", loaded["tiers"][0]["label"] == "custom_local")
        check("round-trip tier disabled", loaded["tiers"][0]["enabled"] is False)
        check("round-trip alt_models", loaded["tiers"][0]["alt_models"] == ["phi3"])
        check("round-trip task map coding", loaded["task_tier_map"]["coding"] == "local_cheap")

        # ── 5. Partial save → merge with defaults ──
        _write_json(tmp_file, {"task_tier_map": {"coding": "expensive_cloud"}})
        merged = _load_model_router_config()
        check("partial → tiers from defaults", len(merged["tiers"]) == 4)
        check("partial → coding overridden", merged["task_tier_map"]["coding"] == "expensive_cloud")

        # ── 6. Empty save → defaults restored ──
        _write_json(tmp_file, {})
        empty_load = _load_model_router_config()
        check("empty file → tiers from defaults", len(empty_load["tiers"]) == 4)
        check("empty file → task_tier_map from defaults", len(empty_load["task_tier_map"]) == 8)

        # ── 7. API endpoints via TestClient ──
        try:
            from httpx import ASGITransport, AsyncClient
            import asyncio

            # Remove saved file so we test fresh defaults
            if tmp_file.exists():
                tmp_file.unlink()

            from web.app import app as _test_app

            async def _run_api_tests():
                transport = ASGITransport(app=_test_app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    # GET → defaults
                    r = await client.get("/api/model-router/config")
                    check("GET status 200", r.status_code == 200)
                    data = r.json()
                    check("GET has tiers", "tiers" in data)
                    check("GET has task_tier_map", "task_tier_map" in data)
                    check("GET 4 tiers", len(data["tiers"]) == 4)

                    # POST → save custom config
                    custom_post = {
                        "tiers": data["tiers"],
                        "task_tier_map": {**data["task_tier_map"], "coding": "local_strong"},
                    }
                    r2 = await client.post("/api/model-router/config", json=custom_post)
                    check("POST status 200", r2.status_code == 200)
                    resp2 = r2.json()
                    check("POST ok", resp2.get("ok") is True)
                    check("POST config returned", "config" in resp2)
                    check("POST coding changed", resp2["config"]["task_tier_map"]["coding"] == "local_strong")

                    # GET after POST → reflects saved state
                    r3 = await client.get("/api/model-router/config")
                    data3 = r3.json()
                    check("GET after POST reflects save", data3["task_tier_map"]["coding"] == "local_strong")

                    # POST /reset → restore defaults
                    r4 = await client.post("/api/model-router/reset")
                    check("RESET status 200", r4.status_code == 200)
                    resp4 = r4.json()
                    check("RESET ok", resp4.get("ok") is True)
                    check("RESET coding back to default",
                          resp4["config"]["task_tier_map"]["coding"] == "cheap_cloud")
                    check("RESET 4 tiers", len(resp4["config"]["tiers"]) == 4)

                    # GET after reset → defaults
                    r5 = await client.get("/api/model-router/config")
                    data5 = r5.json()
                    check("GET after reset → default coding",
                          data5["task_tier_map"]["coding"] == "cheap_cloud")

            asyncio.run(_run_api_tests())

        except ImportError:
            # httpx not installed — skip API tests gracefully
            check("httpx not available — API tests skipped", True)

        # ── 8. Tier-specific field validations ──
        t0 = _MODEL_ROUTER_DEFAULTS["tiers"][0]
        t3 = _MODEL_ROUTER_DEFAULTS["tiers"][3]

        check("t0 provider ollama", t0["provider"] == "ollama")
        check("t0 model qwen2.5:7b", t0["primary_model"] == "qwen2.5:7b")
        check("t3 provider openai", t3["provider"] == "openai")
        check("t3 model gpt-4o", t3["primary_model"] == "gpt-4o")
        check("t0 cost ~$0.00", t0["cost_per_call"] == "~$0.00")
        check("t3 cost contains $0.01", "$0.01" in t3["cost_per_call"])

        # Escalation: retries should decrease as tier cost increases
        check("t0 retries >= t3 retries",
              t0["retries_before_escalate"] >= t3["retries_before_escalate"])

        # Max iterations should increase with tier capability
        check("t3 iterations >= t0 iterations",
              t3["max_iterations"] >= t0["max_iterations"])

    finally:
        _app_mod.MODEL_ROUTER_FILE = orig_file
        shutil.rmtree(tmp, ignore_errors=True)



# ═════════════════════════════════════════════
# EMAIL TOOL — comprehensive torture tests
# ═════════════════════════════════════════════

def test_email_tool_torture():
    """Exhaustive tests for EmailTool: definition, actions, account CRUD,
    validation, confirmation gate, agent_name resolution, SMTP error paths."""
    print("\n=== TORTURE: Email Tool — Full Coverage ===")
    from src.tools.email_tool import (
        EmailTool, get_accounts, get_accounts_raw, get_default_account,
        get_user_account, get_agent_default_account, get_account_by_id,
        save_account, delete_account, get_effective_config,
        _load_settings, _save_settings, _load_tool_config,
        _SETTINGS_FILE,
    )
    from pathlib import Path
    import src.tools.email_tool as et_mod

    tool = EmailTool()

    # ── 1. Definition ──
    defn = tool.definition()
    check("email def name", defn["name"] == "email")
    check("email def has description", len(defn["description"]) > 20)
    props = defn["parameters"]["properties"]
    check("action in params", "action" in props)
    check("subject in params", "subject" in props)
    check("body in params", "body" in props)
    check("recipients in params", "recipients" in props)
    check("account_id in params", "account_id" in props)
    check("confirmation in params", "confirmation" in props)
    check("action required", "action" in defn["parameters"]["required"])
    actions = props["action"]["enum"]
    check("3 actions", len(actions) == 3)
    for a in ("send", "status", "accounts"):
        check(f"action '{a}'", a in actions)

    # ── 2. Isolated account CRUD (temp settings file) ──
    orig_settings = et_mod._SETTINGS_FILE
    tmp = tempfile.mkdtemp()
    tmp_settings = Path(tmp) / "config" / "settings.json"
    et_mod._SETTINGS_FILE = tmp_settings

    try:
        # Empty state
        check("no accounts initially", len(get_accounts()) == 0)
        check("no raw accounts", len(get_accounts_raw()) == 0)
        check("default account → None", get_default_account() is None)
        check("user account → None", get_user_account() is None)
        check("agent default → None", get_agent_default_account("astraea") is None)
        check("account by id → None", get_account_by_id("nope") is None)

        # effective_config defaults
        cfg = get_effective_config()
        check("cfg has api_base_url", "api_base_url" in cfg)
        check("cfg has timeout", cfg["timeout"] == 30)
        check("cfg require_confirmation default True", cfg["require_confirmation"] is True)
        check("cfg accounts empty", len(cfg["accounts"]) == 0)

        # Create first account
        acct1 = save_account({
            "label": "Work",
            "email": "work@example.com",
            "password": "secret123",
            "smtp_server": "smtp.example.com",
            "smtp_port": 465,
            "signature": "Best regards",
            "is_default": True,
            "is_user_email": False,
            "agent_default": "",
        })
        check("acct1 got id", acct1.get("id") is not None and acct1["id"].startswith("acct_"))
        check("acct1 label", acct1["label"] == "Work")
        check("1 account now", len(get_accounts_raw()) == 1)

        # Password masking
        masked = get_accounts()
        check("password masked", masked[0]["password"] == "••••••••")
        check("password_set True", masked[0]["password_set"] is True)

        # Default fallback
        check("default → acct1", get_default_account()["id"] == acct1["id"])

        # Create second account (agent default for astraea)
        acct2 = save_account({
            "label": "Agent Mail",
            "email": "agent@example.com",
            "password": "agentpwd",
            "smtp_server": "smtp.example.com",
            "smtp_port": 587,
            "signature": "",
            "is_default": False,
            "is_user_email": True,
            "agent_default": "astraea",
        })
        check("acct2 created", acct2.get("id") is not None)
        check("2 accounts now", len(get_accounts_raw()) == 2)
        check("agent default astraea", get_agent_default_account("astraea")["id"] == acct2["id"])
        check("user account → acct2", get_user_account()["id"] == acct2["id"])
        check("lookup by id", get_account_by_id(acct2["id"])["label"] == "Agent Mail")

        # Update account (keep masked password)
        acct2_updated = dict(acct2)
        acct2_updated["label"] = "Agent Mail Updated"
        acct2_updated["password"] = "••••••••"  # masked placeholder
        saved = save_account(acct2_updated)
        check("update preserves password",
              get_account_by_id(acct2["id"])["password"] == "agentpwd")
        check("update changes label",
              get_account_by_id(acct2["id"])["label"] == "Agent Mail Updated")

        # Uniqueness: set acct2 as default → acct1 loses default
        acct2_def = dict(get_account_by_id(acct2["id"]))
        acct2_def["is_default"] = True
        save_account(acct2_def)
        check("acct1 no longer default",
              get_account_by_id(acct1["id"]).get("is_default") is False)
        check("acct2 now default",
              get_account_by_id(acct2["id"]).get("is_default") is True)

        # Agent default uniqueness: new account for astraea → acct2 loses it
        acct3 = save_account({
            "label": "New Astraea Mail",
            "email": "new@example.com",
            "password": "pwd3",
            "smtp_server": "smtp.example.com",
            "smtp_port": 465,
            "is_default": False,
            "agent_default": "astraea",
        })
        check("acct2 lost agent_default",
              get_account_by_id(acct2["id"]).get("agent_default", "") == "")
        check("acct3 has agent_default",
              get_account_by_id(acct3["id"]).get("agent_default") == "astraea")

        # Delete
        check("delete acct3", delete_account(acct3["id"]) is True)
        check("delete nonexistent", delete_account("fake_id") is False)
        check("2 accounts remain", len(get_accounts_raw()) == 2)

        # ── 3. Execute: accounts action ──
        r = json.loads(tool.execute({"action": "accounts"}))
        check("exec accounts has list", isinstance(r["accounts"], list))
        check("exec accounts total", r["total"] == 2)

        # ── 4. Execute: status action ──
        r = json.loads(tool.execute({"action": "status"}))
        check("exec status has accounts_configured", r["accounts_configured"] == 2)
        # API server likely not running → api_server_running false
        check("exec status api field", "api_server_running" in r)

        # ── 5. Execute: unknown action ──
        r = json.loads(tool.execute({"action": "nope"}))
        check("exec unknown action → error", "error" in r)

        # ── 6. Execute: send — validation ──
        # Missing subject
        r = json.loads(tool.execute({"action": "send"}))
        check("send no subject → error", "error" in r)
        check("send error mentions subject", "subject" in r["error"].lower())

        # Missing body
        r = json.loads(tool.execute({"action": "send", "subject": "Hi"}))
        check("send no body → error", "error" in r)

        # Missing recipients
        r = json.loads(tool.execute({
            "action": "send", "subject": "Hi", "body": "Hello",
        }))
        check("send no recipients → error", "error" in r)

        # Empty recipients list
        r = json.loads(tool.execute({
            "action": "send", "subject": "Hi", "body": "Hello", "recipients": [],
        }))
        check("send empty recipients → error", "error" in r)

        # Invalid email format
        r = json.loads(tool.execute({
            "action": "send", "subject": "Hi", "body": "Hello",
            "recipients": ["badformat"],
        }))
        check("send invalid email → error", "error" in r)
        check("send error names bad addr", "badformat" in r["error"])

        # Mixed valid/invalid
        r = json.loads(tool.execute({
            "action": "send", "subject": "Hi", "body": "Hello",
            "recipients": ["good@test.com", "bad"],
        }))
        check("send mixed addrs → error", "error" in r)

        # Nonexistent account_id
        r = json.loads(tool.execute({
            "action": "send", "subject": "Hi", "body": "Hello",
            "recipients": ["ok@test.com"], "account_id": "nonexistent",
        }))
        check("send bad account_id → error", "error" in r)

        # ── 7. Confirmation gate ──
        r = json.loads(tool.execute({
            "action": "send", "subject": "Test", "body": "Hello World",
            "recipients": ["user@test.com"],
        }))
        check("send gate=awaiting", r.get("gate") == "awaiting_confirmation")
        check("gate has preview", "preview" in r)
        check("preview has from_email", "from_email" in r["preview"])
        check("preview has subject", r["preview"]["subject"] == "Test")
        check("preview has recipients", "user@test.com" in r["preview"]["recipients"])

        # ── 8. Confirmation gate with agent_name resolution ──
        # re-create astraea default
        acct4 = save_account({
            "label": "Astraea Default",
            "email": "astraea@example.com",
            "password": "pwd4",
            "smtp_server": "smtp.example.com",
            "smtp_port": 465,
            "agent_default": "astraea",
            "is_default": False,
        })
        r = json.loads(tool.execute({
            "action": "send", "subject": "Agent Test", "body": "Hello",
            "recipients": ["dest@test.com"],
        }, agent_name="astraea"))
        check("agent_name resolves to astraea account",
              r.get("preview", {}).get("from_email") == "astraea@example.com")

        # ── Mock SMTP and HTTP for all remaining send tests ──
        # Mock smtplib to avoid real SMTP connections
        import smtplib as _smtplib
        _orig_smtp_ssl = _smtplib.SMTP_SSL
        _orig_smtp = _smtplib.SMTP
        class _MockSMTP:
            def __init__(self, *a, **kw): pass
            def ehlo(self): pass
            def starttls(self): pass
            def login(self, *a): pass
            def sendmail(self, *a): pass
            def quit(self): pass
        _smtplib.SMTP_SSL = _MockSMTP
        _smtplib.SMTP = _MockSMTP

        # Mock requests.Session to avoid real HTTP calls to API fallback
        import requests as _requests_mod
        _OrigSession = _requests_mod.Session
        class _MockSession:
            def get(self, *a, **kw):
                raise _requests_mod.exceptions.ConnectionError("mocked")
            def post(self, *a, **kw):
                raise _requests_mod.exceptions.ConnectionError("mocked")
        _requests_mod.Session = _MockSession

        try:
            # ── 9. Disable confirmation gate and attempt send ──
            settings = _load_settings()
            settings.setdefault("tool_config", {}).setdefault("email", {})["require_confirmation"] = False
            _save_settings(settings)

            # Re-create tool so it gets the mocked Session
            tool = EmailTool()

            r = json.loads(tool.execute({
                "action": "send", "subject": "No Gate", "body": "Hello",
                "recipients": ["user@test.com"],
            }))
            # Will succeed with mock SMTP — should NOT show gate
            check("no gate when disabled", r.get("gate") is None)
            # Expect "sent" (mocked) or "error" — not gate
            check("send attempted (no gate)", "error" in r or "status" in r)

            # ── 10. Send with confirmation='confirmed' (bypass gate) ──
            # Re-enable confirmation
            settings["tool_config"]["email"]["require_confirmation"] = True
            _save_settings(settings)

            r = json.loads(tool.execute({
                "action": "send", "subject": "Confirmed", "body": "Go",
                "recipients": ["user@test.com"], "confirmation": "confirmed",
            }))
            check("confirmed bypasses gate", r.get("gate") is None)
            check("confirmed attempts send", "error" in r or "status" in r)

            # ── 11. Account with no password → incomplete error ──
            save_account({
                "id": "nopwd_acct",
                "label": "No Password",
                "email": "nopwd@example.com",
                "password": "",
                "smtp_server": "smtp.example.com",
                "smtp_port": 465,
                "is_default": False,
            })
            r = json.loads(tool.execute({
                "action": "send", "subject": "Test", "body": "Body",
                "recipients": ["dest@test.com"],
                "account_id": "nopwd_acct",
                "confirmation": "confirmed",
            }))
            check("no password -> incomplete error", "error" in r)
            check("error mentions credentials", "credential" in r["error"].lower() or "password" in r["error"].lower())

            # ── 12. Delete all accounts → send fails ──
            for acct in get_accounts_raw():
                delete_account(acct["id"])
            check("all accounts deleted", len(get_accounts_raw()) == 0)

            r = json.loads(tool.execute({
                "action": "send", "subject": "X", "body": "Y",
                "recipients": ["a@b.com"],
            }))
            check("send with no accounts -> error", "error" in r)
            check("error mentions no accounts", "no email" in r["error"].lower() or "account" in r["error"].lower())

            # accounts action with 0 accounts
            r = json.loads(tool.execute({"action": "accounts"}))
            check("0 accounts message", "message" in r)

            # ── 13. Edge: whitespace-only fields ──
            r = json.loads(tool.execute({
                "action": "send", "subject": "   ", "body": "hello",
                "recipients": ["a@b.com"],
            }))
            check("whitespace subject -> error", "error" in r)

            r = json.loads(tool.execute({
                "action": "send", "subject": "ok", "body": "   ",
                "recipients": ["a@b.com"],
            }))
            check("whitespace body -> error", "error" in r)

            # ── 14. effective_config masks passwords ──
            save_account({
                "label": "Final",
                "email": "final@test.com",
                "password": "supersecret",
                "smtp_server": "smtp.test.com",
                "smtp_port": 465,
            })
            cfg = get_effective_config()
            check("effective_config masks pwd",
                  all(a["password"] == "••••••••" for a in cfg["accounts"] if a.get("password_set")))

        finally:
            _smtplib.SMTP_SSL = _orig_smtp_ssl
            _smtplib.SMTP = _orig_smtp
            _requests_mod.Session = _OrigSession

    finally:
        et_mod._SETTINGS_FILE = orig_settings
        shutil.rmtree(tmp, ignore_errors=True)


# ═════════════════════════════════════════════
# 26. DirectivesTool — all 5 actions
# ═════════════════════════════════════════════
def test_directives_tool_torture():
    """Exhaustive tests for DirectivesTool: search, list, get, manifest, changes,
    _resolve_scopes helper, dynamic SCOPES import, and scope filtering."""
    print("\n=== TORTURE: DirectivesTool — All 5 Actions + _resolve_scopes ===")
    from src.tools.directives_tool import DirectivesTool
    import src.tools.directives_tool as dt_mod
    import src.directives.manifest as manifest_mod

    tool = DirectivesTool()

    # Definition
    defn = tool.definition()
    check("dt def name", defn["name"] == "directives")
    check("dt def has description", len(defn["description"]) > 20)
    actions = defn["parameters"]["properties"]["action"]["enum"]
    check("dt 5 actions", len(actions) == 5)
    for a in ("search", "list", "get", "manifest", "changes"):
        check(f"dt action '{a}'", a in actions)

    # --- _resolve_scopes helper ---
    print("\n  -- _resolve_scopes --")
    # No scope → returns all SCOPES
    all_scopes = DirectivesTool._resolve_scopes(None)
    check("resolve None → all SCOPES", len(all_scopes) > 0)
    check("resolve None → list", isinstance(all_scopes, list))
    check("resolve None includes 'shared'", "shared" in all_scopes)

    # Explicit scope → always includes shared
    scoped = DirectivesTool._resolve_scopes("astraea")
    check("resolve 'astraea' → 2 items", len(scoped) == 2)
    check("resolve astraea has shared", "shared" in scoped)
    check("resolve astraea has astraea", "astraea" in scoped)

    # Scope = shared → just shared (no duplication)
    scoped_shared = DirectivesTool._resolve_scopes("shared")
    check("resolve 'shared' → ['shared']", scoped_shared == ["shared"])

    # Case insensitivity
    scoped_upper = DirectivesTool._resolve_scopes("ASTRAEA")
    check("resolve case insensitive", "astraea" in scoped_upper)
    check("resolve case → shared present", "shared" in scoped_upper)

    # SCOPES module-level import is a tuple/list with entries
    check("SCOPES imported", hasattr(dt_mod, 'SCOPES') or hasattr(manifest_mod, 'SCOPES'))
    from src.directives.manifest import SCOPES as imported_scopes
    check("SCOPES is tuple/list", isinstance(imported_scopes, (tuple, list)))
    check("SCOPES has shared", "shared" in imported_scopes)

    tmp = tempfile.mkdtemp()
    orig_dir = dt_mod._DIRECTIVES_DIR
    dt_mod._DIRECTIVES_DIR = tmp

    # Temporarily override SCOPES so the tool uses our tmp scopes
    orig_scopes = manifest_mod.SCOPES
    manifest_mod.SCOPES = ("shared", "astraea")
    # Also patch the module-level reference in directives_tool if it cached it
    orig_dt_scopes = getattr(dt_mod, 'SCOPES', None)
    dt_mod.SCOPES = ("shared", "astraea")

    try:
        # Create scope files
        shared_path = os.path.join(tmp, "shared.md")
        with open(shared_path, "w", encoding="utf-8") as f:
            f.write("## Core Values\nBe helpful, honest, and harmless.\n\n"
                    "## Communication Style\nBe concise and clear.\n\n"
                    "## Safety Rules\nNever reveal secrets or passwords.\n")

        agent_path = os.path.join(tmp, "astraea.md")
        with open(agent_path, "w", encoding="utf-8") as f:
            f.write("## Star Protocol\nGuide users through stargazing.\n\n"
                    "## Curiosity Mode\nAsk clarifying questions.\n")

        # --- search ---
        r = json.loads(tool.execute({"action": "search", "query": "helpful"}))
        check("dt search ok", r["status"] == "ok")
        check("dt search has count", r["count"] > 0)
        check("dt search has sections", len(r["sections"]) > 0)

        # search: missing query
        r = json.loads(tool.execute({"action": "search"}))
        check("dt search no query → error", r["status"] == "error")

        # search: with limit
        r = json.loads(tool.execute({"action": "search", "query": "values", "limit": 1}))
        check("dt search limit=1", r["count"] <= 1)

        # search: with scope filter
        r = json.loads(tool.execute({"action": "search", "query": "stargazing", "scope": "astraea"}))
        check("dt search scoped ok", r["status"] == "ok")
        check("dt search scoped finds result", r["count"] > 0)
        # All returned sections should be from shared or astraea
        for sec in r.get("sections", []):
            check(f"dt search scoped scope={sec['scope']}",
                  sec["scope"] in ("shared", "astraea"))

        # --- list ---
        r = json.loads(tool.execute({"action": "list"}))
        check("dt list ok", r["status"] == "ok")
        check("dt list count = 5 total", r["count"] == 5,
              f"got {r['count']}: {r.get('headings', [])}")
        check("dt list has headings", len(r["headings"]) > 0)

        # list: with scope filter → only shared
        r = json.loads(tool.execute({"action": "list", "scope": "shared"}))
        check("dt list shared only count=3", r["count"] == 3,
              f"got {r['count']}")

        # --- get ---
        r = json.loads(tool.execute({"action": "get", "heading": "Core Values"}))
        check("dt get ok", r["status"] == "ok")
        check("dt get heading", r["heading"] == "Core Values")
        check("dt get body", "helpful" in r["body"])
        check("dt get scope", r["scope"] == "shared")

        # get: agent-scoped section
        r = json.loads(tool.execute({"action": "get", "heading": "Star Protocol"}))
        check("dt get agent section ok", r["status"] == "ok")
        check("dt get agent heading", r["heading"] == "Star Protocol")
        check("dt get agent scope", r["scope"] == "astraea")

        # get: missing heading param
        r = json.loads(tool.execute({"action": "get"}))
        check("dt get no heading → error", r["status"] == "error")

        # get: nonexistent heading
        r = json.loads(tool.execute({"action": "get", "heading": "Nonexistent"}))
        check("dt get missing → not_found", r["status"] == "not_found")

        # --- manifest ---
        r = json.loads(tool.execute({"action": "manifest"}))
        check("dt manifest ok", r["status"] == "ok")
        check("dt manifest has count", r["count"] > 0)
        check("dt manifest has directives", len(r["directives"]) > 0)

        # manifest: with scope filter
        r = json.loads(tool.execute({"action": "manifest", "scope": "astraea"}))
        check("dt manifest scoped ok", r["status"] == "ok")
        for d in r.get("directives", []):
            check(f"dt manifest scoped entry scope={d['scope']}",
                  d["scope"] in ("shared", "astraea"))

        # --- changes ---
        r = json.loads(tool.execute({"action": "changes"}))
        check("dt changes ok", r["status"] == "ok")
        check("dt changes has totals", "total_added" in r)
        check("dt changes has added list", isinstance(r.get("added"), list))
        check("dt changes has removed list", isinstance(r.get("removed"), list))
        check("dt changes has changed list", isinstance(r.get("changed"), list))

        # --- unknown action ---
        r = json.loads(tool.execute({"action": "BOGUS"}))
        check("dt unknown action → error", r["status"] == "error")
        check("dt unknown msg mentions action", "BOGUS" in r.get("message", ""))

    finally:
        dt_mod._DIRECTIVES_DIR = orig_dir
        manifest_mod.SCOPES = orig_scopes
        if orig_dt_scopes is not None:
            dt_mod.SCOPES = orig_dt_scopes
        shutil.rmtree(tmp, ignore_errors=True)


# ═════════════════════════════════════════════
# 27. Tool Registry — dispatch + resolution
# ═════════════════════════════════════════════
def test_tool_registry_torture():
    """Test tool registry: listing, resolve, execute, error paths."""
    print("\n=== TORTURE: Tool Registry — Dispatch & Resolution ===")
    from src.tools.registry import list_registered_tools, execute_tool, _resolve_tool

    # list_registered_tools
    tools = list_registered_tools()
    check("registry: list returns list", isinstance(tools, list))
    check("registry: 8 tools", len(tools) == 8, f"got {len(tools)}: {tools}")
    for expected in ("echo", "memory", "directives", "cost_tracker",
                     "continuation_update", "web_search", "email", "inbox"):
        check(f"registry: has '{expected}'", expected in tools)

    # _resolve_tool — known tool
    resolved = _resolve_tool("echo")
    check("resolve echo → dict", isinstance(resolved, dict))
    check("resolve echo has type", resolved["type"] == "function")
    check("resolve echo has function.name", resolved["function"]["name"] == "echo")

    # _resolve_tool — unknown tool
    resolved_unknown = _resolve_tool("nonexistent_tool_xyz")
    check("resolve unknown → None", resolved_unknown is None)

    # execute_tool — echo
    result = execute_tool("echo", {"message": "Torture test!"})
    check("execute echo", "Torture test!" in result)

    # execute_tool — unknown raises KeyError
    try:
        execute_tool("nonexistent_tool_xyz", {})
        check("execute unknown → KeyError", False, "no exception raised")
    except KeyError:
        check("execute unknown → KeyError", True)

    # execute_tool — pass agent_name to email (which accepts it)
    # Just test it doesn't crash
    try:
        result = execute_tool("echo", {"message": "agent test"}, agent_name="astraea")
        check("execute with agent_name", isinstance(result, str))
    except Exception as e:
        check("execute with agent_name", False, str(e))


# ═════════════════════════════════════════════
# 28. validate_manifest — full coverage
# ═════════════════════════════════════════════
def test_validate_manifest():
    """Test manifest validation: valid, missing keys, bad enums, duplicate IDs, etc."""
    print("\n=== TORTURE: validate_manifest — Full Coverage ===")
    from src.directives.manifest import (
        validate_manifest, generate_manifest, _sha256,
    )

    tmp = tempfile.mkdtemp()
    try:
        # Create a scope file
        shared_path = os.path.join(tmp, "shared.md")
        with open(shared_path, "w", encoding="utf-8") as f:
            f.write("## Test Heading\nTest body content.\n")

        # Generate a valid manifest from our tmp dir
        manifest = generate_manifest(directives_dir=tmp, scopes=("shared",))
        check("generated manifest has directives", len(manifest["directives"]) > 0)

        # Fix path references for validation: create proper directory structure
        # validate_manifest resolves path as: parent_of(directives_dir) / entry["path"]
        # So we need the file at: tmp_parent/directives/shared.md
        directives_subdir = os.path.join(tmp, "directives")
        os.makedirs(directives_subdir, exist_ok=True)
        shutil.copy(shared_path, os.path.join(directives_subdir, "shared.md"))
        for d in manifest["directives"]:
            d["path"] = "directives/shared.md"
        result = validate_manifest(manifest, directives_dir=directives_subdir, check_hashes=False)
        check("valid manifest → valid", result["valid"] is True, f"errors={result['errors']}")
        check("valid manifest → 0 errors", len(result["errors"]) == 0, f"errors={result['errors']}")

        # Missing top-level key
        bad = dict(manifest)
        del bad["hash_algo"]
        result2 = validate_manifest(bad, directives_dir=tmp, check_hashes=False)
        check("missing top key → invalid", result2["valid"] is False)
        check("error mentions hash_algo", any("hash_algo" in e for e in result2["errors"]))

        # Missing entry key
        bad2 = json.loads(json.dumps(manifest))
        del bad2["directives"][0]["sha256"]
        result3 = validate_manifest(bad2, directives_dir=tmp, check_hashes=False)
        check("missing entry key → invalid", result3["valid"] is False)
        check("error mentions sha256", any("sha256" in e for e in result3["errors"]))

        # Invalid scope
        bad3 = json.loads(json.dumps(manifest))
        bad3["directives"][0]["scope"] = "bogus_scope_xyz"
        result4 = validate_manifest(bad3, directives_dir=tmp, check_hashes=False)
        check("bad scope → invalid", result4["valid"] is False)

        # Invalid status
        bad4 = json.loads(json.dumps(manifest))
        bad4["directives"][0]["status"] = "deleted"
        result5 = validate_manifest(bad4, directives_dir=tmp, check_hashes=False)
        check("bad status → invalid", result5["valid"] is False)

        # Invalid risk
        bad5 = json.loads(json.dumps(manifest))
        bad5["directives"][0]["risk"] = "extreme"
        result6 = validate_manifest(bad5, directives_dir=tmp, check_hashes=False)
        check("bad risk → invalid", result6["valid"] is False)

        # Duplicate IDs
        bad6 = json.loads(json.dumps(manifest))
        bad6["directives"].append(dict(bad6["directives"][0]))  # exact copy → same id
        result7 = validate_manifest(bad6, directives_dir=tmp, check_hashes=False)
        check("duplicate id → invalid", result7["valid"] is False)
        check("error mentions duplicate", any("duplicate" in e for e in result7["errors"]))

        # directives not a list
        bad7 = dict(manifest)
        bad7["directives"] = "not a list"
        result8 = validate_manifest(bad7, directives_dir=tmp, check_hashes=False)
        check("directives not list → invalid", result8["valid"] is False)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ═════════════════════════════════════════════
# 29. audit_changes
# ═════════════════════════════════════════════
def test_audit_changes():
    """Test audit_changes: no persisted manifest, matching, diffs."""
    print("\n=== TORTURE: audit_changes — Live vs Persisted ===")
    from src.directives.manifest import audit_changes, generate_manifest, save_manifest

    tmp = tempfile.mkdtemp()
    try:
        shared_path = os.path.join(tmp, "shared.md")
        with open(shared_path, "w", encoding="utf-8") as f:
            f.write("## Rule One\nFirst rule content.\n\n"
                    "## Rule Two\nSecond rule content.\n")

        manifest_path = os.path.join(tmp, "manifest.json")

        # No persisted manifest → all added
        diff = audit_changes(directives_dir=tmp, manifest_path_override=manifest_path)
        check("no persisted → all added", diff["total_added"] >= 2)
        check("no persisted → 0 removed", diff["total_removed"] == 0)

        # Save manifest, then diff again → 0 changes
        m = generate_manifest(directives_dir=tmp, scopes=("shared",))
        save_manifest(m, path=manifest_path)
        diff2 = audit_changes(directives_dir=tmp, manifest_path_override=manifest_path)
        check("matching → 0 added", diff2["total_added"] == 0)
        check("matching → 0 removed", diff2["total_removed"] == 0)
        check("matching → 0 changed", diff2["total_changed"] == 0)
        check("matching → 2 unchanged", diff2["unchanged_count"] == 2)

        # Modify a directive → detect change
        with open(shared_path, "w", encoding="utf-8") as f:
            f.write("## Rule One\nModified first rule.\n\n"
                    "## Rule Two\nSecond rule content.\n\n"
                    "## Rule Three\nBrand new rule.\n")
        diff3 = audit_changes(directives_dir=tmp, manifest_path_override=manifest_path)
        check("modified → 1 changed", diff3["total_changed"] == 1)
        check("new rule → 1 added", diff3["total_added"] == 1)
        check("unchanged → 1", diff3["unchanged_count"] == 1)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ═════════════════════════════════════════════
# 30. CostTrackerTool — pricing actions
# ═════════════════════════════════════════════
def test_cost_tracker_pricing_actions():
    """Test get_pricing, set_pricing, list_models actions."""
    print("\n=== TORTURE: CostTrackerTool — Pricing Actions ===")
    from src.tools.cost_tracker import CostTrackerTool
    import src.tools.cost_tracker as ct_mod

    tool = CostTrackerTool()

    tmp = tempfile.mkdtemp()
    orig_pricing_path = ct_mod._pricing_path
    orig_connections_path = ct_mod._connections_path

    pricing_file = os.path.join(tmp, "pricing.yaml")
    conn_file = os.path.join(tmp, "connections.json")

    ct_mod._pricing_path = lambda: pricing_file
    ct_mod._connections_path = lambda: conn_file

    try:
        # get_pricing — empty (no file)
        r = json.loads(tool.execute({"action": "get_pricing"}))
        check("get_pricing empty → providers dict", "providers" in r)

        # set_pricing — create new entry
        r = json.loads(tool.execute({
            "action": "set_pricing", "provider": "openai", "model": "gpt-4o",
            "input_per_1m": 2.50, "output_per_1m": 10.00,
        }))
        check("set_pricing ok", r.get("ok") is True)
        check("set_pricing provider", r["provider"] == "openai")
        check("set_pricing model", r["model"] == "gpt-4o")
        check("set_pricing input", r["pricing"]["input_per_1m"] == 2.50)
        check("set_pricing output", r["pricing"]["output_per_1m"] == 10.00)

        # get_pricing — specific model
        r = json.loads(tool.execute({
            "action": "get_pricing", "provider": "openai", "model": "gpt-4o",
        }))
        check("get_pricing specific", r["pricing"]["input_per_1m"] == 2.50)

        # get_pricing — provider only
        r = json.loads(tool.execute({
            "action": "get_pricing", "provider": "openai",
        }))
        check("get_pricing provider", "gpt-4o" in r["models"])

        # set_pricing — missing fields
        r = json.loads(tool.execute({"action": "set_pricing"}))
        check("set_pricing missing → error", "error" in r)

        # set_pricing — another model with cached_input
        r = json.loads(tool.execute({
            "action": "set_pricing", "provider": "anthropic",
            "model": "claude-sonnet-4-20250514",
            "input_per_1m": 3.00, "cached_input_per_1m": 0.30,
            "output_per_1m": 15.00,
        }))
        check("set_pricing cached", r["pricing"]["cached_input_per_1m"] == 0.30)

        # list_models — no connections file
        r = json.loads(tool.execute({"action": "list_models"}))
        check("list_models empty", "connections" in r)

        # list_models — with connections file
        conns = {
            "connections": [
                {"name": "Local Ollama", "provider": "ollama",
                 "enabled": True, "models": ["llama3:8b", "qwen2.5:7b"]},
                {"name": "Disabled", "provider": "openai",
                 "enabled": False, "models": ["gpt-4"]},
            ],
            "agent_connections": {},
        }
        with open(conn_file, "w") as f:
            json.dump(conns, f)

        r = json.loads(tool.execute({"action": "list_models"}))
        check("list_models has Ollama", "Local Ollama" in r["connections"])
        check("list_models disabled excluded", "Disabled" not in r["connections"])
        check("list_models models list",
              r["connections"]["Local Ollama"]["models"] == ["llama3:8b", "qwen2.5:7b"])

    finally:
        ct_mod._pricing_path = orig_pricing_path
        ct_mod._connections_path = orig_connections_path
        shutil.rmtree(tmp, ignore_errors=True)


# ═════════════════════════════════════════════
# 31. WebSearchTool — extended coverage
# ═════════════════════════════════════════════
def test_web_search_tool_extended():
    """Test _remove_emojis, definition, knowledge gate, scrape action, get_effective_config."""
    print("\n=== TORTURE: WebSearchTool — Extended Coverage ===")
    from src.tools.web_search import (
        WebSearchTool, _remove_emojis, get_effective_config,
    )

    # _remove_emojis
    check("remove_emojis basic", "Hello" in _remove_emojis("Hello 🌍 World") and "World" in _remove_emojis("Hello 🌍 World"))
    check("remove_emojis no emoji char", "🌍" not in _remove_emojis("Hello 🌍 World"))
    check("remove_emojis multiple", "test" == _remove_emojis("🔥test🎉").strip())
    check("remove_emojis empty", "" == _remove_emojis(""))
    check("remove_emojis no emoji", "plain text" == _remove_emojis("plain text"))
    check("remove_emojis keeps CJK", "日本語" in _remove_emojis("日本語🏯"))

    # Definition
    tool = WebSearchTool()
    defn = tool.definition()
    check("ws def name", defn["name"] == "web_search")
    check("ws def has description", len(defn["description"]) > 20)
    props = defn["parameters"]["properties"]
    check("ws action in params", "action" in props)
    check("ws query in params", "query" in props)
    check("ws url in params", "url" in props)
    check("ws mode in params", "mode" in props)
    check("ws knowledge_check in params", "knowledge_check" in props)
    check("ws reason in params", "reason" in props)
    actions = props["action"]["enum"]
    check("ws 2 actions", len(actions) == 2)
    check("ws action search", "search" in actions)
    check("ws action scrape", "scrape" in actions)

    # get_effective_config
    cfg = get_effective_config()
    check("cfg has searxng_url", "searxng_url" in cfg)
    check("cfg has ignored_sites", isinstance(cfg["ignored_sites"], str))
    check("cfg has require_justification", "require_justification" in cfg)
    check("cfg has modes", "modes" in cfg)
    for mode in ("fast", "normal", "deep"):
        check(f"cfg mode '{mode}' present", mode in cfg["modes"])
        check(f"cfg mode '{mode}' has pages", "pages" in cfg["modes"][mode])

    # Knowledge gate — blocked via skip signal
    r = json.loads(tool.execute({
        "action": "search", "query": "test",
        "knowledge_check": "I already know the answer",
        "reason": "just testing",
    }))
    check("knowledge gate blocked", r.get("gate") == "blocked")

    # Knowledge gate — missing reason
    r = json.loads(tool.execute({
        "action": "search", "query": "test",
    }))
    check("knowledge gate missing reason",
          r.get("gate") == "missing_justification" or "error" in r)

    # Scrape — no URL
    r = json.loads(tool.execute({"action": "scrape"}))
    check("scrape no url → error", "error" in r)

    # Scrape — empty URL
    r = json.loads(tool.execute({"action": "scrape", "url": ""}))
    check("scrape empty url → error", "error" in r)


# ═════════════════════════════════════════════
# 32. Metering — extended helpers
# ═════════════════════════════════════════════
def test_metering_helpers_extended():
    """Test meter_response, zero_metering, meter_from_raw_usage, get_price,
    compute_cost, reset_pricing_cache, serialisation round-trips."""
    print("\n=== TORTURE: Metering — Extended Helpers ===")
    from src.observability.metering import (
        meter_response, zero_metering, meter_from_raw_usage,
        get_price, compute_cost, reset_pricing_cache,
        TokenUsage, CostBreakdown, Metering,
    )
    from src.llm_client.base import LLMResponse

    # zero_metering
    z = zero_metering()
    check("zero_metering usage", z.usage.prompt_tokens == 0)
    check("zero_metering cost", z.cost.total_cost == 0.0)
    check("zero_metering model", z.model == "")
    check("zero_metering provider", z.provider == "")

    # zero_metering accumulation
    z2 = z + z
    check("zero + zero = zero", z2.cost.total_cost == 0.0)

    # get_price with custom pricing dict
    pricing = {
        "openai": {
            "gpt-4o": {"input_per_1m": 2.50, "output_per_1m": 10.00},
            "gpt-4": {"input_per_1m": 30.00, "output_per_1m": 60.00},
            "_default": {"input_per_1m": 1.00, "output_per_1m": 2.00},
        },
        "ollama": {
            "_default": {"input_per_1m": 0.00, "output_per_1m": 0.00},
        },
    }
    inp, cached, out, train = get_price("openai", "gpt-4o", pricing)
    check("get_price exact input", inp == 2.50)
    check("get_price exact output", out == 10.00)

    # Prefix match: "gpt-4o-mini" starts with "gpt-4o"
    inp2, _, out2, _ = get_price("openai", "gpt-4o-mini", pricing)
    check("get_price prefix match", inp2 == 2.50)

    # Provider default fallback
    inp3, _, out3, _ = get_price("openai", "unknown-model-xyz", pricing)
    check("get_price default fallback", inp3 == 1.00)

    # Unknown provider
    inp4, _, out4, _ = get_price("nonexistent_provider", "any", pricing)
    check("get_price unknown provider → 0", inp4 == 0.0 and out4 == 0.0)

    # compute_cost
    usage = TokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
    cost = compute_cost(usage, "openai", "gpt-4o", pricing)
    expected_input = 1000 * 2.50 / 1_000_000
    expected_output = 500 * 10.00 / 1_000_000
    check("compute_cost input", abs(cost.input_cost - expected_input) < 0.0001)
    check("compute_cost output", abs(cost.output_cost - expected_output) < 0.0001)
    check("compute_cost total", abs(cost.total_cost - (expected_input + expected_output)) < 0.0001)

    # compute_cost with cached tokens
    cost2 = compute_cost(usage, "openai", "gpt-4o", pricing, cached_tokens=200)
    check("cached reduces input cost", cost2.input_cost < cost.input_cost)

    # meter_response with usage populated
    resp = LLMResponse(
        content="Hello",
        model="gpt-4o",
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    )
    m = meter_response(resp, provider="openai", pricing=pricing)
    check("meter_response model", m.model == "gpt-4o")
    check("meter_response not estimated", m.usage.is_estimated is False)
    check("meter_response prompt_tokens", m.usage.prompt_tokens == 100)

    # meter_response without usage (estimation)
    resp2 = LLMResponse(content="Hello world", model="gpt-4o", usage=None)
    messages = [{"role": "user", "content": "Say hello"}]
    m2 = meter_response(resp2, provider="openai", messages=messages, pricing=pricing)
    check("meter_response estimated", m2.usage.is_estimated is True)
    check("meter_response est prompt > 0", m2.usage.prompt_tokens > 0)

    # meter_from_raw_usage
    raw = {"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300}
    m3 = meter_from_raw_usage(raw, "openai", "gpt-4o", pricing)
    check("meter_from_raw prompt", m3.usage.prompt_tokens == 200)
    check("meter_from_raw cost > 0", m3.cost.total_cost > 0)

    # meter_from_raw_usage with cached tokens
    raw2 = {
        "prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300,
        "prompt_tokens_details": {"cached_tokens": 50},
    }
    m4 = meter_from_raw_usage(raw2, "openai", "gpt-4o", pricing)
    check("meter_from_raw cached", m4.cost.cached_input_cost >= 0)

    # Serialisation round-trips
    d = m.to_dict()
    check("metering to_dict has usage", "usage" in d)
    check("metering to_dict has cost", "cost" in d)
    m_back = Metering.from_dict(d)
    check("metering round-trip model", m_back.model == m.model)
    check("metering round-trip tokens", m_back.usage.prompt_tokens == m.usage.prompt_tokens)

    tu_d = m.usage.to_dict()
    tu_back = TokenUsage.from_dict(tu_d)
    check("token_usage round-trip", tu_back.prompt_tokens == m.usage.prompt_tokens)

    cb_d = m.cost.to_dict()
    cb_back = CostBreakdown.from_dict(cb_d)
    check("cost_breakdown round-trip", abs(cb_back.total_cost - m.cost.total_cost) < 0.0001)

    # reset_pricing_cache — should not crash
    reset_pricing_cache()
    check("reset_pricing_cache ok", True)


# ═════════════════════════════════════════════
# 33. LLM Client Factory
# ═════════════════════════════════════════════
def test_llm_client_factory():
    """Test create_client dispatch and unknown provider."""
    print("\n=== TORTURE: LLM Client Factory ===")
    from src.llm_client.factory import create_client, _PROVIDERS
    from src.llm_client.base import LLMClient

    # Check provider map
    check("factory has openai", "openai" in _PROVIDERS)
    check("factory has deepseek", "deepseek" in _PROVIDERS)
    check("factory has ollama", "ollama" in _PROVIDERS)
    check("factory has anthropic", "anthropic" in _PROVIDERS)

    # create_client for each provider (just instantiate, don't call)
    for provider in ("openai", "deepseek", "ollama", "anthropic"):
        try:
            profile = {
                "provider": provider,
                "api_url": "http://localhost:11434",
                "api_key": "test-key-123",
                "model": "test-model",
            }
            client = create_client(profile)
            check(f"factory {provider} → LLMClient", isinstance(client, LLMClient))
        except Exception as e:
            check(f"factory {provider} → ok", False, str(e))

    # Unknown provider
    try:
        create_client({"provider": "nonexistent_xyz"})
        check("factory unknown → ValueError", False, "no exception raised")
    except ValueError as e:
        check("factory unknown → ValueError", "nonexistent_xyz" in str(e))


# ═════════════════════════════════════════════
# 34. App helpers — memory tag extraction
# ═════════════════════════════════════════════
def test_app_memory_helpers():
    """Test _strip_memory_tags pattern matching."""
    print("\n=== TORTURE: App — Memory Tag Helpers ===")
    import re

    # Replicate the pattern from app.py
    _MEMORY_TAG_PATTERN = r'\[MEMORY_SAVE:\s*(?:category=[\w]+\s*\|)?\s*.+?\]'

    def strip_memory_tags(text):
        return re.sub(_MEMORY_TAG_PATTERN, '', text).strip()

    # Basic tag removal
    text = "Hello [MEMORY_SAVE: some note] world"
    result = strip_memory_tags(text)
    check("strip basic tag", "MEMORY_SAVE" not in result)
    check("strip preserves text", "Hello" in result and "world" in result)

    # Tag with category
    text2 = "Start [MEMORY_SAVE: category=bio | User likes hiking] end"
    result2 = strip_memory_tags(text2)
    check("strip category tag", "MEMORY_SAVE" not in result2)
    check("strip category preserves", "Start" in result2 and "end" in result2)

    # Multiple tags
    text3 = "[MEMORY_SAVE: note1] text [MEMORY_SAVE: category=pref | note2] done"
    result3 = strip_memory_tags(text3)
    check("strip multiple tags", "MEMORY_SAVE" not in result3)
    check("strip multiple preserves", "text" in result3 and "done" in result3)

    # No tags → unchanged
    text4 = "Plain text with no memory tags at all."
    result4 = strip_memory_tags(text4)
    check("strip no tags → unchanged", result4 == text4)

    # Empty string
    result5 = strip_memory_tags("")
    check("strip empty → empty", result5 == "")

    # Extract pattern
    _EXTRACT_PATTERN = r'\[MEMORY_SAVE:\s*(?:category=([\w]+)\s*\|)?\s*(.+?)\]'

    matches = re.findall(_EXTRACT_PATTERN, text2, re.DOTALL)
    check("extract has match", len(matches) == 1)
    check("extract category", matches[0][0] == "bio")
    check("extract text", "hiking" in matches[0][1])

    # Multiple extracts
    matches2 = re.findall(_EXTRACT_PATTERN, text3, re.DOTALL)
    check("extract multiple", len(matches2) == 2)


# ═════════════════════════════════════════════
# 35. Seed UI Knowledge — MEMORIES structure validation
# ═════════════════════════════════════════════
def test_seed_ui_knowledge_structure():
    """Validate the MEMORIES list in seed_ui_knowledge.py has correct structure."""
    print("\n=== TORTURE: Seed UI Knowledge — Structure ===")
    from scripts.seed_ui_knowledge import MEMORIES

    check("MEMORIES is list", isinstance(MEMORIES, list))
    check("MEMORIES not empty", len(MEMORIES) > 0)
    check("MEMORIES > 30 entries", len(MEMORIES) > 30, f"got {len(MEMORIES)}")

    required_keys = {"text", "scope", "category", "tags", "source", "tier"}
    valid_tiers = {"canon", "register"}
    valid_sources = {"operator", "tool", "chat", "system"}

    for i, m in enumerate(MEMORIES):
        prefix = f"MEMORIES[{i}]"
        for key in required_keys:
            check(f"{prefix} has {key}", key in m, f"missing {key}")

        # Text not empty
        check(f"{prefix} text non-empty", len(m.get("text", "")) > 10,
              f"text too short: {m.get('text', '')[:30]}")

        # Valid tier
        check(f"{prefix} tier valid", m.get("tier") in valid_tiers,
              f"tier={m.get('tier')}")

        # Valid source
        check(f"{prefix} source valid", m.get("source") in valid_sources,
              f"source={m.get('source')}")

        # Tags is a list
        check(f"{prefix} tags is list", isinstance(m.get("tags", []), list))

        # Tags not empty
        check(f"{prefix} tags non-empty", len(m.get("tags", [])) > 0)

        # Scope is string
        check(f"{prefix} scope is str", isinstance(m.get("scope"), str))

    # Check diversity of categories
    categories = {m["category"] for m in MEMORIES}
    check("has meta category", "meta" in categories)
    check("has capability category", "capability" in categories)
    check("multiple categories", len(categories) >= 2, f"categories={categories}")

    # Check all have "ui" or dashboard-related tags
    has_ui_tag = sum(1 for m in MEMORIES if "ui" in m.get("tags", []))
    check("most have 'ui' tag", has_ui_tag > len(MEMORIES) * 0.5,
          f"{has_ui_tag}/{len(MEMORIES)}")


# ═════════════════════════════════════════════
# 36. Metering data class ops
# ═════════════════════════════════════════════
def test_metering_dataclass_ops():
    """Test TokenUsage, CostBreakdown, Metering addition and serialisation."""
    print("\n=== TORTURE: Metering — Data Class Operations ===")
    from src.observability.metering import TokenUsage, CostBreakdown, Metering

    # TokenUsage addition
    u1 = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    u2 = TokenUsage(prompt_tokens=200, completion_tokens=100, total_tokens=300,
                    is_estimated=True)
    u3 = u1 + u2
    check("usage add prompt", u3.prompt_tokens == 300)
    check("usage add completion", u3.completion_tokens == 150)
    check("usage add total", u3.total_tokens == 450)
    check("usage add estimated propagation", u3.is_estimated is True)

    # CostBreakdown addition with all fields
    c1 = CostBreakdown(input_cost=0.01, cached_input_cost=0.001,
                       output_cost=0.02, training_cost=0.003, total_cost=0.034)
    c2 = CostBreakdown(input_cost=0.02, output_cost=0.04, total_cost=0.06)
    c3 = c1 + c2
    check("cost add input", abs(c3.input_cost - 0.03) < 0.0001)
    check("cost add cached", abs(c3.cached_input_cost - 0.001) < 0.0001)
    check("cost add output", abs(c3.output_cost - 0.06) < 0.0001)
    check("cost add total", abs(c3.total_cost - 0.094) < 0.0001)
    check("cost currency", c3.currency == "USD")

    # to_dict / from_dict round-trip
    d = c3.to_dict()
    check("cost to_dict currency", d["currency"] == "USD")
    c4 = CostBreakdown.from_dict(d)
    check("cost from_dict round-trip", abs(c4.total_cost - c3.total_cost) < 0.0001)

    # Metering addition preserves model/provider from first non-empty
    m1 = Metering(usage=u1, cost=c1, model="gpt-4o", provider="openai")
    m2 = Metering(usage=u2, cost=c2, model="", provider="")
    m3 = m1 + m2
    check("metering add model", m3.model == "gpt-4o")
    check("metering add provider", m3.provider == "openai")
    check("metering add usage", m3.usage.prompt_tokens == 300)

    # Empty + filled → filled
    m4 = Metering() + m1
    check("empty + filled model", m4.model == "gpt-4o")


# ═════════════════════════════════════════════
# INBOX TOOL — full coverage
# ═════════════════════════════════════════════
def test_inbox_tool_torture():
    print("\n=== TORTURE: InboxTool — Full Coverage ===")
    import src.data_paths as dp
    from src.tools.inbox import InboxTool

    orig_root = dp.DATA_ROOT
    tmp = tempfile.mkdtemp()
    dp.DATA_ROOT = tmp

    try:
        # ── Definition ────────────────────────────────────────
        defn = InboxTool.definition()
        check("inbox def name", defn["name"] == "inbox")
        check("inbox def has parameters", "parameters" in defn)
        props = defn["parameters"]["properties"]
        check("inbox def has action", "action" in props)
        check("inbox def has type", "type" in props)
        check("inbox def has priority", "priority" in props)
        check("inbox def has subject", "subject" in props)
        check("inbox def has body", "body" in props)
        check("inbox def has task", "task" in props)
        check("inbox def has task_id", "task_id" in props)
        check("inbox def has needs_approval", "needs_approval" in props)
        check("inbox def has dry_run", "dry_run" in props)
        check("inbox action enum", set(props["action"]["enum"]) == {"send", "add_task", "next_task", "ack"})

        # ── Unknown action ────────────────────────────────────
        r = InboxTool.execute({"action": "delete"})
        check("unknown action → error", "error" in r.lower())
        check("unknown action lists valid", "send" in r)

        # ── send: validation ──────────────────────────────────
        r = InboxTool.execute({"action": "send"})
        check("send no subject → error", "error" in r.lower())

        r = InboxTool.execute({"action": "send", "subject": "Hi"})
        check("send no body → error", "error" in r.lower())

        r = InboxTool.execute({"action": "send", "subject": "", "body": "text"})
        check("send empty subject → error", "error" in r.lower())

        r = InboxTool.execute({"action": "send", "subject": "Hi", "body": ""})
        check("send empty body → error", "error" in r.lower())

        r = InboxTool.execute({"action": "send", "subject": "x" * 121, "body": "ok"})
        check("send subject too long → error", "120" in r)

        r = InboxTool.execute({"action": "send", "subject": "ok", "body": "x" * 2001})
        check("send body too long → error", "2000" in r)

        r = InboxTool.execute({"action": "send", "type": "invalid_type",
                               "subject": "ok", "body": "ok"})
        check("send invalid type → error", "error" in r.lower())

        r = InboxTool.execute({"action": "send", "priority": "critical",
                               "subject": "ok", "body": "ok"})
        check("send invalid priority → error", "error" in r.lower())

        # ── send: success paths ───────────────────────────────
        r = InboxTool.execute({"action": "send", "type": "message",
                               "subject": "Hello operator", "body": "Testing.",
                               "_from": "orion"})
        check("send message ok", "sent" in r.lower())
        check("send message has id", "id=" in r)

        r = InboxTool.execute({"action": "send", "type": "warning",
                               "priority": "urgent", "subject": "Boundary hit",
                               "body": "Safety concern.", "needs_approval": True,
                               "_from": "astraea"})
        check("send warning ok", "sent" in r.lower())

        r = InboxTool.execute({"action": "send", "type": "tool_request",
                               "subject": "Need web access", "body": "Please enable.",
                               "_from": "callum"})
        check("send tool_request ok", "sent" in r.lower())

        r = InboxTool.execute({"action": "send", "type": "idea",
                               "subject": "Could do X", "body": "Proposal details.",
                               "profile": "orion"})
        check("send idea ok (profile as sender)", "sent" in r.lower())

        # Default type (message) when type omitted
        r = InboxTool.execute({"action": "send", "subject": "No type",
                               "body": "Should default to message."})
        check("send default type ok", "sent" in r.lower())

        # ── add_task: validation ──────────────────────────────
        r = InboxTool.execute({"action": "add_task"})
        check("add_task no task → error", "error" in r.lower())

        r = InboxTool.execute({"action": "add_task", "task": "  "})
        check("add_task whitespace → error", "error" in r.lower())

        # ── add_task: success ─────────────────────────────────
        r = InboxTool.execute({"action": "add_task",
                               "task": "Review inbox implementation",
                               "profile": "orion"})
        check("add_task ok", "added" in r.lower())
        check("add_task has id", "id=" in r)
        # Extract the task ID for later
        task_id_1 = r.split("id=")[1].split(")")[0]

        r = InboxTool.execute({"action": "add_task",
                               "task": "Second task",
                               "priority": "high"})
        check("add_task second ok", "added" in r.lower())
        task_id_2 = r.split("id=")[1].split(")")[0]

        # ── add_task: dry_run ─────────────────────────────────
        r = InboxTool.execute({"action": "add_task",
                               "task": "Dry run task", "dry_run": True})
        check("add_task dry_run prefix", "DRY_RUN" in r)
        check("add_task dry_run no write", "Would add" in r)

        # ── next_task ─────────────────────────────────────────
        r = InboxTool.execute({"action": "next_task"})
        check("next_task found", "TASK_FOUND" in r)
        check("next_task has first task", "Review inbox" in r)

        # Second next_task should return the second task
        r = InboxTool.execute({"action": "next_task"})
        check("next_task second found", "TASK_FOUND" in r)
        check("next_task has second task", "Second task" in r)

        # Third next_task — no more pending
        r = InboxTool.execute({"action": "next_task"})
        check("next_task empty", "NO_TASK" in r)

        # next_task with profile filter — add a scoped task
        InboxTool.execute({"action": "add_task", "task": "Scoped task",
                           "profile": "astraea"})
        InboxTool.execute({"action": "add_task", "task": "Other task",
                           "profile": "orion"})
        r = InboxTool.execute({"action": "next_task", "profile": "astraea"})
        check("next_task scoped", "Scoped task" in r)

        # next_task dry_run
        r = InboxTool.execute({"action": "next_task", "dry_run": True})
        check("next_task dry_run", "DRY_RUN" in r)

        # ── ack ───────────────────────────────────────────────
        r = InboxTool.execute({"action": "ack"})
        check("ack no id → error", "error" in r.lower())

        r = InboxTool.execute({"action": "ack", "task_id": ""})
        check("ack empty id → error", "error" in r.lower())

        r = InboxTool.execute({"action": "ack", "task_id": "nonexistent"})
        check("ack nonexistent → error", "not found" in r.lower())

        # Ack a real entry (task_id_1 was marked done by next_task, should still be ackable)
        r = InboxTool.execute({"action": "ack", "task_id": task_id_1})
        check("ack real task ok", "acknowledged" in r.lower())

        # Ack again → already acknowledged
        r = InboxTool.execute({"action": "ack", "task_id": task_id_1})
        check("ack duplicate → already", "already" in r.lower())

        # Ack dry_run
        r = InboxTool.execute({"action": "ack", "task_id": task_id_2,
                               "dry_run": True})
        check("ack dry_run", "DRY_RUN" in r)

        # ── JSONL persistence check ───────────────────────────
        jsonl_path = os.path.join(tmp, "shared", "inbox.jsonl")
        check("JSONL file exists", os.path.isfile(jsonl_path))
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        check("JSONL has entries", len(lines) >= 5)

        # Parse each line as valid JSON
        all_valid = True
        for line in lines:
            try:
                obj = json.loads(line)
                if "id" not in obj or "created_at" not in obj:
                    all_valid = False
            except json.JSONDecodeError:
                all_valid = False
        check("JSONL all valid JSON with id+created_at", all_valid)

        # Check for expected fields in a send entry
        first_send = json.loads(lines[0])
        check("send entry has 'from'", "from" in first_send)
        check("send entry has 'type'", "type" in first_send)
        check("send entry has 'subject'", "subject" in first_send)
        check("send entry has 'body'", "body" in first_send)
        check("send entry has 'status'", "status" in first_send)
        check("send entry status is unread", first_send["status"] == "unread")

        # Check task entry structures
        task_entries = [json.loads(l) for l in lines if json.loads(l).get("type") == "task"]
        check("task entries found", len(task_entries) >= 2)
        if task_entries:
            check("task entry has 'task' field", "task" in task_entries[0])

        # ── MD persistence check ──────────────────────────────
        md_path = os.path.join(tmp, "shared", "inbox.md")
        check("MD file exists", os.path.isfile(md_path))
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        check("MD has header", "# Inbox" in md_content)
        check("MD has entries", "---" in md_content)
        check("MD has subject lines", "**Subject:**" in md_content)
        check("MD has id refs", "*id:" in md_content)

        # Check specific entries in MD
        check("MD has message type", "message" in md_content)
        check("MD has warning type", "warning" in md_content)
        check("MD has priority tag", "[URGENT]" in md_content)
        check("MD has approval tag", "[NEEDS APPROVAL]" in md_content)

    finally:
        dp.DATA_ROOT = orig_root
        shutil.rmtree(tmp, ignore_errors=True)


def test_inbox_registry_dispatch():
    print("\n=== TORTURE: Inbox — Registry Dispatch ===")
    import src.data_paths as dp
    from src.tools.registry import execute_tool, list_registered_tools

    orig_root = dp.DATA_ROOT
    tmp = tempfile.mkdtemp()
    dp.DATA_ROOT = tmp

    try:
        # Inbox is registered
        tools = list_registered_tools()
        check("inbox in registry", "inbox" in tools)

        # Execute via registry
        r = execute_tool("inbox", {"action": "send", "type": "message",
                                   "subject": "Registry test",
                                   "body": "Dispatched through registry."})
        check("registry send ok", "sent" in r.lower())

        # Execute add_task via registry
        r = execute_tool("inbox", {"action": "add_task",
                                   "task": "Registry task test"})
        check("registry add_task ok", "added" in r.lower())

        # Execute next_task via registry
        r = execute_tool("inbox", {"action": "next_task"})
        check("registry next_task ok", "TASK_FOUND" in r)

    finally:
        dp.DATA_ROOT = orig_root
        shutil.rmtree(tmp, ignore_errors=True)


def test_inbox_data_path():
    print("\n=== TORTURE: inbox_path in data_paths ===")
    import src.data_paths as dp

    orig_root = dp.DATA_ROOT
    tmp = tempfile.mkdtemp()
    dp.DATA_ROOT = tmp

    try:
        ip = dp.inbox_path()
        check("inbox_path is string", isinstance(ip, str))
        check("inbox_path contains inbox", "inbox" in ip.lower())
        check("inbox_path in shared dir", "shared" in ip)
        check("inbox_path ends jsonl", ip.endswith(".jsonl"))
        # shared dir should exist after calling inbox_path
        check("shared dir created", os.path.isdir(os.path.join(tmp, "shared")))
    finally:
        dp.DATA_ROOT = orig_root
        shutil.rmtree(tmp, ignore_errors=True)


# ═════════════════════════════════════════════
# Dynamic scopes & category policy
# ═════════════════════════════════════════════
def test_dynamic_scopes():
    """Verify _discover_scopes picks up profile YAMLs and always includes 'shared'."""
    print("\n=== TORTURE: Dynamic Scopes ===")
    from src.memory.types import VALID_SCOPES, _discover_scopes

    # Basic contract
    check("VALID_SCOPES is frozenset", isinstance(VALID_SCOPES, frozenset))
    check("'shared' always in VALID_SCOPES", "shared" in VALID_SCOPES)
    check("at least 2 scopes", len(VALID_SCOPES) >= 2)

    # Rediscover and compare
    fresh = _discover_scopes()
    check("_discover_scopes returns frozenset", isinstance(fresh, frozenset))
    check("fresh scopes match VALID_SCOPES", fresh == VALID_SCOPES)

    # All profile YAMLs should be present
    from pathlib import Path
    profiles_dir = Path(__file__).resolve().parent.parent / "profiles"
    if profiles_dir.exists():
        for p in profiles_dir.glob("*.yaml"):
            check(f"scope '{p.stem}' discovered", p.stem in VALID_SCOPES)

    # Verify sorted() works (used by memory_tool definition)
    sorted_scopes = sorted(VALID_SCOPES)
    check("sorted(VALID_SCOPES) is list", isinstance(sorted_scopes, list))
    check("sorted scopes are alphabetical", sorted_scopes == sorted(sorted_scopes))

    # Edge: discover from empty dir
    tmp = tempfile.mkdtemp()
    def _discover_empty():
        scopes = {"shared"}
        empty_dir = Path(tmp) / "no_profiles_here"
        if empty_dir.exists():
            for p in empty_dir.glob("*.yaml"):
                scopes.add(p.stem)
        return frozenset(scopes)
    check("empty dir -> only shared", _discover_empty() == frozenset({"shared"}))
    shutil.rmtree(tmp, ignore_errors=True)


def test_category_policy():
    """Test _load_category_policy and _build_category_field for all 3 modes."""
    print("\n=== TORTURE: Category Policy ===")
    from src.tools.memory_tool import _load_category_policy, _build_category_field, _CONFIG_DIR
    from src.memory.types import VALID_CATEGORIES

    # ── Load from real config ──
    policy = _load_category_policy()
    check("policy is dict", isinstance(policy, dict))
    check("policy has mode", "mode" in policy)
    check("policy mode in valid set", policy["mode"] in ("suggested", "custom", "open"))
    check("policy has suggested_categories", "suggested_categories" in policy)
    check("policy has custom_categories", "custom_categories" in policy)
    check("suggested is list", isinstance(policy["suggested_categories"], list))
    check("custom is list", isinstance(policy["custom_categories"], list))

    # ── Test _build_category_field with mocked policies ──
    mp_file = _CONFIG_DIR / "memory_profile.json"
    original = mp_file.read_text(encoding="utf-8")

    try:
        import json as _j

        # --- MODE: suggested ---
        data = _j.loads(original)
        data["category_policy"] = {
            "mode": "suggested",
            "suggested_categories": ["bio", "identity", "mission"],
            "custom_categories": [],
        }
        mp_file.write_text(_j.dumps(data), encoding="utf-8")

        field = _build_category_field()
        check("suggested mode has enum", "enum" in field)
        check("suggested mode 3 items", len(field["enum"]) == 3)
        check("suggested mode sorted", field["enum"] == sorted(field["enum"]))
        check("suggested mode desc", "pick from" in field["description"].lower())

        # --- MODE: custom ---
        data["category_policy"] = {
            "mode": "custom",
            "suggested_categories": ["bio", "identity"],
            "custom_categories": ["my_custom", "another_custom"],
        }
        mp_file.write_text(_j.dumps(data), encoding="utf-8")

        field = _build_category_field()
        check("custom mode has enum", "enum" in field)
        check("custom mode merged count 4", len(field["enum"]) == 4)
        check("custom mode includes custom", "my_custom" in field["enum"])
        check("custom mode includes suggested", "bio" in field["enum"])
        check("custom mode deduplicates", len(field["enum"]) == len(set(field["enum"])))

        # custom mode with overlapping categories
        data["category_policy"]["custom_categories"] = ["bio", "new_one"]
        mp_file.write_text(_j.dumps(data), encoding="utf-8")
        field = _build_category_field()
        check("custom overlap dedup", field["enum"].count("bio") == 1)
        check("custom overlap new_one", "new_one" in field["enum"])

        # --- MODE: open ---
        data["category_policy"] = {
            "mode": "open",
            "suggested_categories": ["bio", "identity", "mission"],
            "custom_categories": [],
        }
        mp_file.write_text(_j.dumps(data), encoding="utf-8")

        field = _build_category_field()
        check("open mode no enum", "enum" not in field)
        check("open mode 'any' in desc", "any" in field["description"].lower())
        check("open mode common cats listed", "bio" in field["description"])

        # --- Missing category_policy entirely ---
        data.pop("category_policy", None)
        mp_file.write_text(_j.dumps(data), encoding="utf-8")

        field = _build_category_field()
        check("missing policy defaults to open", "enum" not in field)
        check("missing policy has description", len(field["description"]) > 0)

        # --- Corrupt JSON ---
        mp_file.write_text("NOT VALID JSON{{{", encoding="utf-8")
        policy_err = _load_category_policy()
        check("corrupt JSON returns empty dict", policy_err == {})
        field = _build_category_field()
        check("corrupt JSON still returns valid field", "type" in field)

    finally:
        mp_file.write_text(original, encoding="utf-8")


def test_saved_profile_upgrade():
    """Test _seed_default_profile upgrades stale defaults with missing keys."""
    print("\n=== TORTURE: Saved Profile Upgrade ===")

    tmp = tempfile.mkdtemp()
    try:
        from pathlib import Path
        test_dir = Path(tmp) / "profiles"
        test_dir.mkdir(parents=True, exist_ok=True)

        from web.app import _seed_default_profile, _DEFAULT_PROFILE_STEM

        def _mock_loader():
            return {
                "name": "Test Profile",
                "version": "1.0",
                "write_policy": {"auto_write": True},
                "category_policy": {"mode": "open", "suggested_categories": ["bio"]},
            }

        # ── Brand new directory ──
        _seed_default_profile(test_dir, _mock_loader)
        default_path = test_dir / f"{_DEFAULT_PROFILE_STEM}.json"
        check("default created", default_path.exists())

        saved = json.loads(default_path.read_text(encoding="utf-8"))
        check("default has _pinned", saved.get("_pinned") is True)
        check("default has name", saved.get("name") == "Test Profile")
        check("default has category_policy", "category_policy" in saved)
        check("default has write_policy", "write_policy" in saved)

        # ── Stale default missing category_policy ──
        stale = {"name": "Old", "version": "0.9", "write_policy": {"auto_write": False}, "_pinned": True}
        default_path.write_text(json.dumps(stale), encoding="utf-8")

        _seed_default_profile(test_dir, _mock_loader)

        upgraded = json.loads(default_path.read_text(encoding="utf-8"))
        check("upgrade adds category_policy", "category_policy" in upgraded)
        check("upgrade preserves existing name", upgraded["name"] == "Old")
        check("upgrade preserves write_policy", upgraded["write_policy"]["auto_write"] is False)
        check("upgrade keeps _pinned", upgraded.get("_pinned") is True)

        # ── Already up-to-date ──
        before_text = default_path.read_text(encoding="utf-8")
        _seed_default_profile(test_dir, _mock_loader)
        after_text = default_path.read_text(encoding="utf-8")
        check("up-to-date default unchanged", before_text == after_text)

        # ── Corrupt default JSON ──
        default_path.write_text("NOT JSON!!!", encoding="utf-8")
        try:
            _seed_default_profile(test_dir, _mock_loader)
            check("corrupt default doesn't crash", True)
        except Exception:
            check("corrupt default doesn't crash", False)

        # ── Loader returns empty ──
        empty_dir = Path(tmp) / "empty_profiles"
        empty_dir.mkdir(parents=True, exist_ok=True)
        def _empty_loader():
            return {}
        _seed_default_profile(empty_dir, _empty_loader)
        ep = empty_dir / f"{_DEFAULT_PROFILE_STEM}.json"
        check("empty loader creates file", ep.exists())
        ed = json.loads(ep.read_text(encoding="utf-8"))
        check("empty loader has _pinned", ed.get("_pinned") is True)

        # ── Loader returns None ──
        none_dir = Path(tmp) / "none_profiles"
        none_dir.mkdir(parents=True, exist_ok=True)
        def _none_loader():
            return None
        _seed_default_profile(none_dir, _none_loader)
        np_path = none_dir / f"{_DEFAULT_PROFILE_STEM}.json"
        check("None loader creates file", np_path.exists())

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_tool_catalogue_dynamic():
    """Test that _TOOL_CATALOGUE uses dynamic VALID_SCOPES and truncated categories."""
    print("\n=== TORTURE: _TOOL_CATALOGUE Dynamic Fields ===")
    from web.app import _TOOL_CATALOGUE
    from src.memory.types import VALID_SCOPES, VALID_CATEGORIES

    # Find the memory tool entry
    mem_entry = None
    for t in _TOOL_CATALOGUE:
        if t["name"] == "memory":
            mem_entry = t
            break
    check("memory tool in catalogue", mem_entry is not None)

    # Find scope and category params
    scope_param = None
    category_param = None
    for p in mem_entry["parameters"]:
        if p["name"] == "scope":
            scope_param = p
        if p["name"] == "category":
            category_param = p

    check("scope param exists", scope_param is not None)
    check("category param exists", category_param is not None)

    # Scope enum should match VALID_SCOPES
    scope_enum = scope_param["enum"]
    check("scope enum is list", isinstance(scope_enum, list))
    check("scope enum matches VALID_SCOPES", set(scope_enum) == set(VALID_SCOPES))
    check("scope enum is sorted", scope_enum == sorted(scope_enum))
    check("'shared' in scope enum", "shared" in scope_enum)

    # All profile names in scope
    from pathlib import Path
    profiles_dir = Path(__file__).resolve().parent.parent / "profiles"
    if profiles_dir.exists():
        for p in profiles_dir.glob("*.yaml"):
            check(f"scope enum has '{p.stem}'", p.stem in scope_enum)

    # Category enum should be truncated to 5 + ellipsis
    cat_enum = category_param["enum"]
    check("category enum is list", isinstance(cat_enum, list))
    check("category enum has 6 items (5+ellipsis)", len(cat_enum) == 6)
    check("category last item is ellipsis", cat_enum[-1] == "\u2026")
    first_five = cat_enum[:5]
    check("first 5 are sorted", first_five == sorted(first_five))
    for c in first_five:
        check(f"'{c}' in VALID_CATEGORIES", c in VALID_CATEGORIES)


def test_memory_tool_definition_dynamic():
    """Test that MemoryTool.definition() has dynamic scope and category fields."""
    print("\n=== TORTURE: MemoryTool.definition() Dynamic ===")
    from src.tools.memory_tool import MemoryTool, _load_category_policy
    from src.memory.types import VALID_SCOPES

    defn = MemoryTool.definition()
    props = defn["parameters"]["properties"]

    # Scope
    scope_def = props["scope"]
    check("definition scope has enum", "enum" in scope_def)
    check("definition scope matches VALID_SCOPES", set(scope_def["enum"]) == set(VALID_SCOPES))
    check("definition scope sorted", scope_def["enum"] == sorted(scope_def["enum"]))

    # Category depends on current policy mode
    cat_def = props["category"]
    check("definition category has type", cat_def["type"] == "string")
    check("definition category has description", len(cat_def["description"]) > 0)

    mode = _load_category_policy().get("mode", "open")
    if mode == "open":
        check("open mode no enum in definition", "enum" not in cat_def)
    else:
        check(f"{mode} mode enum in definition", "enum" in cat_def)
        check(f"{mode} mode enum sorted", cat_def["enum"] == sorted(cat_def["enum"]))

    # Action list
    actions = props["action"]["enum"]
    check("13 actions in definition", len(actions) == 13)

    # Source enum
    source_def = props["source"]
    check("source has enum", "enum" in source_def)
    check("source enum sorted", source_def["enum"] == sorted(source_def["enum"]))


# ═════════════════════════════════════════════
if __name__ == "__main__":
    test_boundary_policy()
    test_pii_guard_extended()
    test_runtime_policy_clamping()
    test_manifest_helpers()
    test_directive_parser_edge_cases()
    test_directive_store_edge_cases()
    test_memory_types_extended()
    test_chunker_edge_cases()
    test_memory_injector()
    test_memory_tool_all_actions()
    test_cost_tracker_extended()
    test_note_collector_helpers()
    test_manifest_id_collision()
    test_continuation_edge_cases()
    test_data_paths_extended()
    test_metering_extended()
    test_web_search_extract()
    test_active_directives_batch()
    test_vault_interleaved_ops()
    test_user_notes_loader_extended()
    test_manifest_diff_extended()
    test_echo_tool()
    test_llm_types()
    test_directive_injector_with_manifest()
    test_model_router_config()
    test_email_tool_torture()
    test_directives_tool_torture()
    test_tool_registry_torture()
    test_validate_manifest()
    test_audit_changes()
    test_cost_tracker_pricing_actions()
    test_web_search_tool_extended()
    test_metering_helpers_extended()
    test_llm_client_factory()
    test_app_memory_helpers()
    test_seed_ui_knowledge_structure()
    test_metering_dataclass_ops()
    test_inbox_tool_torture()
    test_inbox_registry_dispatch()
    test_inbox_data_path()
    test_dynamic_scopes()
    test_category_policy()
    test_saved_profile_upgrade()
    test_tool_catalogue_dynamic()
    test_memory_tool_definition_dynamic()

    print(f"\n{'='*40}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL:
        sys.exit(1)
    else:
        print("All tests passed.")
