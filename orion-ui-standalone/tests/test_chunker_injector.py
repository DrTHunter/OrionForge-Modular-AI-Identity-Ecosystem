"""Torture tests for src/memory/chunker.py and src/memory/injector.py.

Run from project root:
    python -m tests.test_chunker_injector

No LLM / FAISS required — exercises pure chunking logic, merge/split,
and the injector formatting helpers.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.memory.chunker import SemanticChunker, chunk_soul_script

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
# SEMANTIC CHUNKER
# ═════════════════════════════════════════════

def test_basic_chunking():
    print("\n=== SemanticChunker — basic chunking ===")
    chunker = SemanticChunker(min_chunk_size=10, max_chunk_size=5000)

    doc = """# Main Title

Some intro text here.

### Section One

Content of section one with some details.
More detail here.

### Section Two

Content of section two.

### Section Three

Section three material.
"""
    chunks = chunker.chunk_by_headers(doc, "doc1", "Test Doc")
    check("chunks created", len(chunks) > 0, f"got {len(chunks)}")

    # Each chunk has required keys
    for i, c in enumerate(chunks):
        check(f"chunk {i} has 'text'", "text" in c)
        check(f"chunk {i} has 'metadata'", "metadata" in c)
        check(f"chunk {i} has document_id", c["metadata"]["document_id"] == "doc1")
        check(f"chunk {i} has chunk_index", "chunk_index" in c["metadata"])
        check(f"chunk {i} has total_chunks", "total_chunks" in c["metadata"])

    # Titles captured
    titles = [c["metadata"]["section_title"] for c in chunks]
    check("captures Section One", any("Section One" in t for t in titles))
    check("captures Section Two", any("Section Two" in t for t in titles))


def test_no_headers():
    print("\n=== SemanticChunker — no headers ===")
    chunker = SemanticChunker(min_chunk_size=10, max_chunk_size=5000)

    doc = "Just a plain paragraph of text without any markdown headers."
    chunks = chunker.chunk_by_headers(doc, "plain", "Plain Doc")
    check("single chunk for headerless", len(chunks) == 1, f"got {len(chunks)}")
    check("chunk text matches", doc in chunks[0]["text"])
    check("section_title = doc title", chunks[0]["metadata"]["section_title"] == "Plain Doc")


def test_empty_document():
    print("\n=== SemanticChunker — empty document ===")
    chunker = SemanticChunker()
    chunks = chunker.chunk_by_headers("", "empty", "Empty")
    # Empty or single empty chunk is acceptable
    check("empty doc → 0 or 1 chunk", len(chunks) <= 1, f"got {len(chunks)}")


def test_merge_small_chunks():
    print("\n=== SemanticChunker — merge small chunks ===")
    # Use high min_chunk_size to force merging
    chunker = SemanticChunker(min_chunk_size=500, max_chunk_size=10000)

    doc = """### A

Short.

### B

Also short.

### C

And this is short too.
"""
    chunks = chunker.chunk_by_headers(doc, "merge_test", "Merge Test")
    # With high min, some chunks should be merged
    check("some merging happened", len(chunks) < 3, f"got {len(chunks)} chunks")

    # Check merged flag
    merged_any = any(c["metadata"].get("merged") for c in chunks)
    check("merged flag set", merged_any)


def test_split_large_chunks():
    print("\n=== SemanticChunker — split large chunks ===")
    # Use very low max_chunk_size to force splitting
    chunker = SemanticChunker(min_chunk_size=10, max_chunk_size=200)

    # Create a large section
    large_section = "\n\n".join([f"Paragraph {i}: " + "x" * 100 for i in range(10)])
    doc = f"### Big Section\n\n{large_section}"

    chunks = chunker.chunk_by_headers(doc, "split_test", "Split Test")
    check("splitting happened", len(chunks) > 1, f"got {len(chunks)} chunks")

    split_any = any(c["metadata"].get("split") for c in chunks)
    check("split flag set", split_any)


def test_metadata_propagation():
    print("\n=== SemanticChunker — metadata propagation ===")
    chunker = SemanticChunker(min_chunk_size=10, max_chunk_size=5000)

    doc = "### Test\n\nContent here."
    extra_meta = {"custom_key": "custom_value", "important": True}
    chunks = chunker.chunk_by_headers(doc, "meta_test", "Meta Test", metadata=extra_meta)

    check("custom metadata propagated",
          chunks[0]["metadata"].get("custom_key") == "custom_value")
    check("boolean metadata works",
          chunks[0]["metadata"].get("important") is True)


def test_section_path():
    print("\n=== SemanticChunker — section_path hierarchy ===")
    chunker = SemanticChunker(min_chunk_size=10, max_chunk_size=5000)

    doc = "Intro text.\n\n### My Section\n\nSection content."
    chunks = chunker.chunk_by_headers(doc, "path_test", "Root Title")

    # Intro chunk → path = Root Title
    intro = [c for c in chunks if "My Section" not in c["metadata"]["section_title"]]
    if intro:
        check("intro path = doc title", intro[0]["metadata"]["section_path"] == "Root Title")

    # Section chunk → path = Root Title > My Section
    section = [c for c in chunks if "My Section" in c["metadata"]["section_title"]]
    if section:
        check("section path hierarchical",
              "Root Title > My Section" in section[0]["metadata"]["section_path"])


def test_chunk_vault_memory():
    print("\n=== SemanticChunker — chunk_vault_memory ===")
    chunker = SemanticChunker()

    memory = {
        "id": "mem_001",
        "text": "The user prefers dark mode in all applications.",
        "metadata": {
            "scope": "shared",
            "category": "preference",
            "tier": "core",
        },
    }
    chunks = chunker.chunk_vault_memory(memory)
    check("single chunk", len(chunks) == 1)
    check("text preserved", chunks[0]["text"] == memory["text"])
    check("has document_id", chunks[0]["metadata"]["document_id"] == "mem_001")
    check("scope in path", "shared" in chunks[0]["metadata"]["section_path"])


def test_chunk_soul_script_helper():
    print("\n=== chunk_soul_script convenience function ===")

    script = """### Identity Core

I am Astraea, a reflective intelligence.

### Behavioral Principles

I prioritize honesty and careful reasoning.

### Communication Style

I speak with warmth and precision.
"""
    chunks = chunk_soul_script(script, "note_123", "Soul Script", "✨")
    check("multiple chunks", len(chunks) >= 2, f"got {len(chunks)}")
    check("is_canon set", all(c["metadata"].get("is_canon") for c in chunks))
    check("immutable set", all(c["metadata"].get("immutable") for c in chunks))
    check("emoji propagated", chunks[0]["metadata"].get("emoji") == "✨")

    # Headers captured in titles
    titles = [c["metadata"]["section_title"] for c in chunks]
    check("Identity Core in titles", any("Identity Core" in t for t in titles))
    check("Behavioral Principles in titles", any("Behavioral" in t for t in titles))


def test_chunk_indices():
    print("\n=== SemanticChunker — chunk indices ===")
    chunker = SemanticChunker(min_chunk_size=10, max_chunk_size=5000)

    doc = "### A\n\nContent A.\n\n### B\n\nContent B.\n\n### C\n\nContent C."
    chunks = chunker.chunk_by_headers(doc, "idx_test", "Idx Test")

    for i, c in enumerate(chunks):
        check(f"chunk {i} index = {i}", c["metadata"]["chunk_index"] == i)
        check(f"chunk {i} total = {len(chunks)}",
              c["metadata"]["total_chunks"] == len(chunks))


def test_char_count_accuracy():
    print("\n=== SemanticChunker — char_count ===")
    chunker = SemanticChunker(min_chunk_size=10, max_chunk_size=5000)

    doc = "### Hello\n\nThis is exactly some text."
    chunks = chunker.chunk_by_headers(doc, "char_test", "Char Test")

    for c in chunks:
        actual_len = len(c["text"])
        reported = c["metadata"]["char_count"]
        check(f"char_count accurate ({actual_len})", actual_len == reported,
              f"text len={actual_len}, metadata={reported}")


# ═════════════════════════════════════════════
# MEMORY INJECTOR HELPERS
# ═════════════════════════════════════════════

def test_injector_norm_scope():
    print("\n=== Injector — _norm_scope ===")
    from src.memory.injector import _norm_scope

    check("string passes through", _norm_scope("shared") == "shared")
    check("single-element list", _norm_scope(["shared"]) == "shared")
    check("multi-element list → None", _norm_scope(["shared", "callum"]) is None)
    check("empty string → empty", _norm_scope("") is None or _norm_scope("") == "")
    check("None → None", _norm_scope(None) is None)


def test_injector_dict_to_display():
    print("\n=== Injector — _dict_to_display ===")
    from src.memory.injector import _dict_to_display

    d = {"text": "hello", "scope": "shared", "category": "fact",
         "tags": ["tag1"], "score": 0.85}
    result = _dict_to_display(d)
    check("text preserved", result["text"] == "hello")
    check("scope preserved", result["scope"] == "shared")
    check("score preserved", result["score"] == 0.85)

    # Missing keys → defaults
    empty = _dict_to_display({})
    check("missing text → empty", empty["text"] == "")
    check("missing category → other", empty["category"] == "other")
    check("missing tags → []", empty["tags"] == [])
    check("missing score → 0", empty["score"] == 0.0)


def test_injector_mem_to_display():
    print("\n=== Injector — _mem_to_display ===")
    from src.memory.injector import _mem_to_display
    from src.memory.types import Memory

    m = Memory(
        id="mem1", text="Test memory", scope="shared",
        category="fact", tags=["tag1", "tag2"],
    )
    result = _mem_to_display(m)
    check("text", result["text"] == "Test memory")
    check("scope", result["scope"] == "shared")
    check("category", result["category"] == "fact")
    check("tags", result["tags"] == ["tag1", "tag2"])

    # Memory with no tags
    m2 = Memory(id="mem2", text="tagless", scope="shared", category="fact")
    r2 = _mem_to_display(m2)
    check("missing tags → []", r2["tags"] == [])


# ═════════════════════════════════════════════
if __name__ == "__main__":
    test_basic_chunking()
    test_no_headers()
    test_empty_document()
    test_merge_small_chunks()
    test_split_large_chunks()
    test_metadata_propagation()
    test_section_path()
    test_chunk_vault_memory()
    test_chunk_soul_script_helper()
    test_chunk_indices()
    test_char_count_accuracy()

    test_injector_norm_scope()
    test_injector_dict_to_display()
    test_injector_mem_to_display()

    print(f"\n{'='*40}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL:
        sys.exit(1)
    else:
        print("All tests passed.")
