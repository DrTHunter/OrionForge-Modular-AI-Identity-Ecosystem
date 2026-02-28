"""Torture tests for src/storage/user_notes_loader.py and src/llm_client/base.py.

Run from project root:
    python -m tests.test_storage_and_llm

Exercises HTML stripping, note loading, and LLMResponse dataclass.
"""

import json
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.storage.user_notes_loader import strip_html, load_json_user_notes
from src.llm_client.base import LLMResponse

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
# strip_html
# ═════════════════════════════════════════════

def test_strip_html_basic():
    print("\n=== strip_html — basic ===")
    check("empty string", strip_html("") == "")
    check("None-ish", strip_html("") == "")
    check("plain text passthrough", strip_html("hello world") == "hello world")


def test_strip_html_tags():
    print("\n=== strip_html — tag removal ===")
    r = strip_html("<p>Hello</p>")
    check("<p> removed", "<p>" not in r and "Hello" in r)

    r2 = strip_html("<b>bold</b> and <i>italic</i>")
    check("inline tags removed", "<b>" not in r2 and "bold" in r2 and "italic" in r2)

    r3 = strip_html('<a href="http://example.com">link</a>')
    check("anchor tag removed", "<a" not in r3 and "link" in r3)

    r4 = strip_html("<div><span>nested</span></div>")
    check("nested tags removed", "nested" in r4 and "<" not in r4)


def test_strip_html_newlines():
    print("\n=== strip_html — newline insertion ===")
    r = strip_html("<p>One</p><p>Two</p>")
    check("</p> creates newlines", "\n" in r)

    r2 = strip_html("line1<br>line2")
    check("<br> creates newline", "\n" in r2)

    r3 = strip_html("<h2>Title</h2><p>Body</p>")
    check("</h2> creates newline", "\n" in r3)


def test_strip_html_entities():
    print("\n=== strip_html — HTML entities ===")
    check("&nbsp;", strip_html("hello&nbsp;world") == "hello world")
    check("&lt;", "<" in strip_html("&lt;tag&gt;"))
    check("&gt;", ">" in strip_html("&lt;tag&gt;"))
    check("&amp;", "&" in strip_html("A &amp; B"))
    check("&quot;", '"' in strip_html('say &quot;hi&quot;'))


def test_strip_html_whitespace():
    print("\n=== strip_html — whitespace cleanup ===")
    r = strip_html("<p>Hello</p>\n\n\n\n\n<p>World</p>")
    # Should collapse excessive newlines
    check("excessive newlines collapsed", "\n\n\n" not in r)
    check("content preserved", "Hello" in r and "World" in r)


def test_strip_html_complex():
    print("\n=== strip_html — complex HTML ===")
    html = """
    <div class="note">
        <h2>My Note</h2>
        <p>First paragraph with <b>bold</b> and <i>italic</i>.</p>
        <ul>
            <li>Item 1</li>
            <li>Item 2</li>
        </ul>
        <p>Final <a href="#">link</a>.</p>
    </div>
    """
    result = strip_html(html)
    check("no HTML tags remain", "<" not in result and ">" not in result.replace(">", ""))
    check("text preserved", "My Note" in result)
    check("bold text preserved", "bold" in result)
    check("link text preserved", "link" in result)
    check("items preserved", "Item 1" in result)


# ═════════════════════════════════════════════
# load_json_user_notes
# ═════════════════════════════════════════════

def test_load_notes_empty():
    print("\n=== load_json_user_notes — empty/missing ===")
    tmp = tempfile.mkdtemp()
    try:
        # No index file
        result = load_json_user_notes(tmp)
        check("no index → empty", result == "")

        # Empty index
        with open(os.path.join(tmp, "index.json"), "w") as f:
            json.dump([], f)
        result2 = load_json_user_notes(tmp)
        check("empty index → empty", result2 == "")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_notes_with_content():
    print("\n=== load_json_user_notes — with content ===")
    tmp = tempfile.mkdtemp()
    try:
        # Create index
        index = [
            {"id": "note_001", "title": "Test Note"},
            {"id": "note_002", "title": "Trashed", "trashed": True},
            {"id": "note_003"},  # Missing file
        ]
        with open(os.path.join(tmp, "index.json"), "w") as f:
            json.dump(index, f)

        # Create note file
        note = {
            "title": "Test Note",
            "emoji": "📝",
            "content_html": "<p>Hello from the note.</p>",
        }
        with open(os.path.join(tmp, "note_001.json"), "w") as f:
            json.dump(note, f)

        result = load_json_user_notes(tmp)
        check("note content loaded", "Hello from the note" in result)
        check("title in output", "Test Note" in result)
        check("emoji in output", "📝" in result)
        check("trashed note excluded", "Trashed" not in result)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_notes_corrupt_json():
    print("\n=== load_json_user_notes — corrupt JSON ===")
    tmp = tempfile.mkdtemp()
    try:
        # Corrupt index
        with open(os.path.join(tmp, "index.json"), "w") as f:
            f.write("{{{bad json")
        result = load_json_user_notes(tmp)
        check("corrupt index → empty", result == "")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_notes_missing_id():
    print("\n=== load_json_user_notes — entries without id ===")
    tmp = tempfile.mkdtemp()
    try:
        index = [{"title": "No ID"}]
        with open(os.path.join(tmp, "index.json"), "w") as f:
            json.dump(index, f)
        result = load_json_user_notes(tmp)
        check("no id → skipped gracefully", result == "")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ═════════════════════════════════════════════
# LLMResponse dataclass
# ═════════════════════════════════════════════

def test_llm_response_defaults():
    print("\n=== LLMResponse — defaults ===")
    r = LLMResponse()
    check("content is None", r.content is None)
    check("tool_calls empty", r.tool_calls == [])
    check("model empty", r.model == "")
    check("usage None", r.usage is None)
    check("raw empty dict", r.raw == {})


def test_llm_response_fields():
    print("\n=== LLMResponse — field population ===")
    r = LLMResponse(
        content="Hello!",
        model="gpt-4",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        tool_calls=[{"call_id": "1", "tool": "echo", "arguments": {"message": "hi"}}],
        raw={"id": "chatcmpl-123"},
    )
    check("content set", r.content == "Hello!")
    check("model set", r.model == "gpt-4")
    check("usage set", r.usage["total_tokens"] == 15)
    check("tool_calls set", len(r.tool_calls) == 1)
    check("tool_calls tool name", r.tool_calls[0]["tool"] == "echo")
    check("raw preserved", r.raw["id"] == "chatcmpl-123")


def test_llm_response_mutability():
    print("\n=== LLMResponse — mutability ===")
    r = LLMResponse(content="first")
    r.content = "second"
    check("content mutable", r.content == "second")

    r.tool_calls.append({"call_id": "tc1", "tool": "test", "arguments": {}})
    check("tool_calls appendable", len(r.tool_calls) == 1)


def test_llm_response_isolation():
    print("\n=== LLMResponse — instance isolation ===")
    r1 = LLMResponse()
    r2 = LLMResponse()
    r1.tool_calls.append({"call_id": "tc1"})
    check("instances isolated", len(r2.tool_calls) == 0,
          f"r2.tool_calls has {len(r2.tool_calls)} items (expected 0)")


# ═════════════════════════════════════════════
if __name__ == "__main__":
    test_strip_html_basic()
    test_strip_html_tags()
    test_strip_html_newlines()
    test_strip_html_entities()
    test_strip_html_whitespace()
    test_strip_html_complex()

    test_load_notes_empty()
    test_load_notes_with_content()
    test_load_notes_corrupt_json()
    test_load_notes_missing_id()

    test_llm_response_defaults()
    test_llm_response_fields()
    test_llm_response_mutability()
    test_llm_response_isolation()

    print(f"\n{'='*40}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL:
        sys.exit(1)
    else:
        print("All tests passed.")
