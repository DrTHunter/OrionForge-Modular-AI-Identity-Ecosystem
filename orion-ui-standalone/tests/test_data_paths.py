"""Torture tests for src/data_paths.py — canonical data directory layout.

Run from project root:
    python -m tests.test_data_paths

Exercises every path function, auto-creation, isolation, and edge cases.
"""

import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.data_paths as dp

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


# ─────────────────────────────────────────────
# 1. Directory getters create dirs
# ─────────────────────────────────────────────
def test_directory_creation():
    print("\n=== Data Paths — Directory Creation ===")
    orig_root = dp.DATA_ROOT
    tmp = tempfile.mkdtemp()
    dp.DATA_ROOT = tmp

    try:
        # profile_dir should auto-create
        pd = dp.profile_dir("test_agent")
        check("profile_dir returns path", pd.endswith(os.path.join("test_agent")))
        check("profile_dir exists", os.path.isdir(pd))

        # memory_dir
        md = dp.memory_dir()
        check("memory_dir exists", os.path.isdir(md))
        check("memory_dir ends with 'memory'", md.endswith("memory"))

        # faiss_dir
        fd = dp.faiss_dir()
        check("faiss_dir exists", os.path.isdir(fd))
        check("faiss_dir nested under memory", "memory" in fd and fd.endswith("faiss"))

        # shared_dir
        sd = dp.shared_dir()
        check("shared_dir exists", os.path.isdir(sd))
        check("shared_dir ends with 'shared'", sd.endswith("shared"))

        # calling again doesn't error (idempotent)
        pd2 = dp.profile_dir("test_agent")
        check("profile_dir idempotent", pd == pd2)

    finally:
        dp.DATA_ROOT = orig_root
        shutil.rmtree(tmp, ignore_errors=True)


# ─────────────────────────────────────────────
# 2. File path getters
# ─────────────────────────────────────────────
def test_file_paths():
    print("\n=== Data Paths — File Paths ===")
    orig_root = dp.DATA_ROOT
    tmp = tempfile.mkdtemp()
    dp.DATA_ROOT = tmp

    try:
        # Per-profile paths
        sp = dp.state_path("orion")
        check("state_path ends with state.json", sp.endswith("state.json"))
        check("state_path contains profile", "orion" in sp)

        jp = dp.journal_path("orion")
        check("journal_path ends with journal.jsonl", jp.endswith("journal.jsonl"))

        smp = dp.summary_path("orion")
        check("summary_path ends with summary.md", smp.endswith("summary.md"))

        cp = dp.continuation_path("orion")
        check("continuation_path ends with continuation.md", cp.endswith("continuation.md"))

        np = dp.narrative_path("orion")
        check("narrative_path ends with narrative.md", np.endswith("narrative.md"))

        # Shared paths
        vp = dp.vault_path()
        check("vault_path ends with vault.jsonl", vp.endswith("vault.jsonl"))
        check("vault_path parent exists", os.path.isdir(os.path.dirname(vp)))

        bep = dp.boundary_events_path()
        check("boundary_events_path ends .jsonl", bep.endswith("boundary_events.jsonl"))

        clp = dp.change_log_path()
        check("change_log_path ends .jsonl", clp.endswith("change_log.jsonl"))

        hjp = dp.human_journal_path()
        check("human_journal_path ends .md", hjp.endswith("journal.md"))

        trp = dp.tool_requests_path()
        check("tool_requests_path ends .md", trp.endswith("tool_requests.md"))

    finally:
        dp.DATA_ROOT = orig_root
        shutil.rmtree(tmp, ignore_errors=True)


# ─────────────────────────────────────────────
# 3. Multiple profiles don't clobber
# ─────────────────────────────────────────────
def test_profile_isolation():
    print("\n=== Data Paths — Profile Isolation ===")
    orig_root = dp.DATA_ROOT
    tmp = tempfile.mkdtemp()
    dp.DATA_ROOT = tmp

    try:
        p1 = dp.profile_dir("alpha")
        p2 = dp.profile_dir("bravo")
        check("different profiles ≠ same dir", p1 != p2)
        check("both exist", os.path.isdir(p1) and os.path.isdir(p2))

        s1 = dp.state_path("alpha")
        s2 = dp.state_path("bravo")
        check("state paths differ", s1 != s2)

    finally:
        dp.DATA_ROOT = orig_root
        shutil.rmtree(tmp, ignore_errors=True)


# ─────────────────────────────────────────────
# 4. Edge cases — weird profile names
# ─────────────────────────────────────────────
def test_edge_cases():
    print("\n=== Data Paths — Edge Cases ===")
    orig_root = dp.DATA_ROOT
    tmp = tempfile.mkdtemp()
    dp.DATA_ROOT = tmp

    try:
        # Profile with spaces
        pd = dp.profile_dir("my agent")
        check("spaces in profile name", os.path.isdir(pd))

        # Profile with dots
        pd2 = dp.profile_dir("agent.v2")
        check("dots in profile name", os.path.isdir(pd2))

        # Empty-string profile (degenerate — should still work)
        pd3 = dp.profile_dir("")
        check("empty profile name returns path", pd3 is not None)

        # Ensure _ensure helper is idempotent
        for _ in range(5):
            dp.profile_dir("repeat_test")
        check("repeated creation idempotent", os.path.isdir(dp.profile_dir("repeat_test")))

    finally:
        dp.DATA_ROOT = orig_root
        shutil.rmtree(tmp, ignore_errors=True)


# ─────────────────────────────────────────────
# 5. Path normalization
# ─────────────────────────────────────────────
def test_path_normalization():
    print("\n=== Data Paths — Normalization ===")
    orig_root = dp.DATA_ROOT
    tmp = tempfile.mkdtemp()
    dp.DATA_ROOT = tmp

    try:
        p = dp.profile_dir("test")
        check("no double separators", os.sep + os.sep not in p)
        sp = dp.state_path("test")
        check("state_path is under profile_dir", sp.startswith(p))

    finally:
        dp.DATA_ROOT = orig_root
        shutil.rmtree(tmp, ignore_errors=True)


# ─────────────────────────────────────────────
if __name__ == "__main__":
    test_directory_creation()
    test_file_paths()
    test_profile_isolation()
    test_edge_cases()
    test_path_normalization()

    print(f"\n{'='*40}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL:
        sys.exit(1)
    else:
        print("All tests passed.")
