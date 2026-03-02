"""Offline tests for tools and runtime policy.

Run from project root:
    python -m tests.test_tools

No LLM connection required â€” exercises everything except the live chat call.
"""

import json
import os
import sys
import tempfile
import shutil

# â”€â”€ ensure project root is on path â”€â”€
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.tools.echo import EchoTool
from src.tools.continuation_update import ContinuationUpdateTool
from src.tools.email_tool import EmailTool
from src.runtime_policy import RuntimePolicy

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


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. Echo tool
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def test_echo():
    print("\n=== Echo Tool ===")
    tool = EchoTool()

    defn = tool.definition()
    check("echo definition exists", defn["name"] == "echo")

    result = tool.execute({"message": "hello world"})
    check("echo returns message", result == "hello world", f"got: {result!r}")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. Continuation Update tool
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def test_continuation_update():
    print("\n=== Continuation Update Tool ===")
    import src.data_paths as dp
    orig_root = dp.DATA_ROOT
    tmp_dir = tempfile.mkdtemp()
    dp.DATA_ROOT = tmp_dir

    tool = ContinuationUpdateTool()

    try:
        # Append mode
        r1 = tool.execute({
            "profile": "orion", "mode": "append", "content": "Started task A."
        })
        check("append succeeds", "Appended" in r1, r1)

        # Append again
        r2 = tool.execute({
            "profile": "orion", "mode": "append", "content": "Finished task A."
        })
        check("second append succeeds", "Appended" in r2, r2)

        # Replace section (new)
        r3 = tool.execute({
            "profile": "orion", "mode": "replace_section",
            "section": "Status", "content": "In progress."
        })
        check("replace_section adds new", "Added" in r3, r3)

        # Replace section (update)
        r4 = tool.execute({
            "profile": "orion", "mode": "replace_section",
            "section": "Status", "content": "Complete."
        })
        check("replace_section updates", "Replaced" in r4, r4)

        # Verify file content
        path = os.path.join(tmp_dir, "orion", "continuation.md")
        with open(path, "r") as f:
            content = f.read()
        check("has 'Started task A'", "Started task A" in content)
        check("has 'Finished task A'", "Finished task A" in content)
        check("Status says Complete", "Complete." in content)
        check("old status replaced", content.count("## Status") == 1,
              f"found {content.count('## Status')} occurrences")

        # Different profile -> separate file
        r5 = tool.execute({
            "profile": "elysia", "mode": "append", "content": "Hello from elysia."
        })
        elysia_path = os.path.join(tmp_dir, "elysia", "continuation.md")
        check("elysia file created", os.path.exists(elysia_path))

        # Path traversal blocked
        r6 = tool.execute({
            "profile": "../escape", "mode": "append", "content": "bad"
        })
        check("path traversal blocked", "Error" in r6, r6)

        # Missing content
        r7 = tool.execute({
            "profile": "orion", "mode": "append", "content": "  "
        })
        check("empty content blocked", "Error" in r7, r7)

    finally:
        dp.DATA_ROOT = orig_root
        shutil.rmtree(tmp_dir, ignore_errors=True)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. Runtime Policy
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def test_policy():
    print("\n=== Runtime Policy ===")
    import time

    p = RuntimePolicy(max_iterations=3, max_wall_time_seconds=0.5)
    check("iter 0 ok", p.check(0, time.time()) is None)
    check("iter 2 ok", p.check(2, time.time()) is None)
    check("iter 3 blocked", "Max iterations" in (p.check(3, time.time()) or ""))

    old = time.time() - 1.0  # 1 second ago
    check("wall time exceeded", "Wall-time" in (p.check(0, old) or ""))

    stasis = RuntimePolicy(stasis_mode=True)
    check("stasis_mode flag", stasis.stasis_mode is True)

    stop_policy = RuntimePolicy(tool_failure_mode="stop")
    check("tool_failure_mode=stop", stop_policy.tool_failure_mode == "stop")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. Email tool
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def test_email():
    print("\n=== Email Tool ===")
    tool = EmailTool()

    # Definition
    defn = tool.definition()
    check("email definition name", defn["name"] == "email")
    check("email has parameters", "parameters" in defn)
    actions = defn["parameters"]["properties"]["action"]["enum"]
    check("email has send action", "send" in actions)
    check("email has status action", "status" in actions)
    check("email has accounts action", "accounts" in actions)

    # Accounts (should return list even if empty)
    r = json.loads(tool.execute({"action": "accounts"}))
    check("accounts returns list", "accounts" in r)
    check("accounts is list type", isinstance(r["accounts"], list))

    # Status check
    r = json.loads(tool.execute({"action": "status"}))
    check("status has accounts_configured", "accounts_configured" in r)

    # Send â€” missing fields
    r = json.loads(tool.execute({"action": "send"}))
    check("send w/o subject â†’ error", "error" in r)

    r = json.loads(tool.execute({"action": "send", "subject": "test"}))
    check("send w/o body â†’ error", "error" in r)

    r = json.loads(tool.execute({"action": "send", "subject": "test", "body": "hello"}))
    check("send w/o recipients â†’ error", "error" in r)

    # Send â€” invalid email address
    r = json.loads(tool.execute({
        "action": "send", "subject": "test", "body": "hello",
        "recipients": ["not-an-email"],
    }))
    check("send invalid email â†’ error", "error" in r)

    # Unknown action
    r = json.loads(tool.execute({"action": "explode"}))
    check("unknown action â†’ error", "error" in r)

    # Execute with agent_name parameter
    r = json.loads(tool.execute({"action": "accounts"}, agent_name="astraea"))
    check("accounts with agent_name", "accounts" in r)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Run all
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == "__main__":
    test_echo()
    test_continuation_update()
    test_policy()
    test_email()

    print(f"\n{'='*40}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL:
        sys.exit(1)
    else:
        print("All tests passed.")

