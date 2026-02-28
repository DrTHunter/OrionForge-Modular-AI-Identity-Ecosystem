"""Master test runner — runs ALL test suites and reports totals.

Run from project root:
    python -m tests.run_all

Executes every test module in sequence, consolidating results.
"""

import subprocess
import sys
import os
import time

# All test modules in dependency order
TEST_MODULES = [
    "tests.test_data_paths",
    "tests.test_memory",
    "tests.test_directives",
    "tests.test_governance",
    "tests.test_tools",
    "tests.test_metering",
    "tests.test_registry_and_tools",
    "tests.test_chunker_injector",
    "tests.test_storage_and_llm",
    "tests.test_stress",
    "tests.test_torture",
]


def main():
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    total_pass = 0
    total_fail = 0
    failed_suites = []

    print("=" * 60)
    print("  OrionForge — Full Test Suite")
    print("=" * 60)
    t_start = time.time()

    for module in TEST_MODULES:
        print(f"\n{'─' * 60}")
        print(f"  Running: {module}")
        print(f"{'─' * 60}")

        result = subprocess.run(
            [sys.executable, "-m", module],
            capture_output=False,
            text=True,
        )

        if result.returncode != 0:
            failed_suites.append(module)
            total_fail += 1
        else:
            total_pass += 1

    elapsed = time.time() - t_start

    print(f"\n{'=' * 60}")
    print(f"  FINAL RESULTS")
    print(f"{'=' * 60}")
    print(f"  Suites passed:  {total_pass}/{len(TEST_MODULES)}")
    print(f"  Suites failed:  {len(failed_suites)}/{len(TEST_MODULES)}")
    print(f"  Total time:     {elapsed:.1f}s")

    if failed_suites:
        print(f"\n  FAILED SUITES:")
        for s in failed_suites:
            print(f"    ✗ {s}")
        print()
        sys.exit(1)
    else:
        print(f"\n  ✓ ALL {len(TEST_MODULES)} SUITES PASSED")
        print()


if __name__ == "__main__":
    main()
