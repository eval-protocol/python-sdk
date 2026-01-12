"""
Benchmark test for CLI startup time.

This test ensures the CLI startup time stays under the target threshold.
Run with: pytest tests/test_cli_startup_benchmark.py -v
"""

import subprocess
import sys
import time

import pytest

# Target: CLI should start in under 1.0 second
CLI_STARTUP_TARGET_SECONDS = 1.0

# Number of runs to average (first run may be slower due to cold cache)
NUM_RUNS = 3


def measure_cli_startup_time() -> float:
    """Measure CLI --help startup time in seconds."""
    start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-m", "eval_protocol.cli", "--help"],
        capture_output=True,
        text=True,
        env={**dict(__import__("os").environ), "FIREWORKS_API_KEY": "benchmark-test-key"},
    )
    elapsed = time.perf_counter() - start

    # Ensure the command succeeded
    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    return elapsed


@pytest.mark.benchmark
def test_cli_startup_time():
    """Test that CLI startup time is under the target threshold."""
    times = []

    for i in range(NUM_RUNS):
        elapsed = measure_cli_startup_time()
        times.append(elapsed)
        print(f"  Run {i + 1}: {elapsed:.3f}s")

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"\n  Average: {avg_time:.3f}s")
    print(f"  Min: {min_time:.3f}s")
    print(f"  Max: {max_time:.3f}s")
    print(f"  Target: {CLI_STARTUP_TARGET_SECONDS}s")

    # Use the best time (min) as some CI environments have variable overhead
    assert min_time < CLI_STARTUP_TARGET_SECONDS, (
        f"CLI startup time ({min_time:.3f}s) exceeds target ({CLI_STARTUP_TARGET_SECONDS}s). "
        f"Check for import-time side effects or eager module loading."
    )


@pytest.mark.benchmark
def test_package_import_time():
    """Test that importing eval_protocol package is fast (lazy loading check)."""
    # Use subprocess to get a clean import measurement
    code = """
import time
start = time.perf_counter()
import eval_protocol
elapsed = time.perf_counter() - start
print(f"{elapsed:.6f}")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Import failed: {result.stderr}"

    import_time = float(result.stdout.strip())
    print(f"\n  Package import time: {import_time * 1000:.1f}ms")

    # Package import should be very fast with lazy loading (< 50ms)
    assert import_time < 0.05, (
        f"Package import time ({import_time * 1000:.1f}ms) is too slow. "
        f"Check that __init__.py uses lazy loading correctly."
    )


if __name__ == "__main__":
    print("=== CLI Startup Benchmark ===\n")

    print("Testing CLI startup time...")
    times = []
    for i in range(NUM_RUNS):
        elapsed = measure_cli_startup_time()
        times.append(elapsed)
        print(f"  Run {i + 1}: {elapsed:.3f}s")

    avg_time = sum(times) / len(times)
    min_time = min(times)

    print(f"\n  Average: {avg_time:.3f}s")
    print(f"  Best: {min_time:.3f}s")
    print(f"  Target: {CLI_STARTUP_TARGET_SECONDS}s")

    if min_time < CLI_STARTUP_TARGET_SECONDS:
        print(f"\n✓ PASS: CLI startup ({min_time:.3f}s) is under target ({CLI_STARTUP_TARGET_SECONDS}s)")
    else:
        print(f"\n✗ FAIL: CLI startup ({min_time:.3f}s) exceeds target ({CLI_STARTUP_TARGET_SECONDS}s)")
        sys.exit(1)
