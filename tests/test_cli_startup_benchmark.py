"""
Benchmark test for CLI startup time and evaluation_test import time.

These are smoke tests that run on schedule (not on every PR) to catch performance regressions.
Run manually with: RUN_BENCHMARK_TESTS=1 pytest tests/test_cli_startup_benchmark.py -v
"""

import os
import subprocess
import sys
import time

import pytest

# Skip benchmark tests unless explicitly enabled via environment variable
# This prevents flaky failures from blocking PRs
SKIP_BENCHMARK = os.environ.get("RUN_BENCHMARK_TESTS", "0") != "1"
SKIP_REASON = "Benchmark tests only run when RUN_BENCHMARK_TESTS=1 (scheduled smoke tests)"

# Target: CLI should start in under 1.5 seconds (CI runners are slower)
CLI_STARTUP_TARGET_SECONDS = 1.5

# Target: evaluation_test import should be under 10.0 seconds (CI runners can be very slow)
EVALUATION_TEST_IMPORT_TARGET_SECONDS = 10.0

# Number of runs to average (first run may be slower due to cold cache)
NUM_RUNS = 3


def measure_cli_startup_time() -> float:
    """Measure CLI --help startup time in seconds."""
    start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-m", "eval_protocol.cli", "--help"],
        capture_output=True,
        text=True,
        env={**dict(os.environ), "FIREWORKS_API_KEY": "benchmark-test-key"},
    )
    elapsed = time.perf_counter() - start

    # Ensure the command succeeded
    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    return elapsed


@pytest.mark.benchmark
@pytest.mark.skipif(SKIP_BENCHMARK, reason=SKIP_REASON)
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
        f"CLI startup time ({min_time:.3f}s) exceeds target ({CLI_STARTUP_TARGET_SECONDS}s)."
    )


@pytest.mark.benchmark
@pytest.mark.skipif(SKIP_BENCHMARK, reason=SKIP_REASON)
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

    # Package import should be very fast with lazy loading (< 100ms for CI)
    assert import_time < 0.1, f"Package import time ({import_time * 1000:.1f}ms) is too slow."


@pytest.mark.benchmark
@pytest.mark.skipif(SKIP_BENCHMARK, reason=SKIP_REASON)
def test_evaluation_test_import_time():
    """Test that importing evaluation_test decorator is under the target threshold."""
    code = """
import sys
import time
start = time.perf_counter()
from eval_protocol import evaluation_test
elapsed = time.perf_counter() - start
litellm_loaded = "litellm" in sys.modules
print(f"{elapsed:.6f}")
print(f"{litellm_loaded}")
"""
    times = []

    for i in range(NUM_RUNS):
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Import failed: {result.stderr}"

        lines = result.stdout.strip().split("\n")
        import_time = float(lines[0])
        litellm_loaded = lines[1] == "True"
        times.append(import_time)
        print(f"  Run {i + 1}: {import_time:.3f}s (litellm loaded: {litellm_loaded})")

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"\n  Average: {avg_time:.3f}s")
    print(f"  Min: {min_time:.3f}s")
    print(f"  Max: {max_time:.3f}s")
    print(f"  Target: {EVALUATION_TEST_IMPORT_TARGET_SECONDS}s")

    # Use the best time (min) as some CI environments have variable overhead
    assert min_time < EVALUATION_TEST_IMPORT_TARGET_SECONDS, (
        f"evaluation_test import time ({min_time:.3f}s) exceeds target ({EVALUATION_TEST_IMPORT_TARGET_SECONDS}s)."
    )


if __name__ == "__main__":
    # When run directly, always execute (ignore SKIP_BENCHMARK)
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
