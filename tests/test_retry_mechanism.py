#!/usr/bin/env python3
"""
Simple test to verify the retry mechanism works with evaluation_test.
"""

import asyncio
import os
import time
from dataclasses import dataclass
from typing import AsyncIterator, List

from eval_protocol.models import EvaluateResult, EvaluationRow, Message, RolloutStatus
from eval_protocol.pytest.evaluation_test import evaluation_test
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig

os.environ["EP_MAX_RETRY"] = "2"  # Allow up to 2 retries

start_time = time.time()
timing_results = []  # Collect timing data for assertions


class MockRolloutProcessorWithRetries(RolloutProcessor):
    """Mock rollout processor that fails second task alphabetically on first attempt, succeeds on retry"""

    def __call__(self, rows: List[EvaluationRow], config: RolloutProcessorConfig) -> List[asyncio.Task[EvaluationRow]]:
        row_setup = {
            0: {"delay": 3.0, "should_fail": False},
            1: {"delay": 3.0, "should_fail": True},
            2: {"delay": 5.0, "should_fail": False},
            3: {"delay": 5.0, "should_fail": False},
            4: {"delay": 5.0, "should_fail": False},
        }

        async def process_single_row(row: EvaluationRow, delay: float, should_fail: bool = False) -> EvaluationRow:
            await asyncio.sleep(delay)

            elapsed = time.time() - start_time
            print(
                f"🎉 FINISHED {'error' if should_fail else 'finished'} at {elapsed:.2f}s: {row.execution_metadata.rollout_id}"
            )

            if should_fail:
                raise Exception("Simulated failure for testing")

            return row

        # Create and return tasks (let evaluation_test handle them)
        tasks = [
            asyncio.create_task(process_single_row(row, row_setup[i]["delay"], row_setup[i]["should_fail"]))
            for i, row in enumerate(rows)
        ]

        return tasks


@evaluation_test(
    completion_params=[{"model": "gpt-4o-mini", "temperature": 0}],
    input_messages=[
        [Message(role="user", content="Task A")],
        [Message(role="user", content="Task B")],
        [Message(role="user", content="Task C")],
        [Message(role="user", content="Task D")],
        [Message(role="user", content="Task E")],
    ],
    rollout_processor=MockRolloutProcessorWithRetries(),
    num_runs=1,
    mode="pointwise",
)
def test_retry_mechanism(row: EvaluationRow) -> EvaluationRow:
    """MOCK TEST: first 2 rows take 3s, last 3 take 5s, second row fails on first attempt, succeeds on retry. Should take around 6s total."""
    # Just print the timing - we'll parse it from output
    elapsed = time.time() - start_time
    print(
        f"📊 EVALUATED at {elapsed:.2f}s: {row.execution_metadata.rollout_id} ({'SUCCESS' if row.rollout_status.status == 'finished' else 'FAILURE'})"
    )

    # Assign a score based on success/failure
    score = 1.0 if row.rollout_status.status == "finished" else 0.0
    row.evaluation_result = EvaluateResult(score=score)

    return row


def test_timing_assertions():
    """Validate that timing results match expected pipeline behavior"""
    global start_time

    # Reset and run the evaluation test
    start_time = time.time()

    # Capture pytest output
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__ + "::test_retry_mechanism", "-v", "-s"],
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
    )

    print(result.stdout)  # Show the original output

    # Parse timing from output
    import re

    timing_results = []
    for line in result.stdout.split("\n"):
        match = re.search(r"📊 EVALUATED at (\d+\.\d+)s:", line)
        if match:
            timing_results.append(float(match.group(1)))

    print(f"\n📊 PIPELINE TIMING ANALYSIS:")
    print(f"   Results received at: {[f'{t:.2f}s' for t in sorted(timing_results)]}")

    # Assertions for expected timing behavior
    sorted_times = sorted(timing_results)

    assert len(sorted_times) == 5, f"Expected 5 evaluation results, got {len(sorted_times)}"

    # First result should be around 3s (row 0 success)
    assert 2.5 <= sorted_times[0] <= 3.5, f"First result at {sorted_times[0]:.2f}s, expected ~3s"

    # Next three results should be around 5s (rows 2,3,4)
    assert 4.5 <= sorted_times[1] <= 5.5, f"Second result at {sorted_times[1]:.2f}s, expected ~5s"
    assert 4.5 <= sorted_times[2] <= 5.5, f"Third result at {sorted_times[2]:.2f}s, expected ~5s"
    assert 4.5 <= sorted_times[3] <= 5.5, f"Fourth result at {sorted_times[3]:.2f}s, expected ~5s"

    # Last result should be around 6s (row 1 retry success)
    assert 5.5 <= sorted_times[4] <= 6.5, f"Fifth result at {sorted_times[4]:.2f}s, expected ~6s (retry success)"

    print("✅ All timing assertions passed! Pipeline behavior is correct.")


if __name__ == "__main__":
    test_timing_assertions()
