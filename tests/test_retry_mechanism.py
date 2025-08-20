#!/usr/bin/env python3
"""
Simple test to verify the retry mechanism works with evaluation_test.
"""

import asyncio
import os
from collections import Counter
from typing import List
from unittest.mock import Mock, patch

import pytest

from eval_protocol.models import EvaluateResult, EvaluationRow, Message, Status
from eval_protocol.pytest.evaluation_test import evaluation_test
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig


class MockRolloutProcessorWithRetries(RolloutProcessor):
    """Mock rollout processor that fails second task alphabetically on first attempt, succeeds on retry"""

    def __init__(self):
        self.mock_tracker = Mock()

    def __call__(self, rows: List[EvaluationRow], config: RolloutProcessorConfig) -> List[asyncio.Task[EvaluationRow]]:
        # Track this batch call
        self.mock_tracker.batch_call(len(rows))

        row_setup = {
            0: {"delay": 0.01, "should_fail": False},
            1: {"delay": 0.01, "should_fail": True},  # Will be adjusted based on attempt number
            2: {"delay": 0.01, "should_fail": False},
            3: {"delay": 0.01, "should_fail": False},
            4: {"delay": 0.01, "should_fail": False},
        }

        async def process_single_row(
            row: EvaluationRow, delay: float, base_should_fail: bool = False
        ) -> EvaluationRow:
            rollout_id = row.execution_metadata.rollout_id

            # Track individual row processing call
            self.mock_tracker.process_row_call(rollout_id)

            # Determine attempt number by counting previous calls for this rollout_id
            previous_calls = [
                call for call in self.mock_tracker.process_row_call.call_args_list if call[0][0] == rollout_id
            ]
            attempt_number = len(previous_calls)

            # Determine if this specific attempt should fail
            # Row 1 fails on first attempt (attempt_number == 1), succeeds on retry (attempt_number == 2)
            should_fail = base_should_fail and attempt_number == 1

            print(f"🔄 ATTEMPTING rollout_id={rollout_id}, attempt={attempt_number}, will_fail={should_fail}")

            await asyncio.sleep(delay)
            print(f"🎉 FINISHED {'error' if should_fail else 'finished'}: {row.execution_metadata.rollout_id}")

            if should_fail:
                raise Exception("Simulated failure for testing")

            return row

        # Create and return tasks (let evaluation_test handle them)
        tasks = [
            asyncio.create_task(process_single_row(row, row_setup[i]["delay"], row_setup[i]["should_fail"]))
            for i, row in enumerate(rows)
        ]

        return tasks


# Create a shared processor instance for testing
shared_processor = MockRolloutProcessorWithRetries()


@patch.dict(os.environ, {"EP_MAX_RETRY": "2"})
@evaluation_test(
    completion_params=[{"model": "gpt-4o-mini", "temperature": 0}],
    input_messages=[
        [Message(role="user", content="Task A")],
        [Message(role="user", content="Task B")],
        [Message(role="user", content="Task C")],
        [Message(role="user", content="Task D")],
        [Message(role="user", content="Task E")],
    ],
    rollout_processor=shared_processor,
    num_runs=1,
    mode="pointwise",
)
def test_retry_mechanism(row: EvaluationRow) -> EvaluationRow:
    """MOCK TEST: Tests that retry mechanism works - one task fails on first attempt, succeeds on retry."""
    print(
        f"📊 EVALUATED: {row.execution_metadata.rollout_id} ({'SUCCESS' if row.rollout_status.is_finished() else 'FAILURE'})"
    )

    # Assign a score based on success/failure
    score = 1.0 if row.rollout_status.is_finished() else 0.0
    row.evaluation_result = EvaluateResult(score=score)

    return row


@patch.dict(os.environ, {"EP_MAX_RETRY": "2"})
def test_retry_mechanism_mock_verification():
    """Test that verifies the retry mechanism worked by checking the mock calls"""
    # Get our mock tracker
    mock_tracker = shared_processor.mock_tracker

    print("\n🔄 MOCK CALL ANALYSIS:")
    print(f"   Batch calls made: {mock_tracker.batch_call.call_count}")
    print(f"   Total row processing calls: {mock_tracker.process_row_call.call_count}")

    if mock_tracker.process_row_call.call_count == 0:
        print("⚠️  No calls recorded yet. The evaluation test may not have run or completed.")
        return

    # Get all rollout_ids that were processed
    call_args = mock_tracker.process_row_call.call_args_list
    rollout_ids = [call[0][0] for call in call_args]

    # Count calls per rollout_id
    call_counts = Counter(rollout_ids)

    print(f"   Call counts per rollout_id: {dict(call_counts)}")
    print("   Individual calls:")
    for i, call_arg in enumerate(call_args, 1):
        rollout_id = call_arg[0][0]
        attempt_num = rollout_ids[:i].count(rollout_id)
        print(f"     {i}. rollout_id={rollout_id}, attempt={attempt_num}")

    # ASSERTIONS USING MOCK DATA
    # Should have exactly 6 total row processing calls (5 initial + 1 retry)
    assert mock_tracker.process_row_call.call_count == 6, (
        f"Expected 6 total calls, got {mock_tracker.process_row_call.call_count}"
    )

    # Should have exactly 2 batch calls (initial batch + retry batch)
    assert mock_tracker.batch_call.call_count == 2, f"Expected 2 batch calls, got {mock_tracker.batch_call.call_count}"

    # First batch should have 5 rows, second batch should have 1 row (the retry)
    batch_call_args = mock_tracker.batch_call.call_args_list
    assert batch_call_args[0][0][0] == 5, f"Expected first batch to have 5 rows, got {batch_call_args[0][0][0]}"
    assert batch_call_args[1][0][0] == 1, f"Expected second batch to have 1 row, got {batch_call_args[1][0][0]}"

    # Exactly one rollout_id should be called twice, others called once
    call_count_values = list(call_counts.values())
    assert call_count_values.count(2) == 1, (
        f"Expected exactly 1 rollout_id to be called twice, got counts: {dict(call_counts)}"
    )
    assert call_count_values.count(1) == 4, (
        f"Expected exactly 4 rollout_ids to be called once, got counts: {dict(call_counts)}"
    )

    print("✅ All mock-based assertions passed! Retry mechanism is working correctly.")
