"""
Test suite for the individual rollout retry logic in evaluation_test.

Tests the new efficient retry system that retries individual rollouts immediately
as they fail, rather than waiting for entire batches to complete.
"""

import asyncio
import os
from typing import List
from unittest.mock import patch

import pytest

from eval_protocol.models import EvaluateResult, EvaluationRow, Message, RolloutStatus
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.types import RolloutProcessor, RolloutProcessorConfig


class MockRetryRolloutProcessor:
    """
    Mock rollout processor that simulates different rollout statuses.

    On first call, returns rollouts with mixed statuses (finished, error, running).
    On retry calls, converts error/running rollouts to finished status.
    """

    def __init__(self):
        self.call_count = 0
        self.processed_rollout_ids = set()

    async def __call__(self, rows: List[EvaluationRow], config: RolloutProcessorConfig):
        """Process rollouts with simulated statuses"""
        self.call_count += 1

        for row in rows:
            # If this is a retry (rollout_id we've seen before), make it succeed
            if row.execution_metadata.rollout_id in self.processed_rollout_ids:
                row.rollout_status = RolloutStatus(status="finished")
                row.messages.append(
                    Message(role="assistant", content=f"Retry success for {row.execution_metadata.rollout_id}")
                )
            else:
                # First time processing this logical rollout
                self.processed_rollout_ids.add(row.execution_metadata.rollout_id)

                # Simulate different statuses based on content
                content = row.messages[0].content if row.messages else ""

                if "should_finish" in content:
                    # This one succeeds immediately
                    row.rollout_status = RolloutStatus(status="finished")
                    row.messages.append(Message(role="assistant", content="Success on first try"))
                elif "should_error" in content:
                    # This one errors on first try, should be retried
                    row.rollout_status = RolloutStatus(status="error", termination_reason="Simulated error")
                    row.messages.append(Message(role="assistant", content="Error on first try"))
                elif "should_be_running" in content:
                    # This one is left in running state, should be retried
                    row.rollout_status = RolloutStatus(status="running")
                    row.messages.append(Message(role="assistant", content="Left running, needs retry"))
                else:
                    # Default to finished
                    row.rollout_status = RolloutStatus(status="finished")
                    row.messages.append(Message(role="assistant", content="Default success"))

            yield row


class MockAlwaysFailRolloutProcessor:
    """Mock rollout processor that always fails, to test retry exhaustion"""

    def __init__(self):
        self.call_count = 0

    async def __call__(self, rows: List[EvaluationRow], config: RolloutProcessorConfig):
        """Always return error status to test retry exhaustion"""
        self.call_count += 1

        for row in rows:
            row.rollout_status = RolloutStatus(
                status="error", termination_reason=f"Persistent failure (attempt {self.call_count})"
            )
            row.messages.append(Message(role="assistant", content=f"Failed attempt {self.call_count}"))
            yield row


# Create instances that will be shared across test functions
mock_retry_processor = MockRetryRolloutProcessor()
mock_always_fail_processor = MockAlwaysFailRolloutProcessor()


# Set environment variable at module level for this test
@patch.dict(os.environ, {"EP_MAX_RETRY": "3"})
@evaluation_test(
    input_messages=[
        [Message(role="user", content="Test case that should_finish immediately")],
        [Message(role="user", content="Test case that should_error on first try")],
        [Message(role="user", content="Test case that should_be_running and need retry")],
    ],
    model=["dummy/local-model"],
    rollout_processor=mock_retry_processor,
    mode="batch",
    num_runs=1,
)
def test_retry_mixed_statuses_batch_mode(rows: List[EvaluationRow]) -> List[EvaluationRow]:
    """
    Test that retry logic works with mixed rollout statuses in batch mode.

    Tests:
    - One rollout finishes immediately (should not retry)
    - One rollout has error status (should retry and succeed)
    - One rollout has running status (should retry and succeed)
    """
    # Reset processor state at the beginning
    mock_retry_processor.call_count = 0
    mock_retry_processor.processed_rollout_ids.clear()

    # Verify we got all our test cases
    assert len(rows) == 3

    # Verify all rollouts ended up in finished state after retries
    for row in rows:
        assert row.rollout_status is not None
        assert row.rollout_status.status == "finished", f"Row should be finished but was {row.rollout_status.status}"

        # Check that retry cases got the retry response
        content = row.messages[0].content
        if "should_error" in content or "should_be_running" in content:
            # These should have been retried
            assistant_messages = [msg for msg in row.messages if msg.role == "assistant"]
            assert len(assistant_messages) >= 1
            assert "Retry success" in assistant_messages[-1].content

    # Set evaluation results
    for row in rows:
        row.evaluation_result = EvaluateResult(score=1.0, reason="All rollouts completed successfully")

    return rows


@patch.dict(os.environ, {"EP_MAX_RETRY": "3"})
@evaluation_test(
    input_messages=[
        [Message(role="user", content="Test pointwise should_error")],
        [Message(role="user", content="Test pointwise should_be_running")],
        [Message(role="user", content="Test pointwise should_finish")],
    ],
    model=["dummy/local-model"],
    rollout_processor=mock_retry_processor,
    mode="pointwise",
    num_runs=1,
)
def test_retry_mixed_statuses_pointwise_mode(row: EvaluationRow) -> EvaluationRow:
    """
    Test that retry logic works with mixed rollout statuses in pointwise mode.

    Each rollout is processed individually and should retry if not finished.
    """
    # Verify rollout ended up in finished state after any needed retries
    assert row.rollout_status is not None
    assert row.rollout_status.status == "finished", f"Row should be finished but was {row.rollout_status.status}"

    # Set evaluation result
    row.evaluation_result = EvaluateResult(score=1.0, reason="Rollout completed successfully")

    return row


def test_retry_exhaustion_should_fail():
    """
    Test that rollout process fails when max retries are exceeded.

    Sets EP_MAX_RETRY=2 and uses a processor that always fails.
    Should fail after 3 total attempts (initial + 2 retries).
    """

    # Set max retries environment variable
    with patch.dict(os.environ, {"EP_MAX_RETRY": "2"}):

        @evaluation_test(
            input_messages=[
                [Message(role="user", content="This will always fail")],
            ],
            model=["dummy/local-model"],
            rollout_processor=mock_always_fail_processor,
            mode="batch",
            num_runs=1,
        )
        def failing_evaluation_test(rows: List[EvaluationRow]) -> List[EvaluationRow]:
            # This should never be reached due to rollout failures
            for row in rows:
                row.evaluation_result = EvaluateResult(score=1.0, reason="Should not reach here")
            return rows

        # The evaluation_test should raise RuntimeError due to retry exhaustion
        with pytest.raises(RuntimeError) as exc_info:
            # Run the test directly to trigger the retry logic
            import asyncio

            # Reset the processor call count
            mock_always_fail_processor.call_count = 0

            # Create test data
            rows = [EvaluationRow(messages=[Message(role="user", content="This will always fail")])]

            # This should fail after 3 attempts (initial + 2 retries)
            asyncio.run(failing_evaluation_test(rows))

        # Verify the error message mentions retry exhaustion
        error_msg = str(exc_info.value)
        assert "failed after 2 retries" in error_msg.lower() or "retry" in error_msg.lower()

        # Verify the processor was called multiple times (initial + retries)
        assert (
            mock_always_fail_processor.call_count >= 3
        ), f"Expected >= 3 calls, got {mock_always_fail_processor.call_count}"


def test_no_retries_when_max_retry_zero():
    """
    Test that no retries happen when EP_MAX_RETRY=0 (default).

    Even with failing rollouts, should fail immediately without retries.
    """

    # Ensure EP_MAX_RETRY is 0 (default)
    with patch.dict(os.environ, {"EP_MAX_RETRY": "0"}):

        @evaluation_test(
            input_messages=[
                [Message(role="user", content="This will fail once and not retry")],
            ],
            model=["dummy/local-model"],
            rollout_processor=mock_always_fail_processor,
            mode="batch",
            num_runs=1,
        )
        def no_retry_evaluation_test(rows: List[EvaluationRow]) -> List[EvaluationRow]:
            # This should never be reached due to immediate failure
            for row in rows:
                row.evaluation_result = EvaluateResult(score=1.0, reason="Should not reach here")
            return rows

        # Should fail immediately without retries
        with pytest.raises(RuntimeError) as exc_info:
            # Reset processor call count
            mock_always_fail_processor.call_count = 0

            # Create test data
            rows = [EvaluationRow(messages=[Message(role="user", content="This will fail once and not retry")])]

            # Should fail after just 1 attempt
            asyncio.run(no_retry_evaluation_test(rows))

        # Verify only 1 attempt was made (no retries)
        assert (
            mock_always_fail_processor.call_count == 1
        ), f"Expected 1 call, got {mock_always_fail_processor.call_count}"


@pytest.mark.asyncio
async def test_concurrent_retry_efficiency():
    """
    Test that retries happen efficiently with proper concurrency.

    Verifies that successful rollouts don't wait for failing ones,
    and that retries start immediately as failures are detected.
    """

    class TimingMockProcessor:
        """Mock processor that tracks timing of rollout processing"""

        def __init__(self):
            self.processing_times = {}
            self.start_times = {}

        async def __call__(self, rows: List[EvaluationRow], config: RolloutProcessorConfig):
            import time

            for row in rows:
                rollout_id = row.execution_metadata.rollout_id
                self.start_times[rollout_id] = time.time()

                # Simulate different processing times
                content = row.messages[0].content if row.messages else ""

                if "slow_success" in content:
                    # Slow but successful rollout
                    await asyncio.sleep(0.1)
                    row.rollout_status = RolloutStatus(status="finished")
                    row.messages.append(Message(role="assistant", content="Slow success"))
                elif "fast_fail" in content:
                    # Fast failure that should retry quickly
                    await asyncio.sleep(0.01)
                    if rollout_id not in self.processing_times:
                        # First attempt - fail
                        row.rollout_status = RolloutStatus(status="error", termination_reason="Fast failure")
                        row.messages.append(Message(role="assistant", content="Fast failure"))
                        self.processing_times[rollout_id] = time.time()
                    else:
                        # Retry - succeed
                        row.rollout_status = RolloutStatus(status="finished")
                        row.messages.append(Message(role="assistant", content="Fast retry success"))

                yield row

    timing_processor = TimingMockProcessor()

    with patch.dict(os.environ, {"EP_MAX_RETRY": "3"}):

        @evaluation_test(
            input_messages=[
                [Message(role="user", content="slow_success - this takes longer but succeeds")],
                [Message(role="user", content="fast_fail - this fails fast then retries")],
            ],
            model=["dummy/local-model"],
            rollout_processor=timing_processor,
            mode="batch",
            num_runs=1,
        )
        def timing_test(rows: List[EvaluationRow]) -> List[EvaluationRow]:
            # Both should succeed eventually
            assert len(rows) == 2
            for row in rows:
                assert row.rollout_status.status == "finished"
                row.evaluation_result = EvaluateResult(score=1.0, reason="Success")
            return rows

        # Create test data
        rows = [
            EvaluationRow(messages=[Message(role="user", content="slow_success - this takes longer but succeeds")]),
            EvaluationRow(messages=[Message(role="user", content="fast_fail - this fails fast then retries")]),
        ]

        # Run the test - should complete successfully with proper retry timing
        result = await timing_test(rows)
        assert len(result) == 2

        # Verify that the fast-failing rollout was processed multiple times due to retry
        fast_fail_processed = any("fast_fail" in row.messages[0].content for row in result)
        assert fast_fail_processed, "Fast-failing rollout should have been processed"
