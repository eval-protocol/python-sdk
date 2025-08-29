#!/usr/bin/env python3
"""
Simple test to verify the retry mechanism works with evaluation_test.
"""
# pyright: reportAny=false
# pyright: reportPrivateImportUsage=false

import asyncio
from collections import Counter
from typing_extensions import override
from unittest.mock import Mock

from eval_protocol.models import EvaluateResult, EvaluationRow, Message
from eval_protocol.pytest.evaluation_test import evaluation_test
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig
from eval_protocol.pytest.exception_config import ExceptionHandlerConfig, BackoffConfig
import litellm


class MockRolloutProcessorWithRetries(RolloutProcessor):
    """Mock rollout processor that fails second task alphabetically on first attempt, succeeds on retry"""

    def __init__(self):
        self.mock_tracker: Mock = Mock()

    @override
    def __call__(self, rows: list[EvaluationRow], config: RolloutProcessorConfig) -> list[asyncio.Task[EvaluationRow]]:
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
                raise ConnectionError("Simulated failure for testing")

            return row

        # Create and return tasks (let evaluation_test handle them)
        tasks = [
            asyncio.create_task(process_single_row(row, row_setup[i]["delay"], row_setup[i]["should_fail"]))  # pyright: ignore[reportArgumentType]
            for i, row in enumerate(rows)
        ]

        return tasks


# Create a shared processor instance for testing
shared_processor = MockRolloutProcessorWithRetries()


@evaluation_test(
    completion_params=[{"model": "gpt-4o-mini", "temperature": 0}],
    input_messages=[
        [
            [Message(role="user", content="Task A")],
            [Message(role="user", content="Task B")],
            [Message(role="user", content="Task C")],
            [Message(role="user", content="Task D")],
            [Message(role="user", content="Task E")],
        ]
    ],
    rollout_processor=shared_processor,
    num_runs=1,
    mode="pointwise",
    exception_handler_config=ExceptionHandlerConfig(backoff_config=BackoffConfig(max_tries=3)),
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


# Test 2: Fail-fast exceptions should not retry
class MockRolloutProcessorFailFast(RolloutProcessor):
    """Mock processor that always raises ValueError (fail-fast exception)"""

    def __init__(self):
        self.mock_tracker: Mock = Mock()

    @override
    def __call__(self, rows: list[EvaluationRow], config: RolloutProcessorConfig) -> list[asyncio.Task[EvaluationRow]]:
        self.mock_tracker.batch_call(len(rows))

        async def process_single_row(row: EvaluationRow) -> EvaluationRow:
            self.mock_tracker.process_row_call(row.execution_metadata.rollout_id)
            # Always raise ValueError (fail-fast exception)
            raise ValueError("This should not be retried")

        tasks = [asyncio.create_task(process_single_row(row)) for row in rows]
        return tasks


shared_processor_fail_fast = MockRolloutProcessorFailFast()


@evaluation_test(
    completion_params=[{"model": "gpt-4o-mini", "temperature": 0}],
    input_messages=[[[Message(role="user", content="Test")]]],
    rollout_processor=shared_processor_fail_fast,
    num_runs=1,
    mode="pointwise",
    exception_handler_config=ExceptionHandlerConfig(backoff_config=BackoffConfig(max_tries=4)),
)
def test_fail_fast_exceptions(row: EvaluationRow) -> EvaluationRow:
    """Test that fail-fast exceptions like ValueError are not retried."""
    print(
        f"📊 EVALUATED: {row.execution_metadata.rollout_id} ({'SUCCESS' if row.rollout_status.is_finished() else 'FAILURE'})"
    )
    score = 1.0 if row.rollout_status.is_finished() else 0.0
    row.evaluation_result = EvaluateResult(score=score)
    return row


def test_fail_fast_verification():
    """Verify that fail-fast exceptions are not retried"""
    mock_tracker = shared_processor_fail_fast.mock_tracker

    print("\n🔄 FAIL-FAST TEST ANALYSIS:")
    print(f"   Batch calls made: {mock_tracker.batch_call.call_count}")
    print(f"   Total row processing calls: {mock_tracker.process_row_call.call_count}")

    # Debug: Print all the calls that were made
    print("   Batch call args:", mock_tracker.batch_call.call_args_list)
    print("   Process row call args:", mock_tracker.process_row_call.call_args_list)

    # Should have exactly 1 call (no retries for fail-fast exceptions)
    assert mock_tracker.process_row_call.call_count == 1, (
        f"Expected 1 call for fail-fast exception, got {mock_tracker.process_row_call.call_count}"
    )

    # Should have exactly 1 batch call (no retry batches)
    assert mock_tracker.batch_call.call_count == 1, f"Expected 1 batch call, got {mock_tracker.batch_call.call_count}"

    print("✅ Fail-fast exception test passed! ValueError was not retried.")


# Test 3: Custom giveup function
class MockRolloutProcessorCustomGiveup(RolloutProcessor):
    """Mock processor for testing custom giveup functions"""

    def __init__(self):
        self.mock_tracker: Mock = Mock()

    @override
    def __call__(self, rows: list[EvaluationRow], config: RolloutProcessorConfig) -> list[asyncio.Task[EvaluationRow]]:
        self.mock_tracker.batch_call(len(rows))

        async def process_single_row(row: EvaluationRow) -> EvaluationRow:
            self.mock_tracker.process_row_call(row.execution_metadata.rollout_id)

            # Raise real litellm exceptions based on task content
            task_content = row.messages[0].content if row.messages else ""
            if task_content is not None and "429" in task_content:
                raise litellm.RateLimitError(
                    "Rate limit exceeded", llm_provider="test", model="test-model"
                )  # Should retry
            else:
                raise litellm.BadRequestError(
                    "Bad request", model="test-model", llm_provider="test"
                )  # Should not retry

        tasks = [asyncio.create_task(process_single_row(row)) for row in rows]
        return tasks


shared_processor_custom_giveup = MockRolloutProcessorCustomGiveup()


# Custom giveup function for litellm exceptions
def custom_http_giveup(e: Exception) -> bool:
    # Don't retry bad requests (400-level errors), but do retry rate limits (429)
    if isinstance(e, litellm.BadRequestError):
        return True  # Give up immediately on bad requests
    elif isinstance(e, litellm.RateLimitError):
        return False  # Retry rate limits with backoff

    return False  # Retry everything else


@evaluation_test(
    completion_params=[{"model": "gpt-4o-mini", "temperature": 0}],
    input_messages=[
        [
            [Message(role="user", content="Test 429")],  # Should retry
            [Message(role="user", content="Test 400")],  # Should not retry
        ]
    ],
    rollout_processor=shared_processor_custom_giveup,
    num_runs=1,
    mode="pointwise",
    exception_handler_config=ExceptionHandlerConfig(
        retryable_exceptions={
            litellm.RateLimitError,
            litellm.BadRequestError,
        },
        backoff_config=BackoffConfig(max_tries=3, giveup_func=custom_http_giveup),
    ),
)
def test_custom_giveup_function(row: EvaluationRow) -> EvaluationRow:
    """Test custom giveup function behavior."""
    task_content = row.messages[0].content if row.messages else ""
    print(f"📊 EVALUATED: {task_content} ({'SUCCESS' if row.rollout_status.is_finished() else 'FAILURE'})")
    score = 1.0 if row.rollout_status.is_finished() else 0.0
    row.evaluation_result = EvaluateResult(score=score)
    return row


def test_custom_giveup_verification():
    """Verify custom giveup function works correctly"""
    mock_tracker = shared_processor_custom_giveup.mock_tracker

    print("\n🔄 CUSTOM GIVEUP TEST ANALYSIS:")
    print(f"   Batch calls made: {mock_tracker.batch_call.call_count}")
    print(f"   Total row processing calls: {mock_tracker.process_row_call.call_count}")

    call_args = mock_tracker.process_row_call.call_args_list
    rollout_ids = [call[0][0] for call in call_args]
    call_counts = Counter(rollout_ids)

    print(f"   Call counts per rollout_id: {dict(call_counts)}")

    # Should have 5 calls: 1 for 400 error (giveup immediately), 4 for 429 error (1 original + 3 backoff)
    assert mock_tracker.process_row_call.call_count == 5, (
        f"Expected 5 calls total, got {mock_tracker.process_row_call.call_count}"
    )

    # One rollout should be called 4 times (RateLimitError: 1 original + 3 backoff), one called once (BadRequestError: immediate giveup)
    call_count_values = list(call_counts.values())
    assert call_count_values.count(4) == 1, (
        f"Expected 1 rollout with 4 calls (RateLimitError: 1 original + 3 backoff), got {call_count_values}"
    )
    assert call_count_values.count(1) == 1, (
        f"Expected 1 rollout with 1 call (BadRequestError: immediate giveup), got {call_count_values}"
    )

    print("✅ Custom giveup function test passed! HTTP status-based retry logic worked correctly.")


# Test 4: Simple giveup function - retry all exceptions but give up on 4xx
class MockRolloutProcessorSimpleGiveup(RolloutProcessor):
    """Mock processor that raises BadRequestError"""

    def __init__(self):
        self.mock_tracker: Mock = Mock()

    @override
    def __call__(self, rows: list[EvaluationRow], config: RolloutProcessorConfig) -> list[asyncio.Task[EvaluationRow]]:
        self.mock_tracker.batch_call(len(rows))

        async def process_single_row(row: EvaluationRow) -> EvaluationRow:
            self.mock_tracker.process_row_call(row.execution_metadata.rollout_id)
            # Always raise BadRequestError (400) - should be caught by giveup
            mock_response = Mock()
            mock_response.status_code = 400
            error = litellm.BadRequestError("Bad request", model="test-model", llm_provider="test")
            error.response = mock_response
            raise error

        tasks = [asyncio.create_task(process_single_row(row)) for row in rows]
        return tasks


shared_processor_simple_giveup = MockRolloutProcessorSimpleGiveup()


# Simple giveup function for 4xx errors
def simple_4xx_giveup(e: Exception) -> bool:
    if hasattr(e, "response") and hasattr(e.response, "status_code"):  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
        status = e.response.status_code  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportAttributeAccessIssue]
        return 400 <= status < 500  # Give up on all 4xx client errors  # pyright: ignore[reportUnknownVariableType]
    return False  # Retry everything else


@evaluation_test(
    completion_params=[{"model": "gpt-4o-mini", "temperature": 0}],
    input_messages=[[[Message(role="user", content="Test 400 giveup")]]],
    rollout_processor=shared_processor_simple_giveup,
    num_runs=1,
    mode="pointwise",
    exception_handler_config=ExceptionHandlerConfig(
        retryable_exceptions={Exception},  # Retry all exceptions
        backoff_config=BackoffConfig(max_tries=5, giveup_func=simple_4xx_giveup),
    ),
)
def test_simple_giveup_function(row: EvaluationRow) -> EvaluationRow:
    """Test that giveup function prevents retries immediately."""
    print(
        f"📊 EVALUATED: {row.execution_metadata.rollout_id} ({'SUCCESS' if row.rollout_status.is_finished() else 'FAILURE'})"
    )
    score = 1.0 if row.rollout_status.is_finished() else 0.0
    row.evaluation_result = EvaluateResult(score=score)
    return row


def test_simple_giveup_verification():
    """Verify that giveup function prevents retries."""
    mock_tracker = shared_processor_simple_giveup.mock_tracker

    print("\n🔄 SIMPLE GIVEUP TEST ANALYSIS:")
    print(f"   Batch calls made: {mock_tracker.batch_call.call_count}")
    print(f"   Total row processing calls: {mock_tracker.process_row_call.call_count}")
    print("   Process row call args:", mock_tracker.process_row_call.call_args_list)

    # Should have exactly 1 call (giveup function should prevent retries)
    assert mock_tracker.process_row_call.call_count == 1, (
        f"Expected 1 call due to giveup, got {mock_tracker.process_row_call.call_count}"
    )

    print("✅ Simple giveup test passed! 4xx error was not retried due to giveup function.")
