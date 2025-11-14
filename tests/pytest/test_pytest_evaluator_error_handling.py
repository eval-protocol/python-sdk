"""
Unit tests for evaluator error handling in evaluation_test.py.

Tests the error handling behavior added in lines 439-449 (pointwise) and
lines 477-488 (groupwise) that catches exceptions during evaluation and
properly sets eval_metadata.status and evaluation_result fields.

Key behaviors tested:
1. When an exception occurs during evaluation, the exception is caught
2. evaluation_result is set with:
   - score=0.0
   - is_score_valid=False
   - reason containing "Error during evaluation: {ExceptionType}: {message}"
3. eval_metadata.status is initially set to error, but then:
   - Gets overridden to eval_finished() at lines 601-606 if no rollout error
   - Gets overridden to score_invalid() in postprocess (lines 92-94) because is_score_valid=False
4. The final state has status.is_score_invalid() == True, with error details preserved in evaluation_result.reason

"""

import pytest
from typing_extensions import override
from eval_protocol.models import EvaluationRow, Message, Status, EvaluateResult
from eval_protocol.pytest.default_no_op_rollout_processor import NoOpRolloutProcessor
from eval_protocol.dataset_logger.dataset_logger import DatasetLogger


@pytest.fixture(autouse=True)
def _force_catch_eval_exceptions(monkeypatch: pytest.MonkeyPatch):
    """
    These tests validate the behavior when evaluation exceptions are caught and converted
    into evaluation_result/status fields. Ensure the env var is set to disable raising.
    """
    monkeypatch.setenv("EP_RAISE_EVAL_EXCEPTIONS", "false")


class TrackingLogger(DatasetLogger):
    """Custom logger that tracks all logged rows for testing."""

    def __init__(self, rollouts: dict[str, EvaluationRow]):
        self.rollouts: dict[str, EvaluationRow] = rollouts

    @override
    def log(self, row: EvaluationRow):
        if row.execution_metadata.rollout_id is None:
            raise ValueError("Rollout ID is None")
        self.rollouts[row.execution_metadata.rollout_id] = row

    @override
    def read(self, row_id: str | None = None) -> list[EvaluationRow]:
        return []


class TestPointwiseEvaluatorErrorHandling:
    """Test error handling in pointwise evaluation mode."""

    async def test_pointwise_evaluation_value_error(self):
        """Test that ValueError in evaluation function is properly caught and handled."""
        from eval_protocol.pytest.evaluation_test import evaluation_test

        input_messages = [
            [
                Message(
                    role="user",
                    content="Test message",
                ),
            ]
        ]

        rollouts: dict[str, EvaluationRow] = {}
        logger = TrackingLogger(rollouts)

        @evaluation_test(
            input_messages=[input_messages],
            rollout_processor=NoOpRolloutProcessor(),
            mode="pointwise",
            num_runs=1,
            logger=logger,
        )
        def eval_fn(row: EvaluationRow) -> EvaluationRow:
            # Simulate an error during evaluation
            raise ValueError("Test error in evaluation function")

        # Execute the test
        await eval_fn(input_messages=input_messages)  # pyright: ignore[reportCallIssue]

        # Verify error handling
        assert len(rollouts) == 1
        row = list(rollouts.values())[0]

        # Check evaluation_result was set with error details
        assert row.evaluation_result is not None
        assert row.evaluation_result.score == 0.0
        assert row.evaluation_result.is_score_valid is False
        assert "Error during evaluation: ValueError: Test error in evaluation function" in row.evaluation_result.reason  # pyright: ignore[reportOperatorIssue]

        # Check eval_metadata.status was set to error and is preserved (not overridden by postprocess)
        assert row.eval_metadata is not None
        assert row.eval_metadata.status is not None
        assert row.eval_metadata.status.is_error()
        assert (
            "Error during evaluation: ValueError: Test error in evaluation function"
            in row.eval_metadata.status.message
        )

    async def test_pointwise_evaluation_runtime_error(self):
        """Test that RuntimeError in evaluation function is properly caught and handled."""
        from eval_protocol.pytest.evaluation_test import evaluation_test

        input_messages = [
            [Message(role="user", content="Test message")],
        ]

        rollouts: dict[str, EvaluationRow] = {}
        logger = TrackingLogger(rollouts)

        @evaluation_test(
            input_messages=[input_messages],
            rollout_processor=NoOpRolloutProcessor(),
            mode="pointwise",
            num_runs=1,
            logger=logger,
        )
        def eval_fn(row: EvaluationRow) -> EvaluationRow:
            raise RuntimeError("Runtime error during evaluation")

        await eval_fn(input_messages=input_messages)  # pyright: ignore[reportCallIssue]

        # Verify error handling
        assert len(rollouts) == 1
        row = list(rollouts.values())[0]

        # Check error type is included in reason
        assert row.evaluation_result is not None
        assert "RuntimeError" in row.evaluation_result.reason  # pyright: ignore[reportOperatorIssue]
        # Status will be error and preserved (not overridden by postprocess)
        assert row.eval_metadata is not None
        assert row.eval_metadata.status is not None
        assert row.eval_metadata.status.is_error()

    async def test_pointwise_evaluation_multiple_runs_with_errors(self):
        """Test that errors are handled consistently across multiple runs."""
        from eval_protocol.pytest.evaluation_test import evaluation_test

        input_messages = [
            [Message(role="user", content="Test message")],
        ]

        rollouts: dict[str, EvaluationRow] = {}
        logger = TrackingLogger(rollouts)

        @evaluation_test(
            input_messages=[input_messages],
            rollout_processor=NoOpRolloutProcessor(),
            mode="pointwise",
            num_runs=3,
            logger=logger,
        )
        def eval_fn(row: EvaluationRow) -> EvaluationRow:
            raise ValueError("Consistent error")

        await eval_fn(input_messages=input_messages)  # pyright: ignore[reportCallIssue]

        # Verify all runs have error handling
        assert len(rollouts) == 3
        for row in rollouts.values():
            assert row.evaluation_result is not None
            assert row.evaluation_result.score == 0.0
            assert row.evaluation_result.is_score_valid is False
            assert "ValueError" in row.evaluation_result.reason  # pyright: ignore[reportOperatorIssue]
            # Status will be error and preserved
            assert row.eval_metadata is not None
            assert row.eval_metadata.status is not None
            assert row.eval_metadata.status.is_error()

    async def test_pointwise_evaluation_custom_exception(self):
        """Test handling of custom exception types."""
        from eval_protocol.pytest.evaluation_test import evaluation_test

        class CustomEvaluationError(Exception):
            """Custom exception for testing."""

            pass

        input_messages = [
            [Message(role="user", content="Test message")],
        ]

        rollouts: dict[str, EvaluationRow] = {}
        logger = TrackingLogger(rollouts)

        @evaluation_test(
            input_messages=[input_messages],
            rollout_processor=NoOpRolloutProcessor(),
            mode="pointwise",
            num_runs=1,
            logger=logger,
        )
        def eval_fn(row: EvaluationRow) -> EvaluationRow:
            raise CustomEvaluationError("Custom error with details")

        await eval_fn(input_messages=input_messages)  # pyright: ignore[reportCallIssue]

        # Verify custom exception is properly handled
        assert len(rollouts) == 1
        row = list(rollouts.values())[0]

        assert row.evaluation_result is not None
        assert "CustomEvaluationError" in row.evaluation_result.reason  # pyright: ignore[reportOperatorIssue]
        assert "Custom error with details" in row.evaluation_result.reason  # pyright: ignore[reportOperatorIssue]
        # Status will be error and preserved
        assert row.eval_metadata is not None
        assert row.eval_metadata.status is not None
        assert row.eval_metadata.status.is_error()

    async def test_pointwise_evaluation_error_with_multiline_message(self):
        """Test handling of errors with multiline error messages."""
        from eval_protocol.pytest.evaluation_test import evaluation_test

        input_messages = [
            [Message(role="user", content="Test message")],
        ]

        rollouts: dict[str, EvaluationRow] = {}
        logger = TrackingLogger(rollouts)

        @evaluation_test(
            input_messages=[input_messages],
            rollout_processor=NoOpRolloutProcessor(),
            mode="pointwise",
            num_runs=1,
            logger=logger,
        )
        def eval_fn(row: EvaluationRow) -> EvaluationRow:
            raise ValueError("Line 1\nLine 2\nLine 3")

        await eval_fn(input_messages=input_messages)  # pyright: ignore[reportCallIssue]

        # Verify multiline error message is captured
        assert len(rollouts) == 1
        row = list(rollouts.values())[0]

        assert row.evaluation_result is not None
        assert "Line 1\nLine 2\nLine 3" in row.evaluation_result.reason  # pyright: ignore[reportOperatorIssue]


class TestGroupwiseEvaluatorErrorHandling:
    """Test error handling in groupwise evaluation mode."""

    async def test_groupwise_evaluation_value_error(self):
        """Test that ValueError in groupwise evaluation function is properly caught and handled."""
        from eval_protocol.pytest.evaluation_test import evaluation_test

        input_messages = [
            [Message(role="user", content="Test message")],
        ]

        # Groupwise mode requires at least 2 completion_params
        completion_params_list = [
            {"model": "test/model-1"},
            {"model": "test/model-2"},
        ]

        rollouts: dict[str, EvaluationRow] = {}
        logger = TrackingLogger(rollouts)

        @evaluation_test(
            input_messages=[input_messages],
            completion_params=completion_params_list,
            rollout_processor=NoOpRolloutProcessor(),
            mode="groupwise",
            num_runs=1,
            logger=logger,
        )
        def eval_fn(rows: list[EvaluationRow]) -> list[EvaluationRow]:
            # Simulate an error during groupwise evaluation
            raise ValueError("Test error in groupwise evaluation")

        # Execute the test - groupwise mode groups all completion params together
        await eval_fn(input_messages=input_messages, completion_params=completion_params_list[0])  # pyright: ignore[reportCallIssue]

        # Verify error handling - groupwise should have rows for all completion params
        assert len(rollouts) > 0

        # Check that all rows have proper error handling
        for row in rollouts.values():
            if row.evaluation_result is not None:
                assert row.evaluation_result.score == 0.0
                assert row.evaluation_result.is_score_valid is False
                assert (
                    "Error during evaluation: ValueError: Test error in groupwise evaluation"
                    in row.evaluation_result.reason  # pyright: ignore[reportOperatorIssue]
                )

                # Status will be error and preserved
                assert row.eval_metadata is not None
                assert row.eval_metadata.status is not None
                assert row.eval_metadata.status.is_error()

    async def test_groupwise_evaluation_runtime_error(self):
        """Test that RuntimeError in groupwise evaluation function is properly caught and handled."""
        from eval_protocol.pytest.evaluation_test import evaluation_test

        input_messages = [
            [Message(role="user", content="Test message")],
        ]

        # Groupwise mode requires at least 2 completion_params
        completion_params_list = [
            {"model": "test/model-1"},
            {"model": "test/model-2"},
        ]

        rollouts: dict[str, EvaluationRow] = {}
        logger = TrackingLogger(rollouts)

        @evaluation_test(
            input_messages=[input_messages],
            completion_params=completion_params_list,
            rollout_processor=NoOpRolloutProcessor(),
            mode="groupwise",
            num_runs=1,
            logger=logger,
        )
        def eval_fn(rows: list[EvaluationRow]) -> list[EvaluationRow]:
            raise RuntimeError("Runtime error during groupwise evaluation")

        await eval_fn(input_messages=input_messages, completion_params=completion_params_list[0])  # pyright: ignore[reportCallIssue]

        # Verify error handling
        assert len(rollouts) > 0

        for row in rollouts.values():
            if row.evaluation_result is not None:
                assert "RuntimeError" in row.evaluation_result.reason  # pyright: ignore[reportOperatorIssue]
                # Status will be error and preserved
                assert row.eval_metadata is not None
                assert row.eval_metadata.status is not None
                assert row.eval_metadata.status.is_error()


class TestEvaluatorErrorHandlingEdgeCases:
    """Test edge cases for evaluator error handling."""

    async def test_evaluation_error_with_missing_eval_metadata(self):
        """Test error handling when eval_metadata is None (shouldn't happen but defensive)."""
        from eval_protocol.pytest.evaluation_test import evaluation_test

        input_messages = [
            [Message(role="user", content="Test message")],
        ]

        rollouts: dict[str, EvaluationRow] = {}
        logger = TrackingLogger(rollouts)

        @evaluation_test(
            input_messages=[input_messages],
            rollout_processor=NoOpRolloutProcessor(),
            mode="pointwise",
            num_runs=1,
            logger=logger,
        )
        def eval_fn(row: EvaluationRow) -> EvaluationRow:
            # Manually set eval_metadata to None to test defensive handling
            row.eval_metadata = None
            raise ValueError("Error with missing eval_metadata")

        await eval_fn(input_messages=input_messages)  # pyright: ignore[reportCallIssue]

        # Verify error handling doesn't crash even without eval_metadata
        assert len(rollouts) == 1
        row = list(rollouts.values())[0]

        # evaluation_result should still be set
        assert row.evaluation_result is not None
        assert row.evaluation_result.score == 0.0
        assert row.evaluation_result.is_score_valid is False

    async def test_evaluation_error_preserves_row_data(self):
        """Test that error handling preserves existing row data."""
        from eval_protocol.pytest.evaluation_test import evaluation_test

        input_messages = [
            [Message(role="user", content="Original message")],
        ]

        rollouts: dict[str, EvaluationRow] = {}
        logger = TrackingLogger(rollouts)

        @evaluation_test(
            input_messages=[input_messages],
            rollout_processor=NoOpRolloutProcessor(),
            mode="pointwise",
            num_runs=1,
            logger=logger,
        )
        def eval_fn(row: EvaluationRow) -> EvaluationRow:
            # Add some data to the row before error
            row.messages.append(Message(role="assistant", content="Response"))
            raise ValueError("Error after modifying row")

        await eval_fn(input_messages=input_messages)  # pyright: ignore[reportCallIssue]

        # Verify row data is preserved
        assert len(rollouts) == 1
        row = list(rollouts.values())[0]

        # Original messages should still be there
        assert len(row.messages) >= 1
        assert any(msg.content == "Original message" for msg in row.messages if msg.content)

    async def test_evaluation_error_with_empty_exception_message(self):
        """Test handling of exceptions with empty error messages."""
        from eval_protocol.pytest.evaluation_test import evaluation_test

        input_messages = [
            [Message(role="user", content="Test message")],
        ]

        rollouts: dict[str, EvaluationRow] = {}
        logger = TrackingLogger(rollouts)

        @evaluation_test(
            input_messages=[input_messages],
            rollout_processor=NoOpRolloutProcessor(),
            mode="pointwise",
            num_runs=1,
            logger=logger,
        )
        def eval_fn(row: EvaluationRow) -> EvaluationRow:
            raise ValueError("")  # Empty error message

        await eval_fn(input_messages=input_messages)  # pyright: ignore[reportCallIssue]

        # Verify error handling works with empty message
        assert len(rollouts) == 1
        row = list(rollouts.values())[0]

        assert row.evaluation_result is not None
        assert (
            "ValueError" in row.evaluation_result.reason
        )  # Should at least have the exception type  # pyright: ignore[reportOperatorIssue]


class TestEvaluatorErrorHandlingWithInputRows:
    """Test error handling when using input_rows parameter."""

    async def test_evaluation_error_with_input_rows(self):
        """Test error handling works correctly with input_rows parameter."""
        from eval_protocol.pytest.evaluation_test import evaluation_test

        # Create pre-constructed EvaluationRow
        input_row = EvaluationRow(
            messages=[
                Message(role="user", content="Test from input_rows"),
            ]
        )

        rollouts: dict[str, EvaluationRow] = {}
        logger = TrackingLogger(rollouts)

        @evaluation_test(
            input_rows=[[input_row]],
            rollout_processor=NoOpRolloutProcessor(),
            mode="pointwise",
            num_runs=1,
            logger=logger,
        )
        def eval_fn(row: EvaluationRow) -> EvaluationRow:
            raise ValueError("Error with input_rows")

        await eval_fn(input_rows=[input_row])  # pyright: ignore[reportCallIssue]

        # Verify error handling
        assert len(rollouts) == 1
        row = list(rollouts.values())[0]

        assert row.evaluation_result is not None
        assert row.evaluation_result.score == 0.0
        assert row.evaluation_result.is_score_valid is False
        assert "ValueError" in row.evaluation_result.reason  # pyright: ignore[reportOperatorIssue]
        # Status will be error and preserved
        assert row.eval_metadata is not None
        assert row.eval_metadata.status is not None
        assert row.eval_metadata.status.is_error()


class TestEvaluatorErrorHandlingStatusCodes:
    """Test that Status codes are correctly set for different error scenarios."""

    async def test_error_status_uses_internal_code(self):
        """Test that error status uses Status.Code.INTERNAL and is preserved."""
        from eval_protocol.pytest.evaluation_test import evaluation_test

        input_messages = [
            [Message(role="user", content="Test message")],
        ]

        rollouts: dict[str, EvaluationRow] = {}
        logger = TrackingLogger(rollouts)

        @evaluation_test(
            input_messages=[input_messages],
            rollout_processor=NoOpRolloutProcessor(),
            mode="pointwise",
            num_runs=1,
            logger=logger,
        )
        def eval_fn(row: EvaluationRow) -> EvaluationRow:
            raise ValueError("Test error")

        await eval_fn(input_messages=input_messages)  # pyright: ignore[reportCallIssue]

        assert len(rollouts) == 1
        row = list(rollouts.values())[0]

        # Verify status code is INTERNAL (13) and preserved (not overridden by postprocess)
        assert row.eval_metadata is not None
        assert row.eval_metadata.status is not None
        assert row.eval_metadata.status.code == Status.Code.INTERNAL
        assert row.eval_metadata.status.is_error()

    async def test_evaluation_result_reason_format(self):
        """Test that evaluation_result.reason contains the error details."""
        from eval_protocol.pytest.evaluation_test import evaluation_test

        input_messages = [
            [Message(role="user", content="Test message")],
        ]

        rollouts: dict[str, EvaluationRow] = {}
        logger = TrackingLogger(rollouts)

        @evaluation_test(
            input_messages=[input_messages],
            rollout_processor=NoOpRolloutProcessor(),
            mode="pointwise",
            num_runs=1,
            logger=logger,
        )
        def eval_fn(row: EvaluationRow) -> EvaluationRow:
            raise KeyError("missing_key")

        await eval_fn(input_messages=input_messages)  # pyright: ignore[reportCallIssue]

        assert len(rollouts) == 1
        row = list(rollouts.values())[0]

        # Verify reason format in evaluation_result: "Error during evaluation: ExceptionType: message"
        assert row.evaluation_result is not None
        reason = row.evaluation_result.reason
        assert reason is not None
        assert reason.startswith("Error during evaluation: ")
        assert "KeyError" in reason  # pyright: ignore[reportOperatorIssue]
        assert "missing_key" in reason  # pyright: ignore[reportOperatorIssue]

        # Status will be error and preserved
        assert row.eval_metadata is not None
        assert row.eval_metadata.status is not None
        assert row.eval_metadata.status.is_error()
        assert "KeyError" in row.eval_metadata.status.message
