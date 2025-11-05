import atexit
import shutil
import tempfile
from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.default_no_op_rollout_processor import NoOpRolloutProcessor
from eval_protocol.pytest.default_single_turn_rollout_process import SingleTurnRolloutProcessor
from eval_protocol.pytest.plugin import pytest_configure
import os
from unittest import mock
from unittest.mock import MagicMock


with mock.patch.dict(os.environ, {"EP_INVOCATION_ID": "test-invocation-123"}):

    @evaluation_test(
        input_rows=[[EvaluationRow(messages=[Message(role="user", content="What is the capital of France?")])]],
        completion_params=[{"model": "no-op"}],
        rollout_processor=NoOpRolloutProcessor(),
        mode="pointwise",
    )
    def test_input_messages_in_decorator(row: EvaluationRow) -> EvaluationRow:
        """Run math evaluation on sample dataset using pytest interface."""
        assert row.messages[0].content == "What is the capital of France?"
        assert row.execution_metadata.invocation_id == "test-invocation-123"
        return row


with mock.patch.dict(os.environ, {"EP_COMPLETION_PARAMS": '[{"model": "gpt-40"}]'}):

    @evaluation_test(
        input_rows=[[EvaluationRow(messages=[Message(role="user", content="What is 5 * 6?")])]],
        completion_params=[{"model": "no-op"}],  # This should be overridden by the env var
        rollout_processor=NoOpRolloutProcessor(),
        mode="pointwise",
    )
    def test_input_messages_in_env(row: EvaluationRow) -> EvaluationRow:
        """Run math evaluation on sample dataset using pytest interface."""
        assert row.messages[0].content == "What is 5 * 6?"
        assert row.input_metadata.completion_params["model"] == "gpt-40"
        return row


_jsonl_tmpdir = tempfile.mkdtemp()
atexit.register(shutil.rmtree, _jsonl_tmpdir, ignore_errors=True)

input_path = os.path.join(_jsonl_tmpdir, "input.jsonl")
with open(input_path, "w") as f:
    f.write(
        '{"messages": [{"role": "user", "content": "What is 10 / 2?"}], "input_metadata": {"some_key": "some_value"}}\n'
    )
print(f"finish prepare input file {input_path}")
with mock.patch.dict(os.environ, {"EP_JSONL_PATH": input_path}):

    @evaluation_test(
        input_rows=[[EvaluationRow(messages=[Message(role="user", content="This will be ignored")])]],
        completion_params=[{"model": "no-op"}],
        rollout_processor=NoOpRolloutProcessor(),
        mode="pointwise",
    )
    def test_input_override(row: EvaluationRow) -> EvaluationRow:
        assert row.messages[0].content == "What is 10 / 2?"
        return row


# Tests for EP_USE_NO_OP_ROLLOUT_PROCESSOR override
with mock.patch.dict(os.environ, {"EP_USE_NO_OP_ROLLOUT_PROCESSOR": "1"}):

    @evaluation_test(
        input_rows=[[EvaluationRow(messages=[Message(role="user", content="Test message")])]],
        completion_params=[{"model": "no-op"}],
        rollout_processor=None,  # Should be overridden to NoOpRolloutProcessor
        mode="pointwise",
    )
    def test_no_op_rollout_processor_override_from_none(row: EvaluationRow) -> EvaluationRow:
        """Test that EP_USE_NO_OP_ROLLOUT_PROCESSOR overrides None rollout processor."""
        assert row.messages[0].content == "Test message"
        # With NoOpRolloutProcessor, the row should pass through unchanged
        # Verify that no actual model call was made (NoOpRolloutProcessor doesn't modify messages)
        assert len(row.messages) == 1
        assert row.messages[0].role == "user"
        return row

    @evaluation_test(
        input_rows=[[EvaluationRow(messages=[Message(role="user", content="Test override")])]],
        completion_params=[{"model": "no-op"}],
        rollout_processor=SingleTurnRolloutProcessor(),  # Should be overridden to NoOpRolloutProcessor
        mode="pointwise",
    )
    def test_no_op_rollout_processor_override_from_other(row: EvaluationRow) -> EvaluationRow:
        """Test that EP_USE_NO_OP_ROLLOUT_PROCESSOR overrides other rollout processors."""
        assert row.messages[0].content == "Test override"
        # With NoOpRolloutProcessor, the row should pass through unchanged without calling the model
        # Verify that no actual model call was made (NoOpRolloutProcessor doesn't modify messages)
        assert len(row.messages) == 1
        assert row.messages[0].role == "user"
        # Verify the original message content is preserved (no assistant response added)
        assert row.messages[0].content == "Test override"
        return row

    @evaluation_test(
        input_rows=[
            [
                EvaluationRow(messages=[Message(role="user", content="First")]),
                EvaluationRow(messages=[Message(role="user", content="Second")]),
            ]
        ],
        completion_params=[{"model": "no-op"}],
        rollout_processor=SingleTurnRolloutProcessor(),  # Should be overridden
        mode="pointwise",
    )
    def test_no_op_rollout_processor_override_multiple_rows(row: EvaluationRow) -> EvaluationRow:
        """Test that EP_USE_NO_OP_ROLLOUT_PROCESSOR works with multiple rows."""
        assert row.messages[0].content in ["First", "Second"]
        # Verify rows pass through unchanged
        assert len(row.messages) == 1
        assert row.messages[0].role == "user"
        return row


def test_pytest_plugin_sets_no_op_rollout_processor_env_var():
    """Test that pytest_configure sets EP_USE_NO_OP_ROLLOUT_PROCESSOR when flag is provided."""
    # Create a mock config object
    mock_config = MagicMock()

    # Mock getoption to return True when called with the flag name, None for others
    def getoption_side_effect(opt):
        if opt == "--ep-no-op-rollout-processor":
            return True
        return None

    mock_config.getoption = MagicMock(side_effect=getoption_side_effect)

    # Save original env var value if it exists
    original_value = os.environ.get("EP_USE_NO_OP_ROLLOUT_PROCESSOR")

    # Clear the environment variable first
    if "EP_USE_NO_OP_ROLLOUT_PROCESSOR" in os.environ:
        del os.environ["EP_USE_NO_OP_ROLLOUT_PROCESSOR"]

    try:
        # Call pytest_configure
        pytest_configure(mock_config)

        # Verify the environment variable was set
        assert os.environ.get("EP_USE_NO_OP_ROLLOUT_PROCESSOR") == "1"
    finally:
        # Clean up - restore original or remove
        if original_value is not None:
            os.environ["EP_USE_NO_OP_ROLLOUT_PROCESSOR"] = original_value
        elif "EP_USE_NO_OP_ROLLOUT_PROCESSOR" in os.environ:
            del os.environ["EP_USE_NO_OP_ROLLOUT_PROCESSOR"]


def test_pytest_plugin_does_not_set_env_var_when_flag_not_provided():
    """Test that pytest_configure does not set EP_USE_NO_OP_ROLLOUT_PROCESSOR when flag is not provided."""
    # Create a mock config object
    mock_config = MagicMock()

    # Mock getoption to return False when called with the flag name, None for others
    def getoption_side_effect(opt):
        if opt == "--ep-no-op-rollout-processor":
            return False
        return None

    mock_config.getoption = MagicMock(side_effect=getoption_side_effect)

    # Save original env var value if it exists
    original_value = os.environ.get("EP_USE_NO_OP_ROLLOUT_PROCESSOR")

    # Clear the environment variable first
    if "EP_USE_NO_OP_ROLLOUT_PROCESSOR" in os.environ:
        del os.environ["EP_USE_NO_OP_ROLLOUT_PROCESSOR"]

    try:
        # Call pytest_configure
        pytest_configure(mock_config)

        # Verify the environment variable was NOT set
        assert "EP_USE_NO_OP_ROLLOUT_PROCESSOR" not in os.environ
    finally:
        # Clean up - restore original if it existed
        if original_value is not None:
            os.environ["EP_USE_NO_OP_ROLLOUT_PROCESSOR"] = original_value
