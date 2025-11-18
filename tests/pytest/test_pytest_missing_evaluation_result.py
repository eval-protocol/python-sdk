import pytest

from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest.default_no_op_rollout_processor import NoOpRolloutProcessor
from eval_protocol.pytest.evaluation_test import evaluation_test


@pytest.mark.asyncio
async def test_missing_evaluation_result_raises_assertion_error() -> None:
    """evaluation_test should raise if any EvaluationRow is missing evaluation_result."""

    input_messages = [
        [Message(role="user", content="Test message")],
    ]

    @evaluation_test(
        input_messages=[input_messages],
        rollout_processor=NoOpRolloutProcessor(),
        mode="pointwise",
        num_runs=1,
    )
    def eval_fn(row: EvaluationRow) -> EvaluationRow:
        # Intentionally forget to set row.evaluation_result
        return row

    with pytest.raises(AssertionError) as excinfo:
        # Trigger the evaluation; this should hit the assertion added in evaluation_test.py
        await eval_fn(input_messages=input_messages)  # pyright: ignore[reportCallIssue]

    msg = str(excinfo.value)
    assert "Some EvaluationRow instances are missing evaluation_result" in msg
    assert "must set `row.evaluation_result`" in msg
