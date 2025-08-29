from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.default_no_op_rollout_processor import NoOpRolloutProcessor


@evaluation_test(
    input_rows=[[EvaluationRow(messages=[Message(role="user", content="What is the capital of France?")])]],
    completion_params=[{"model": "no-op"}],
    rollout_processor=NoOpRolloutProcessor(),
    mode="pointwise",
)
def test_input_messages_in_decorator(row: EvaluationRow) -> EvaluationRow:
    """Run math evaluation on sample dataset using pytest interface."""
    assert row.messages[0].content == "What is the capital of France?"
    return row
