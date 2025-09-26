from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.default_no_op_rollout_processor import NoOpRolloutProcessor
import os
from unittest import mock


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
