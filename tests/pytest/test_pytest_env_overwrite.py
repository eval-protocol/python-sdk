import atexit
import shutil
import tempfile
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
