from typing import List

from eval_protocol.models import EvaluationRow, Message, EvaluateResult
from eval_protocol.pytest import SingleTurnRolloutProcessor, evaluation_test

@evaluation_test(
    input_messages=[
        [
            Message(role="user", content="What is the capital of France?"),
        ]
    ],
    completion_params=[
        {"model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b"},
        {"model": "fireworks_ai/accounts/fireworks/models/gpt-4.1"},
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    mode="groupwise",
)
def test_pytest_groupwise(rows: List[EvaluationRow]) -> List[EvaluationRow]:
    """Run math evaluation on sample dataset using pytest interface."""
    assert rows[0].input_metadata.completion_params["model"] == "fireworks_ai/accounts/fireworks/models/gpt-oss-120b"
    assert rows[1].input_metadata.completion_params["model"] == "fireworks_ai/accounts/fireworks/models/gpt-4.1"
    rows[0].evaluation_result = EvaluateResult(score=1.0, reason="test")
    rows[1].evaluation_result = EvaluateResult(score=0.0, reason="test")
    print(rows[0].model_dump_json())
    print(rows[1].model_dump_json())
    return rows