from typing import List

import pytest
from eval_protocol.models import EvaluationRow, Message, EvaluateResult
from eval_protocol.pytest import SingleTurnRolloutProcessor, evaluation_test


@pytest.mark.parametrize("completion_params", [{"model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b"}])
@evaluation_test(
    input_messages=[
        [
            [
                Message(role="user", content="What is the capital of France?"),
            ]
        ]
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    mode="all",
)
def test_input_messages_in_decorator(rows: List[EvaluationRow]) -> List[EvaluationRow]:
    """Run math evaluation on sample dataset using pytest interface."""
    for row in rows:
        row.evaluation_result = EvaluateResult(score=0.0, reason="Dummy evaluation result")
    return rows
