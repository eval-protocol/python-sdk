from typing import List

from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest import default_single_turn_rollout_processor, evaluation_test


@evaluation_test(
    input_messages=[
        [
            Message(role="user", content="What is the capital of France?"),
        ]
    ],
    completion_params=[{"model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b"}],
    rollout_processor=default_single_turn_rollout_processor,
)
def test_input_messages_in_decorator(rows: List[EvaluationRow]) -> List[EvaluationRow]:
    """Run math evaluation on sample dataset using pytest interface."""
    return rows
