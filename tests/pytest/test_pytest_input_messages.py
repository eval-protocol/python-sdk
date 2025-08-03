from typing import List

from eval_protocol.models import EvaluationRow
from eval_protocol.pytest import default_single_turn_rollout_processor, evaluation_test


@evaluation_test(
    input_messages=[
        [
            {"role": "user", "content": "What is the capital of France?"},
        ]
    ],
    model=["accounts/fireworks/models/kimi-k2-instruct"],
    rollout_processor=default_single_turn_rollout_processor,
)
def test_input_messages_in_decorator(input_dataset: List[EvaluationRow], model) -> List[EvaluationRow]:
    """Run math evaluation on sample dataset using pytest interface."""
    return input_dataset
