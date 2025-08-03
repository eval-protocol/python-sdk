from typing import List

from eval_protocol.models import EvaluationRow
from eval_protocol.pytest import evaluation_test
from examples.math_example.main import evaluate as math_evaluate


@evaluation_test(
    input_messages=[
        [
            {"role": "user", "content": "What is the capital of France?"},
        ],
        [
            {"role": "user", "content": "What is the capital of the moon?"},
        ],
    ],
    model=["accounts/fireworks/models/kimi-k2-instruct"],
)
async def test_pytest_async(input_dataset: List[EvaluationRow], model) -> List[EvaluationRow]:
    """Run math evaluation on sample dataset using pytest interface."""
    return input_dataset
