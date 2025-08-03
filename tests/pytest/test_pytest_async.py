from typing import List

from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest import evaluation_test
from examples.math_example.main import evaluate as math_evaluate


@evaluation_test(
    input_messages=[
        [
            Message(role="user", content="What is the capital of France?"),
        ],
        [
            Message(role="user", content="What is the capital of the moon?"),
        ],
    ],
    model=["accounts/fireworks/models/kimi-k2-instruct"],
)
async def test_pytest_async(rows: List[EvaluationRow]) -> List[EvaluationRow]:
    """Run math evaluation on sample dataset using pytest interface."""
    return rows
