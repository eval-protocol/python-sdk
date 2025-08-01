from eval_protocol.models import EvaluateResult, EvaluationRow
from eval_protocol.pytest_utils import evaluate, evaluation_test
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
    model=["accounts/fireworks/models/kimi-k2-instruct", "gpt-4o"],
)
async def test_input_messages_in_decorator(input_messages, model):
    """Run math evaluation on sample dataset using pytest interface."""
    return [
        EvaluationRow(
            messages=input_messages,
            evaluation_result=EvaluateResult(
                score=0.0,
            ),
        )
    ]
