from eval_protocol.models import Message, EvaluationRow, EvaluateResult
from eval_protocol.pytest import SingleTurnRolloutProcessor, evaluation_test
from typing import List
import pytest


@evaluation_test(
    input_messages=[
        [
            Message(role="user", content="What is the capital of France?"),
        ],
        [
            Message(role="user", content="What is the capital of the moon?"),
        ],
    ],
    completion_params=[{"model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b"}],
    rollout_processor=SingleTurnRolloutProcessor(),
    mode="all",
)
def test_direct_run(rows: List[EvaluationRow]) -> List[EvaluationRow]:
    """Run math evaluation on sample dataset using pytest interface."""
    for idx, row in enumerate(rows):
        row.evaluation_result = EvaluateResult(score=idx, reason="test")
    return rows


def test_direct_run_main():
    rows = [
        EvaluationRow(
            messages=[
                Message(role="user", content="What is the capital of France?"),
            ],
        ),
        EvaluationRow(
            messages=[
                Message(role="user", content="What is the capital of the moon?"),
            ],
        ),
    ]
    res = test_direct_run(rows)
    assert res[0].evaluation_result.score == 0
    assert res[1].evaluation_result.score == 1


@evaluation_test(
    input_messages=[
        [
            Message(role="user", content="What is the capital of France?"),
        ],
        [
            Message(role="user", content="What is the capital of the moon?"),
        ],
    ],
    completion_params=[{"model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b"}],
    rollout_processor=SingleTurnRolloutProcessor(),
    mode="all",
)
async def test_direct_run_async(rows: List[EvaluationRow]) -> List[EvaluationRow]:
    """Run math evaluation on sample dataset using pytest interface."""
    for idx, row in enumerate(rows):
        row.evaluation_result = EvaluateResult(score=idx, reason="test")
    return rows


@pytest.mark.asyncio
async def test_direct_run_async_main():
    rows = [
        EvaluationRow(
            messages=[
                Message(role="user", content="1"),
            ],
        ),
        EvaluationRow(
            messages=[
                Message(role="user", content="2"),
            ],
        ),
    ]
    res = await test_direct_run_async(rows)
    assert res[0].messages[0].content == "1"
    assert res[1].messages[0].content == "2"
    assert res[0].evaluation_result.score == 0
    assert res[1].evaluation_result.score == 1
