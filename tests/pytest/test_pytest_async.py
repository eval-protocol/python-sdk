import pytest

from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest import evaluation_test


@evaluation_test(
    input_messages=[
        [
            [
                Message(role="user", content="What is the capital of France?"),
            ],
            [
                Message(role="user", content="What is the capital of the moon?"),
            ],
        ]
    ],
    completion_params=[{"model": "accounts/fireworks/models/kimi-k2-instruct"}],
    mode="all",
)
async def test_pytest_async(rows: list[EvaluationRow]) -> list[EvaluationRow]:
    """Run math evaluation on sample dataset using pytest interface."""
    return rows


@evaluation_test(
    input_messages=[
        [
            [
                Message(role="user", content="What is the capital of France?"),
            ],
        ]
    ],
    completion_params=[{"model": "accounts/fireworks/models/kimi-k2-instruct"}],
    mode="pointwise",
)
async def test_pytest_async_pointwise(row: EvaluationRow) -> EvaluationRow:
    """Run pointwise evaluation on sample dataset using pytest interface."""
    return row


@pytest.mark.asyncio
async def test_pytest_async_main():
    """
    Tests that we can just run the test function directly
    """
    rows = [
        EvaluationRow(
            messages=[
                Message(role="user", content="What is the capital of France?"),
            ],
        )
    ]
    result = await test_pytest_async(rows)  # pyright: ignore[reportGeneralTypeIssues, reportUnknownVariableType, reportArgumentType, reportCallIssue]
    assert result == rows


@pytest.mark.asyncio
async def test_pytest_async_pointwise_main():
    """
    Tests that we can just run the pointwise test function directly
    """
    row = EvaluationRow(
        messages=[
            Message(role="user", content="What is the capital of France?"),
        ],
    )
    result = await test_pytest_async_pointwise(row)  # pyright: ignore[reportGeneralTypeIssues, reportArgumentType, reportUnknownVariableType, reportCallIssue]
    assert result == row
