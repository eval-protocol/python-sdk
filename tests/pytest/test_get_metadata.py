import asyncio

from eval_protocol.pytest import evaluation_test
from eval_protocol.models import EvaluationRow, Message


@evaluation_test(
    input_messages=[
        [
            Message(role="user", content="What is the capital of France?"),
        ],
        [
            Message(role="user", content="What is the capital of the moon?"),
        ],
    ],
    completion_params=[{"model": "accounts/fireworks/models/kimi-k2-instruct"}] * 2,
    mode="groupwise",
    max_concurrent_rollouts=5,
    max_concurrent_evaluations=10,
)
def test_pytest_async(rows: list[EvaluationRow]) -> list[EvaluationRow]:
    """Run math evaluation on sample dataset using pytest interface."""
    return rows


def test_pytest_func_metainfo():
    assert hasattr(test_pytest_async, "_origin_func")
    origin_func = test_pytest_async._origin_func  # pyright: ignore[reportAny, reportFunctionMemberAccess]
    assert not asyncio.iscoroutinefunction(origin_func)  # pyright: ignore[reportAny]
    assert asyncio.iscoroutinefunction(test_pytest_async)
    assert test_pytest_async._metainfo["mode"] == "groupwise"  # pyright: ignore[reportAny, reportFunctionMemberAccess]
    assert test_pytest_async._metainfo["max_rollout_concurrency"] == 5  # pyright: ignore[reportAny, reportFunctionMemberAccess]
    assert test_pytest_async._metainfo["max_evaluation_concurrency"] == 10  # pyright: ignore[reportAny, reportFunctionMemberAccess]
