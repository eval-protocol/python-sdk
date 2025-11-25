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


@pytest.mark.parametrize(
    "completion_params",
    [
        {
            "model": "fireworks_ai/accounts/fireworks/models/qwen3-30b-a3b",
            "logprobs": True,
            # "include_routing_matrix": True,  # Requires --enable-moe-stats on server
            "temperature": 0.6,
            "max_tokens": 256,
        }
    ],
)
@evaluation_test(
    input_messages=[
        [
            [
                Message(role="user", content="What is 2+2?"),
            ]
        ]
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    mode="all",
)
def test_single_turn_with_logprobs_and_routing_matrix(rows: List[EvaluationRow]) -> List[EvaluationRow]:
    """Test SingleTurnRolloutProcessor with logprobs and routing_matrix extraction."""
    for row in rows:
        # Check if extra metadata was extracted
        extra = row.execution_metadata.extra
        print("\n=== DEBUG: execution_metadata.extra ===")
        print(f"extra type: {type(extra)}")
        print(f"extra keys: {extra.keys() if isinstance(extra, dict) else 'N/A'}")

        if isinstance(extra, dict):
            if "token_ids" in extra:
                token_ids = extra["token_ids"]
                print(f"token_ids: found, len={len(token_ids)}, first 10 ids={token_ids[:10]}")
            else:
                print("token_ids: NOT FOUND")

            if "routing_matrix" in extra:
                routing_matrix = extra["routing_matrix"]
                print(f"routing_matrix: found, len={len(routing_matrix)}")
            else:
                print("routing_matrix: NOT FOUND")

            if "logprobs" in extra:
                print("logprobs: found")
            else:
                print("logprobs: NOT FOUND")

        print("=" * 50)

        row.evaluation_result = EvaluateResult(score=1.0, reason="Test passed")
    return rows
