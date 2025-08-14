"""
Pytest test for APPS coding evaluation using the evaluation_test decorator.

This test demonstrates how to evaluate code correctness for competitive programming problems
using the actual evaluate_apps_solution function from apps_coding_reward.py.
"""

import json
from typing import Any, Dict, List

from eval_protocol.models import EvaluateResult, EvaluationRow, Message
from eval_protocol.pytest import SingleTurnRolloutProcessor, evaluation_test
from eval_protocol.rewards.apps_coding_reward import evaluate_apps_solution


def apps_dataset_to_evaluation_row(data: List[Dict[str, Any]]) -> List[EvaluationRow]:
    """
    Convert entries from APPS dataset to EvaluationRow objects.
    """
    return [
        EvaluationRow(messages=[Message(role="user", content=row["question"])], ground_truth=row["input_output"])
        for row in data
    ]


@evaluation_test(
    input_dataset=["tests/pytest/data/apps_sample_dataset.jsonl"],
    dataset_adapter=apps_dataset_to_evaluation_row,
    completion_params=[
        {"temperature": 0.0, "max_tokens": 4096, "model": "fireworks_ai/accounts/fireworks/models/kimi-k2-instruct"}
    ],
    passed_threshold=0.33,
    rollout_processor=SingleTurnRolloutProcessor(),
    num_runs=1,
    mode="pointwise",
)
def test_apps_code_evaluation(row: EvaluationRow) -> EvaluationRow:
    """
    Evaluation function that tests APPS coding problems using evaluate_apps_solution.

    Args:
        row: EvaluationRow containing the conversation messages and ground_truth as JSON string

    Returns:
        EvaluationRow with the evaluation result
    """
    # Use evaluate_apps_solution directly
    result = evaluate_apps_solution(
        messages=row.messages,
        ground_truth=row.ground_truth,
    )

    # Set the evaluation result on the row
    row.evaluation_result = result

    return row
