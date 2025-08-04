"""
Pytest test for APPS coding evaluation using the evaluation_test decorator.

This test demonstrates how to evaluate code correctness for competitive programming problems
using the actual evaluate_apps_solution function from apps_coding_reward.py.
"""

import json
from typing import Any, Dict, List

from eval_protocol.models import EvaluateResult, EvaluationRow, Message
from eval_protocol.pytest import default_single_turn_rollout_processor, evaluation_test
from eval_protocol.rewards.apps_coding_reward import evaluate_apps_solution


def apps_dataset_to_evaluation_row(data: List[Dict[str, Any]]) -> List[EvaluationRow]:
    """
    Convert entries from APPS dataset to EvaluationRow objects.
    """
    return [
        EvaluationRow(
            messages=[Message(role="user", content=row["question"])],
            ground_truth=row["input_output"]
        )
        for row in data
    ]


@evaluation_test(
    input_dataset=["tests/pytest/data/apps_sample_dataset.jsonl"],
    dataset_adapter=apps_dataset_to_evaluation_row,
    model=["accounts/fireworks/models/kimi-k2-instruct"],
    rollout_input_params=[{"temperature": 0.0, "max_tokens": 4096}],
    threshold_of_success=0.33,
    rollout_processor=default_single_turn_rollout_processor,
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