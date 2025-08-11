import json
from typing import Any, Dict, List

from eval_protocol.models import EvaluationRow
from eval_protocol.pytest import default_single_turn_rollout_processor, evaluation_test
from eval_protocol.rewards.function_calling import exact_tool_match_reward


def function_calling_to_evaluation_row(rows: List[Dict[str, Any]]) -> List[EvaluationRow]:
    """
    Convert a function calling row to an evaluation row.
    """
    dataset: List[EvaluationRow] = []
    for row in rows:
        dataset.append(
            EvaluationRow(messages=row["messages"][:1], tools=row["tools"], ground_truth=row["ground_truth"])
        )
    return dataset


@evaluation_test(
    input_dataset=["tests/pytest/data/function_calling.jsonl"],
    model=["fireworks_ai/accounts/fireworks/models/gpt-oss-120b"],
    mode="pointwise",
    dataset_adapter=function_calling_to_evaluation_row,
    rollout_processor=default_single_turn_rollout_processor,
)
async def test_pytest_function_calling(row: EvaluationRow) -> EvaluationRow:
    """Run pointwise evaluation on sample dataset using pytest interface."""
    ground_truth = json.loads(row.ground_truth)
    result = exact_tool_match_reward(row.messages, ground_truth)
    row.evaluation_result = result
    print(result)
    return row
