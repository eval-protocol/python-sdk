import json
from typing import Any, Dict, List

from eval_protocol.models import EvaluationRow
from eval_protocol.pytest import SingleTurnRolloutProcessor, evaluation_test
from eval_protocol.rewards.json_schema import json_schema_reward


def json_schema_to_evaluation_row(rows: List[Dict[str, Any]]) -> List[EvaluationRow]:
    """
    Convert a json schema row to an evaluation row.
    """
    dataset: List[EvaluationRow] = []
    for row in rows:
        dataset.append(
            EvaluationRow(
                messages=row["messages"][:1],
                ground_truth=row["ground_truth"],
                input_metadata=row["input_metadata"],
            )
        )
    return dataset


@evaluation_test(
    input_dataset=["tests/pytest/data/json_schema.jsonl"],
    completion_params=[{"model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b"}],
    mode="pointwise",
    rollout_processor=SingleTurnRolloutProcessor(),
    dataset_adapter=json_schema_to_evaluation_row,
)
async def test_pytest_function_calling(row: EvaluationRow) -> EvaluationRow:
    """Run pointwise evaluation on sample dataset using pytest interface."""
    expected_schema = row.input_metadata.dataset_info["expected_schema"]
    result = json_schema_reward(row.messages, expected_schema=expected_schema)
    row.evaluation_result = result
    print(row.evaluation_result)
    return row
