"""
Pytest test for frozen lake evaluation using the evaluation_test decorator.

This test demonstrates how to use frozen lake environments within the pytest framework,
similar to the test_frozen_lake_e2e test but integrated with the pytest evaluation system.
"""

from typing import Any, Dict, List

from eval_protocol.models import EvaluateResult, EvaluationRow, InputMetadata, Message
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.default_mcp_gym_rollout_processor import MCPGymRolloutProcessor


def frozen_lake_to_evaluation_row(data: List[Dict[str, Any]]) -> List[EvaluationRow]:
    """
    Convert entries from frozen lake dataset to EvaluationRow objects.
    """
    rows = []

    for row in data:
        eval_row = EvaluationRow(
            messages=[Message(role="system", content=row["system_prompt"])],
            input_metadata=InputMetadata(
                row_id=row["id"],
                dataset_info={
                    "environment_context": row["environment_context"],
                    "user_prompt_template": row["user_prompt_template"],
                },
            ),
        )

        rows.append(eval_row)

    return rows


@evaluation_test(
    input_dataset=["tests/pytest/data/frozen_lake_dataset.jsonl"],
    dataset_adapter=frozen_lake_to_evaluation_row,
    completion_params=[
        {"temperature": 0.0, "max_tokens": 4096, "model": "fireworks_ai/accounts/fireworks/models/kimi-k2-instruct"}
    ],
    rollout_processor=MCPGymRolloutProcessor(),
    passed_threshold=0.66,
    num_runs=1,
    max_concurrent_rollouts=3,
    mode="pointwise",
    server_script_path="eval_protocol/mcp_servers/frozen_lake/server.py",
)
def test_frozen_lake_evaluation(row: EvaluationRow) -> EvaluationRow:
    """
    Test frozen lake evaluation using the pytest framework.

    This test evaluates how well the model can navigate the FrozenLake environment
    by checking if it successfully reaches the goal while avoiding holes.

    Args:
        row: EvaluationRow object from frozen lake dataset

    Returns:
        EvaluationRow object with evaluation results
    """
    score = row.get_total_reward()

    if score == 1.0:
        reason = "Agent reached the goal"
    else:
        reason = "Agent did not reach the goal"

    row.evaluation_result = EvaluateResult(
        score=score,
        reason=reason,
    )

    return row
