"""
Pytest test for lunar lander evaluation using the evaluation_test decorator.

This test demonstrates how to use lunar lander environments within the pytest framework,
similar to the test_lunar_lander_e2e test but integrated with the pytest evaluation system.
"""

from typing import Any, Dict, List

from eval_protocol.models import EvaluateResult, EvaluationRow, Message, InputMetadata, CompletionParams
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.default_mcp_gym_rollout_processor import default_mcp_gym_rollout_processor


def lunar_lander_to_evaluation_row(data: List[Dict[str, Any]]) -> List[EvaluationRow]:
    """
    Convert entries from lunar lander dataset to EvaluationRow objects.
    """
    rows = []
    
    for entry in data:
        row = EvaluationRow(
            messages=[Message(role="system", content=entry.get("system_prompt", ""))],
            input_metadata=InputMetadata(
                row_id=entry.get("id"),
                completion_params=CompletionParams(model="placeholder"), # This gets populated by the rollout processor
                dataset_info={
                    "environment_context": entry.get("environment_context", {}),
                    "user_prompt_template": entry.get("user_prompt_template", ""),
                }
            )
        )
        
        rows.append(row)
    
    return rows


@evaluation_test(
    input_dataset=["tests/pytest/data/lunar_lander_dataset.jsonl"],
    dataset_adapter=lunar_lander_to_evaluation_row,
    model=["gpt-4.1"],
    rollout_input_params=[{"temperature": 0.0, "max_tokens": 4096}],
    rollout_processor=default_mcp_gym_rollout_processor,
    threshold_of_success=0.0,
    num_runs=1,
    mode="pointwise",
    max_concurrent_rollouts=3,
    steps=15,
    server_script_path="examples/lunar_lander_mcp/server.py",
)
def test_lunar_lander_evaluation(row: EvaluationRow) -> EvaluationRow:
    """
    Test lunar lander evaluation using the pytest framework.
    
    This test evaluates how well the model can control the lunar lander to achieve
    a successful landing by checking the final reward and termination status.
    
    Args:
        row: EvaluationRow object from lunar lander dataset
        
    Returns:
        EvaluationRow object with evaluation results
    """
    score = row.get_total_reward()

    evaluation_score = 1.0 if score >= 200 else 0.0
    reason = f"✅ Successful landing with reward {score:.2f}" if score >= 200 else f"❌ Failed landing with reward {score:.2f}"

    row.evaluation_result = EvaluateResult(
        score=evaluation_score,
        reason=reason,
    )
    
    return row 