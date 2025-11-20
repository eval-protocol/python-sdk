from typing import Any, Dict, List

from eval_protocol.models import (
    EvaluateResult,
    EvaluationRow,
    MetricResult,
)
from eval_protocol.pytest.evaluation_test import evaluation_test
from eval_protocol.integrations.tinker_rollout_processor import TinkerRolloutProcessor
from eval_protocol.adapters.huggingface import create_gsm8k_adapter

# Import grading logic from tinker-cookbook to ensure consistency
try:
    from tinker_cookbook.recipes.math_rl.math_grading import grade_answer
except ImportError:
    grade_answer = None


# Separate data loading for reuse in train.py
def get_gsm8k_input_rows(limit: int = 10) -> List[EvaluationRow]:
    adapter = create_gsm8k_adapter()
    return list(adapter.get_evaluation_rows(split="test", limit=limit))


# Fetch some rows for the test
gsm8k_input_rows = get_gsm8k_input_rows(limit=10)


@evaluation_test(
    input_rows=gsm8k_input_rows,
    completion_params=[
        {
            "max_tokens": 512,
            "temperature": 0.0,  # Greedy for eval
        }
    ],
    rollout_processor=TinkerRolloutProcessor(model_name="meta-llama/Llama-3.1-8B-Instruct", renderer_name="llama3"),
    aggregation_method="mean",
    num_runs=1,
    max_concurrent_rollouts=4,
    mode="pointwise",
)
def test_gsm8k_tinker(row: EvaluationRow) -> EvaluationRow:
    assistant_msgs = [m for m in row.messages if m.role == "assistant"]
    if not assistant_msgs:
        score = 0.0
        reason = "No assistant response"
    else:
        model_response = assistant_msgs[-1].content
        # The content might be a list of content parts, handle that
        if model_response is None:
            model_response = ""
        elif not isinstance(model_response, str):
            # Simple join for now if it's a list
            model_response = "".join([p.text for p in model_response if hasattr(p, "text")])

        ground_truth = row.ground_truth

        if grade_answer:
            # Use Tinker's grading logic
            is_correct = grade_answer(model_response, str(ground_truth))
            score = 1.0 if is_correct else 0.0
            reason = f"Graded: {is_correct}. GT: {ground_truth}"
        else:
            # Fallback simple check
            score = 0.0
            reason = "Grading function not available"
            print("DEBUG: grade_answer is None")

    # DEBUG
    # print(f"DEBUG: Score: {score}, Reason: {reason[:100]}")

    row.evaluation_result = EvaluateResult(
        score=score, reason=reason, metrics={"accuracy": MetricResult(score=score, reason=reason)}
    )
    return row
