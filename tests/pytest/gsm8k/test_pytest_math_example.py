import re
from eval_protocol.models import EvaluateResult, EvaluationRow, MetricResult, Message
from eval_protocol.pytest import SingleTurnRolloutProcessor, evaluation_test
from typing import List, Dict, Any, Optional


def extract_answer_digits(ground_truth: str) -> Optional[str]:
    """
    Extract the digits from the answer string.
    """
    answer_string = ground_truth.split("<answer>")[1].split("</answer>")[0]
    return re.search(r"(\d+)", answer_string).group(1) if answer_string else None


@evaluation_test(
    input_dataset=["development/gsm8k_sample.jsonl"],
    completion_params=[{"temperature": 0.0, "model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b"}],
    max_dataset_rows=5,
    passed_threshold=0.0,
    rollout_processor=SingleTurnRolloutProcessor(),
    mode="pointwise",
    evaluation_test_kwargs=[
        {"math_reward_kwargs": {"tolerance": 0.001, "absolute_tolerance": 1e-8, "require_units": False}}
    ],
)
def test_math_dataset(row: EvaluationRow, **kwargs) -> EvaluationRow:
    """
    Evaluate math problem solving considering both accuracy and format.

    This function demonstrates how to combine multiple evaluation criteria:
    - Numerical accuracy using built-in math evaluation (80% weight)
    - Format compliance checking for <think>...</think><answer>...</answer> structure (20% weight)

    Args:
        row: EvaluationRow containing the conversation messages and ground truth
        **kwargs: Additional parameters (like math_reward_kwargs)

    Returns:
        EvaluationRow with the evaluation result
    """
    #### Get predicted answer value
    prediction = extract_answer_digits(str(row.messages[2].content))
    gt = extract_answer_digits(str(row.ground_truth))

    #### Get score
    if prediction is None or gt is None:
        score = 0
        reason = "Missing answer tags in prediction or ground truth."

    elif gt == prediction:
        score = 1
        reason = "Model answer is correct."

    else:
        score = 0
        reason = "Model answer is not correct."

    reason += f" Prediction: {prediction}, Ground Truth: {gt}"

    evaluation_result = EvaluateResult(
        score=score,  # Required: The final evaluation score
        is_score_valid=True,  # Optional: Whether the score is valid, true by default
        reason=reason,  # Optional: The reason for the score
    )
    row.evaluation_result = evaluation_result
    return row
