from eval_protocol.models import EvaluateResult, EvaluationRow, MetricResult
from eval_protocol.pytest import default_single_turn_rollout_processor, evaluation_test
from eval_protocol.rewards.math import math_reward
from examples.math_example.main import check_think_answer_format
from tests.pytest.helper.gsm8k_to_evaluation_row import gsm8k_to_evaluation_row


@evaluation_test(
    input_dataset=["development/gsm8k_sample.jsonl"],
    dataset_adapter=gsm8k_to_evaluation_row,
    completion_params=[{"temperature": 0.0, "model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b"}],
    max_dataset_rows=5,
    passed_threshold=0.0,
    rollout_processor=default_single_turn_rollout_processor,
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
    # Get the assistant's response
    assistant_message = row.messages[-1]
    if isinstance(assistant_message, dict):
        assistant_response = assistant_message.get("content", "")
    else:
        assistant_response = assistant_message.content or ""

    # Evaluate numerical accuracy using built-in function
    accuracy_result = math_reward(messages=row.messages, ground_truth=row.ground_truth, **kwargs["math_reward_kwargs"])

    # Evaluate format compliance (looking for <think>...</think><answer>...</answer> format)
    format_correct = check_think_answer_format(assistant_response)
    format_score = 1.0 if format_correct else 0.0

    # Calculate combined score with 80% accuracy and 20% formatting weight
    combined_score = (0.8 * accuracy_result.score) + (0.2 * format_score)

    # Create metrics structure expected by tests
    metrics = {
        "accuracy_reward": MetricResult(
            score=accuracy_result.score,
            reason=f"Numerical accuracy: {accuracy_result.reason}",
            is_score_valid=True,
        ),
        "format_reward": MetricResult(
            score=format_score,
            reason=f"Format compliance: {'correct' if format_correct else 'incorrect'} <think>...</think><answer>...</answer> structure",
            is_score_valid=True,
        ),
    }

    row.evaluation_result = EvaluateResult(
        score=combined_score,
        reason=f"Combined score: {combined_score:.2f} (accuracy: {accuracy_result.score:.2f}, format: {format_score:.2f})",
        metrics=metrics,
    )
    return row
