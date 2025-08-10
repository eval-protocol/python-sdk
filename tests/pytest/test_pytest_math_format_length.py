import math

from eval_protocol.models import EvaluateResult, EvaluationRow, MetricResult
from eval_protocol.pytest import default_single_turn_rollout_processor, evaluation_test
from eval_protocol.rewards.length import count_tokens
from eval_protocol.rewards.math import math_reward
from examples.math_with_format_and_length.main import check_think_answer_format
from tests.pytest.helper.gsm8k_to_evaluation_row import gsm8k_to_evaluation_row


@evaluation_test(
    input_dataset=["development/gsm8k_sample.jsonl"],
    dataset_adapter=gsm8k_to_evaluation_row,
    model=["fireworks_ai/accounts/fireworks/models/kimi-k2-instruct"],
    rollout_input_params=[{"temperature": 0.0}],
    max_dataset_rows=5,
    threshold=0.0,
    rollout_processor=default_single_turn_rollout_processor,
    mode="pointwise",
    evaluation_test_kwargs=[
        {
            "config": {
                "max_length": 1000,
                "min_value_wrong": 0.0,
                "max_value_wrong": 0.3,
                "min_value_correct": 0.5,
                "max_value_correct": 1.0,
                "token_method": "whitespace",
            },
            "math_reward_kwargs": {"tolerance": 0.001, "absolute_tolerance": 1e-8, "require_units": False},
        }
    ],
)
def test_math_format_length_dataset(row: EvaluationRow, **kwargs) -> EvaluationRow:
    """Evaluate math reasoning with format and length considerations."""
    config = kwargs["config"]
    assistant_message = row.messages[-1]
    text = assistant_message["content"] if isinstance(assistant_message, dict) else assistant_message.content or ""

    # Accuracy using built-in math reward
    accuracy_result = math_reward(messages=row.messages, ground_truth=row.ground_truth, **kwargs["math_reward_kwargs"])
    accuracy_score = accuracy_result.score

    # Format compliance
    format_correct = check_think_answer_format(text)
    format_score = 1.0 if format_correct else 0.0

    # Length score (cosine scaled)
    token_count = count_tokens(text, method=config["token_method"])
    progress = min(1.0, token_count / config["max_length"])
    cosine_factor = math.cos(progress * math.pi)
    if accuracy_score == 1.0:
        min_v = config["min_value_correct"]
        max_v = config["max_value_correct"]
    else:
        min_v = config["max_value_wrong"]
        max_v = config["min_value_wrong"]
    length_score = min_v + 0.5 * (max_v - min_v) * (1.0 + cosine_factor)

    combined_score = (accuracy_score + format_score + length_score) / 3.0

    metrics = {
        "accuracy_reward": MetricResult(score=accuracy_score, reason=accuracy_result.reason, is_score_valid=True),
        "format_reward": MetricResult(
            score=format_score,
            reason="correct format" if format_correct else "incorrect format",
            is_score_valid=True,
        ),
        "length_reward": MetricResult(
            score=length_score,
            reason=f"{token_count} tokens",
            is_score_valid=token_count <= config["max_length"],
        ),
    }

    result = EvaluateResult(
        score=combined_score,
        reason=(
            f"Combined score {combined_score:.2f} (acc: {accuracy_score:.2f}, "
            f"format: {format_score:.2f}, length: {length_score:.2f})"
        ),
        metrics=metrics,
    )
    row.evaluation_result = result
    return row
