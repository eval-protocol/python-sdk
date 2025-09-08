"""
Reward function that combines accuracy with cosine-scaled length rewards.

This module provides a reward function that evaluates both the accuracy of
model responses and their length efficiency, combining them into a single
reward score.
"""

import math
from typing import Any, Callable, Dict, List, Optional, Union, cast

from ..models import EvaluateResult, Message, MetricResult
from ..typed_interface import reward_function
from .accuracy import accuracy_reward
from .length import count_tokens


@reward_function
def cosine_scaled_accuracy_length_reward(
    messages: List[Message],
    ground_truth: Optional[List[Message]] = None,
    extract_fn: Optional[Callable[[str], str]] = None,
    compare_fn: Optional[Callable[[str, str], float]] = None,
    max_length: int = 1000,
    min_value_wrong: float = 0.0,
    max_value_wrong: float = 0.3,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    token_method: str = "whitespace",
    correctness_weight: float = 0.7,
    length_weight: float = 0.3,
    **kwargs: Any,
) -> EvaluateResult:
    """
    Reward function that combines accuracy with cosine-scaled length rewards.

    Evaluates both the accuracy of the response and its length efficiency,
    combining them into a single score. Shorter correct answers are rewarded
    more than longer ones, while maintaining separation between answers.

    Args:
        messages: List of conversation messages
        ground_truth: Expected correct answer
        extract_fn: Optional function to extract answer from text
        compare_fn: Optional function to compare answers
        max_length: Maximum length for scaling (longer responses get penalized)
        min_value_wrong: Minimum reward for wrong answers
        max_value_wrong: Maximum reward for wrong answers
        min_value_correct: Minimum reward for correct answers
        max_value_correct: Maximum reward for correct answers
        token_method: Method to count tokens ('whitespace', 'character', etc)
        correctness_weight: Weight for the accuracy component (default: 0.7)
        length_weight: Weight for the length component (default: 0.3)
        **kwargs: Additional arguments

    Returns:
        EvaluateResult with score combining accuracy and length
    """
    if not messages or len(messages) == 0:
        return EvaluateResult(
            score=0.0,
            reason="No messages provided",
            metrics={"combined_reward": MetricResult(score=0.0, is_score_valid=False, reason="No messages provided")},
        )

    response = messages[-1]

    if response.role != "assistant" or not response.content:
        return EvaluateResult(
            score=0.0,
            reason="No assistant response found or response has no content",
            metrics={
                "combined_reward": MetricResult(
                    score=0.0,
                    is_score_valid=False,
                    reason="Message not from assistant or has no content",
                )
            },
        )
    # Coerce response content to string
    text: str
    if isinstance(response.content, str):
        text = response.content
    elif isinstance(response.content, list) and response.content:
        # Join text parts if provided as structured content
        try:
            text = " ".join(part.text for part in response.content)  # type: ignore[union-attr]
        except Exception:
            text = ""
    else:
        text = ""

    # Step 1: Evaluate accuracy
    # Ensure ground_truth is a list if provided; default to [] for compatibility
    gt_for_accuracy = ground_truth if ground_truth is not None else []
    accuracy_eval_result = cast(Any, accuracy_reward)(
        messages=messages,
        ground_truth=gt_for_accuracy,
        extract_fn=extract_fn,
        compare_fn=compare_fn,
    )

    accuracy_score = accuracy_eval_result.score
    # Ensure answer_accuracy metric exists, provide a default if not
    answer_accuracy_metric = accuracy_eval_result.metrics.get(
        "answer_accuracy",
        MetricResult(score=0.0, is_score_valid=False, reason="Accuracy metric not found"),
    )
    accuracy_success = answer_accuracy_metric.is_score_valid
    accuracy_reason = accuracy_eval_result.reason or "No reason from accuracy_reward"

    # Step 2: Calculate length-based score
    token_count = count_tokens(text, method=token_method)

    # Normalize token count relative to max_length
    progress = min(1.0, token_count / max_length)

    # Apply cosine scaling
    cosine_factor = math.cos(progress * math.pi)

    # Determine reward range based on correctness
    if accuracy_success:
        # For correct answers: shorter is better
        min_value = min_value_correct
        max_value = max_value_correct
        success = True
    else:
        # For incorrect answers: longer is slightly better (showing work)
        min_value = max_value_wrong
        max_value = min_value_wrong
        success = False

    # Calculate length-scaled score
    scale_factor = 0.5 * (max_value - min_value) * (1.0 + cosine_factor)
    length_score = min_value + scale_factor

    # Step 3: Calculate combined score (weighted average)
    acc_component = accuracy_score * correctness_weight
    len_component = length_score * length_weight
    combined_score = acc_component + len_component

    # Ensure the combined score is properly bounded
    combined_score = max(0.0, min(1.0, combined_score))

    # Prepare detailed reason
    reward_type = "reward" if accuracy_success else "penalty"
    length_reason = (
        f"Length-based {reward_type}: {token_count}/{max_length} tokens, cosine factor: {cosine_factor:.2f}"
    )

    combined_reason = (
        f"Combined score (acc:{accuracy_score:.2f}*{correctness_weight:.1f} + "
        f"len:{length_score:.2f}*{length_weight:.1f} = {combined_score:.2f}). "
        f"Accuracy: {accuracy_reason}. Length: {length_reason}"
    )

    # Prepare metrics
    metrics = {
        "combined_reward": MetricResult(
            score=combined_score,
            is_score_valid=success,
            reason=f"Combined score: {combined_score:.2f}",
        ),
        "accuracy": MetricResult(
            score=accuracy_score,
            is_score_valid=accuracy_success,
            reason=f"Accuracy: {accuracy_score:.2f}",
        ),
        "length": MetricResult(
            score=length_score,
            is_score_valid=token_count <= max_length,
            reason=f"Length: {token_count}/{max_length} tokens, score: {length_score:.2f}",  # noqa
        ),
        "token_count": MetricResult(
            score=min(1.0, max(0.0, 1.0 - progress)),
            is_score_valid=token_count <= max_length,
            reason=f"Token count: {token_count}/{max_length}",
        ),
    }

    return EvaluateResult(
        score=combined_score,
        reason=combined_reason,
        metrics=metrics,
        is_score_valid=combined_score > 0.0,
    )
