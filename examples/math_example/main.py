"""
Math Evaluation Example

This example shows how to create a custom reward function for math problems
that combines accuracy checking with format validation. The math example
expects answers to be in <think>...</think><answer>...</answer> format.
"""

import re
from typing import Any, Dict, List, Optional, Union

from reward_protocol import EvaluateResult, MetricResult, reward_function
from reward_protocol.models import Message

# Import the existing reward function from reward-kit
from reward_protocol.rewards.math import math_reward


def check_think_answer_format(text: str) -> bool:
    """Check if text follows <think>...</think><answer>...</answer> format."""
    if not text:
        return False
    pattern = r"<think>[\s\S]*?</think>[\s\S]*?<answer>[\s\S]*?</answer>"
    return bool(re.search(pattern, text))


@reward_function
def evaluate(
    messages: Union[List[Message], List[Dict[str, Any]]],
    ground_truth: Optional[str] = None,
    **kwargs,
) -> EvaluateResult:
    """
    Evaluate math problem solving considering both accuracy and format.

    This function demonstrates how to combine multiple evaluation criteria:
    - Numerical accuracy using built-in math evaluation
    - Format compliance checking for <think>...</think><answer>...</answer> structure

    Args:
        messages: The conversation messages including the math solution
        ground_truth: Expected answer for comparison
        **kwargs: Additional parameters (like tolerance)

    Returns:
        EvaluateResult with combined score and detailed metrics
    """
    # Get the assistant's response
    assistant_message = messages[-1]
    if isinstance(assistant_message, dict):
        assistant_response = assistant_message.get("content", "")
    else:
        assistant_response = assistant_message.content or ""

    # Evaluate numerical accuracy using built-in function
    accuracy_result = math_reward(
        messages=messages, ground_truth=ground_truth, **kwargs
    )

    # Evaluate format compliance (looking for <think>...</think><answer>...</answer> format)
    format_correct = check_think_answer_format(assistant_response)
    format_score = 1.0 if format_correct else 0.0

    # For math_example, accuracy takes priority - if accuracy is 0, overall score is 0
    # If accuracy is 1, then format can contribute to the score
    if accuracy_result.score == 0.0:
        combined_score = 0.0
    else:
        combined_score = accuracy_result.score  # Only accuracy matters for math_example

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

    return EvaluateResult(
        score=combined_score,
        reason=f"Combined score: {combined_score:.2f} (accuracy: {accuracy_result.score:.2f}, format: {format_score:.2f})",
        metrics=metrics,
    )
