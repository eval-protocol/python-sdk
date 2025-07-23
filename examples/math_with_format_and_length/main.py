"""Math with Format and Length Evaluation Example.

This module demonstrates a custom reward function that combines:
- Numerical accuracy using the built-in `math_reward`
- Format compliance for `<think>...</think><answer>...</answer>`
- Length efficiency using a cosine-scaled penalty
"""

import math
import re
from typing import Any, Dict, List, Optional, Union

from eval_protocol import EvaluateResult, MetricResult, reward_function
from eval_protocol.models import Message
from eval_protocol.rewards.length import count_tokens
from eval_protocol.rewards.math import math_reward


def check_think_answer_format(text: str) -> bool:
    """Return True if text matches the required format."""
    if not text:
        return False
    pattern = r"^<think>[\s\S]*?</think>\s*<answer>[\s\S]*?</answer>$"
    return bool(re.match(pattern, text.strip()))


@reward_function
def evaluate(
    messages: Union[List[Message], List[Dict[str, Any]]],
    ground_truth: Optional[str] = None,
    *,
    max_length: int = 1000,
    min_value_wrong: float = 0.0,
    max_value_wrong: float = 0.3,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    token_method: str = "whitespace",
    **kwargs: Any,
) -> EvaluateResult:
    """Evaluate math reasoning with format and length considerations."""
    assistant_message = messages[-1]
    text = (
        assistant_message["content"]
        if isinstance(assistant_message, dict)
        else assistant_message.content or ""
    )

    # Accuracy using built-in math reward
    accuracy_result = math_reward(
        messages=messages, ground_truth=ground_truth, **kwargs
    )
    accuracy_score = accuracy_result.score

    # Format compliance
    format_correct = check_think_answer_format(text)
    format_score = 1.0 if format_correct else 0.0

    # Length score (cosine scaled)
    token_count = count_tokens(text, method=token_method)
    progress = min(1.0, token_count / max_length)
    cosine_factor = math.cos(progress * math.pi)
    if accuracy_score == 1.0:
        min_v = min_value_correct
        max_v = max_value_correct
    else:
        min_v = max_value_wrong
        max_v = min_value_wrong
    length_score = min_v + 0.5 * (max_v - min_v) * (1.0 + cosine_factor)

    combined_score = (accuracy_score + format_score + length_score) / 3.0

    metrics = {
        "accuracy_reward": MetricResult(
            score=accuracy_score, reason=accuracy_result.reason, is_score_valid=True
        ),
        "format_reward": MetricResult(
            score=format_score,
            reason="correct format" if format_correct else "incorrect format",
            is_score_valid=True,
        ),
        "length_reward": MetricResult(
            score=length_score,
            reason=f"{token_count} tokens",
            is_score_valid=token_count <= max_length,
        ),
    }

    return EvaluateResult(
        score=combined_score,
        reason=(
            f"Combined score {combined_score:.2f} (acc: {accuracy_score:.2f}, "
            f"format: {format_score:.2f}, length: {length_score:.2f})"
        ),
        metrics=metrics,
    )
