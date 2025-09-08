"""
Reward functions for validating text format.

This module provides reward functions that validate if text responses
adhere to specific formatting requirements, such as containing specific tags
in the correct order.
"""

import re
from typing import Any, Dict, List, Optional, Union  # Added Optional

from ..models import EvaluateResult, Message, MetricResult
from ..typed_interface import reward_function


@reward_function
def format_reward(
    messages: Union[List[Message], List[Dict[str, Any]]],
    ground_truth: Optional[Union[List[Message], List[Dict[str, Any]]]] = None,
    format_regex: str = r"^<think>\n.*?</think>\n<answer>\n.*?</answer>$",
    require_exact_match: bool = True,
    **kwargs: Any,
) -> EvaluateResult:
    """
    Reward function that validates if text follows a specific format pattern.
    The model's response is assumed to be the last message in the `messages` list.

    By default, this checks for <think> and <answer> tags in the correct order,
    ensuring proper separation of reasoning and final answer.

    Args:
        messages: List of conversation messages, where `messages[-1]` is the model's response.
        ground_truth: Optional. Expected assistant response trajectory. Not directly used by this format reward.
        format_regex: Regular expression pattern to match. Default checks for
                      <think>...</think> followed by <answer>...</answer>.
        require_exact_match: If True, the entire text must match the pattern.
                             If False, pattern just needs to be found in text.
        **kwargs: Additional arguments.

    Returns:
        EvaluateResult with score 1.0 if format is correct, 0.0 otherwise
    """
    if not messages or len(messages) == 0:
        return EvaluateResult(
            score=0.0,
            reason="No messages provided",
            metrics={"format_check": MetricResult(score=0.0, is_score_valid=False, reason="No messages provided")},
            is_score_valid=False,
        )

    response = messages[-1]

    if isinstance(response, Message):
        if response.role != "assistant" or not response.content:
            return EvaluateResult(
                score=0.0,
                reason="No assistant response found",
                metrics={
                    "format_check": MetricResult(
                        score=0.0,
                        is_score_valid=False,
                        reason="Message not from assistant or has no content",
                    )
                },
                is_score_valid=False,
            )
        text = response.content
    elif isinstance(response, dict):
        if response.get("role") != "assistant" or not response.get("content"):
            return EvaluateResult(
                score=0.0,
                reason="No assistant response found",
                metrics={
                    "format_check": MetricResult(
                        score=0.0,
                        is_score_valid=False,
                        reason="Message not from assistant or has no content",
                    )
                },
                is_score_valid=False,
            )
        text = response.get("content", "")
    else:
        return EvaluateResult(
            score=0.0,
            reason="Last message is of unexpected type.",
            metrics={
                "format_check": MetricResult(
                    score=0.0,
                    is_score_valid=False,
                    reason="Invalid message type in messages.",
                )
            },
            is_score_valid=False,
        )

    pattern = re.compile(format_regex, re.DOTALL)

    # Ensure text is a string for regex functions
    text_str = text if isinstance(text, str) else str(text)

    if require_exact_match:
        match = pattern.match(text_str)
    else:
        match = pattern.search(text_str)

    if match:
        return EvaluateResult(
            score=1.0,
            reason="Format is correct",
            metrics={
                "format_check": MetricResult(
                    score=1.0,
                    is_score_valid=True,
                    reason="Text follows the required format pattern",
                )
            },
            is_score_valid=True,
        )
    else:
        return EvaluateResult(
            score=0.0,
            reason="Format is incorrect",
            metrics={
                "format_check": MetricResult(
                    score=0.0,
                    is_score_valid=False,
                    reason="Text does not follow the required format pattern",
                )
            },
            is_score_valid=False,
        )
