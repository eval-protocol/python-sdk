"""
Reward functions for evaluating response length.

This module provides reward functions that evaluate the length of model responses,
either by simple token/character count or using cosine-scaled rewards to promote
token efficiency.
"""

import math
import re
from typing import Any, Callable, Dict, List, Optional, Union

from ..models import EvaluateResult, Message, MetricResult
from ..typed_interface import reward_function


def count_tokens(text: str, method: str = "whitespace") -> int:
    """
    Count tokens in text using different methods.

    Args:
        text: The text to tokenize
        method: Tokenization method to use ('whitespace', 'character', or 'words')

    Returns:
        Token count based on the selected method
    """
    if method == "character":
        return len(text)
    elif method == "whitespace":
        return len(re.split(r"\s+", text.strip()))
    elif method == "words":
        return len(re.findall(r"\b[\w\d]+\b", text))
    else:
        return len(re.split(r"\s+", text.strip()))


@reward_function  # type: ignore[arg-type]
def length_reward(
    messages: Union[List[Message], List[Dict[str, Any]]],
    *,
    ground_truth: Optional[
        Union[List[Message], List[Dict[str, Any]]]
    ] = None,  # Not used by this function but part of standard signature
    target_length: Optional[int] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    token_method: str = "whitespace",
    scaling: str = "linear",
    reward_range: Optional[List[float]] = None,
    **kwargs: Any,
) -> EvaluateResult:
    """
    Reward function that evaluates the length of model responses.
    The model's response is assumed to be the last message in the `messages` list.

    This function can calculate rewards based on token count and can encourage either
    conciseness or thoroughness by setting appropriate min/max/target parameters.

    Args:
        messages: List of conversation messages, where `messages[-1]` is the model's response.
        ground_truth: Optional. Expected assistant response trajectory. Not directly used by this length reward.
        target_length: Optional target token count (optimal length).
        min_length: Minimum acceptable token count.
        max_length: Maximum acceptable token count.
        token_method: Method to count tokens ('whitespace', 'character', or 'words')
        scaling: Scaling method for reward calculation ('linear' or 'cosine')
        reward_range: Range for reward values, default is [0.0, 1.0]
        **kwargs: Additional arguments

    Returns:
        EvaluateResult with score based on length evaluation
    """
    if not messages or len(messages) == 0:
        return EvaluateResult(
            score=0.0,
            reason="No messages provided",
            metrics={"length": MetricResult(score=0.0, is_score_valid=False, reason="No messages provided")},
        )

    response = messages[-1]

    if isinstance(response, Message):
        if response.role != "assistant" or not response.content:
            return EvaluateResult(
                score=0.0,
                reason="No assistant response found",
                metrics={
                    "length": MetricResult(
                        score=0.0,
                        is_score_valid=False,
                        reason="Message not from assistant or has no content",
                    )
                },
            )
        text = response.content if isinstance(response.content, str) else ""
    elif isinstance(response, dict):
        if response.get("role") != "assistant" or not response.get("content"):
            return EvaluateResult(
                score=0.0,
                reason="No assistant response found",
                metrics={
                    "length": MetricResult(
                        score=0.0,
                        is_score_valid=False,
                        reason="Message not from assistant or has no content",
                    )
                },
            )
        text_val = response.get("content", "")
        text = text_val if isinstance(text_val, str) else ""
    else:
        return EvaluateResult(
            score=0.0,
            reason="Last message is of unexpected type.",
            metrics={
                "length": MetricResult(
                    score=0.0,
                    is_score_valid=False,
                    reason="Invalid message type in messages.",
                )
            },
        )

    token_count = count_tokens(text, method=token_method)

    if reward_range is None:
        reward_range = [0.0, 1.0]
    min_reward, max_reward = reward_range

    if target_length is not None:
        normalized_diff = abs(token_count - target_length) / target_length if target_length > 0 else 1.0
        if scaling == "cosine":
            progress = min(1.0, normalized_diff)
            score = min_reward + (max_reward - min_reward) * (1.0 + math.cos(progress * math.pi)) / 2.0
        else:
            score = max(
                min_reward,
                max_reward - normalized_diff * (max_reward - min_reward),
            )
        reason = (
            f"Response length ({token_count} tokens) deviated by {normalized_diff:.2f} from target ({target_length})"
        )
        success = normalized_diff < 0.2
    elif min_length is not None and max_length is not None:
        if token_count < min_length:
            progress = token_count / min_length
            if scaling == "cosine":
                score = min_reward + (max_reward - min_reward) * (1.0 - math.cos(progress * math.pi / 2.0))
            else:
                score = min_reward + (max_reward - min_reward) * progress
            reason = f"Response length ({token_count} tokens) is below minimum ({min_length})"
            success = False
        elif token_count > max_length:
            excess = token_count - max_length
            range_size = (
                max_length - min_length if max_length > min_length else 1
            )  # Avoid division by zero if min_length == max_length
            progress = min(
                1.0,
                excess / range_size if range_size > 0 else (1.0 if excess > 0 else 0.0),
            )

            if scaling == "cosine":
                score = max_reward - (max_reward - min_reward) * (1.0 - math.cos(progress * math.pi / 2.0))
            else:
                score = max_reward - (max_reward - min_reward) * progress
            reason = f"Response length ({token_count} tokens) exceeds maximum ({max_length})"
            success = False
        else:
            score = max_reward
            reason = f"Response length ({token_count} tokens) is within acceptable range ({min_length}-{max_length})"
            success = True
    elif min_length is not None:
        if token_count < min_length:
            progress = token_count / min_length
            if scaling == "cosine":
                score = min_reward + (max_reward - min_reward) * (1.0 - math.cos(progress * math.pi / 2.0))
            else:
                score = min_reward + (max_reward - min_reward) * progress
            reason = f"Response length ({token_count} tokens) is below minimum ({min_length})"
            success = False
        else:
            score = max_reward
            reason = f"Response length ({token_count} tokens) meets minimum requirement ({min_length})"
            success = True
    elif max_length is not None:
        if token_count > max_length:
            excess = token_count - max_length
            progress = min(
                1.0,
                excess / max_length if max_length > 0 else (1.0 if excess > 0 else 0.0),
            )
            if scaling == "cosine":
                score = max_reward - (max_reward - min_reward) * (1.0 - math.cos(progress * math.pi / 2.0))
            else:
                score = max_reward - (max_reward - min_reward) * progress
            reason = f"Response length ({token_count} tokens) exceeds maximum ({max_length})"
            success = False
        else:
            score = max_reward
            reason = f"Response length ({token_count} tokens) is within maximum limit ({max_length})"
            success = True
    else:
        # This is useful when combined with correctness metrics
        # E.g., shorter correct answers > longer correct answers > incorrect answers
        reference_length = 100  # Default length for normalization
        normalized_length = token_count / reference_length
        if scaling == "cosine":
            progress = min(1.0, normalized_length)
            score = min_reward + (max_reward - min_reward) * (1.0 + math.cos(progress * math.pi)) / 2.0
        else:
            progress = min(1.0, normalized_length)
            score = max_reward - progress * (max_reward - min_reward)
        reason = f"Response length: {token_count} tokens"
        success = True

    metrics = {
        "length": MetricResult(score=score, is_score_valid=success, reason=reason),
        "token_count": MetricResult(
            score=min(
                1.0,
                float(token_count) / (target_length or max_length or min_length or 100),
            ),
            is_score_valid=success,
            reason=f"Token count: {token_count}",
        ),
    }

    return EvaluateResult(score=score, reason=reason, metrics=metrics)


@reward_function  # type: ignore[arg-type]
def cosine_length_reward(
    messages: Union[List[Message], List[Dict[str, Any]]],
    *,
    ground_truth: Optional[
        Union[List[Message], List[Dict[str, Any]]]
    ] = None,  # Not used by this function but part of standard signature
    correctness: Optional[float] = None,
    is_correct: Optional[bool] = None,
    max_length: int = 1000,
    min_value_wrong: float = 0.0,
    max_value_wrong: float = 0.3,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    token_method: str = "whitespace",
    **kwargs: Any,
) -> EvaluateResult:
    """
    Reward function that scales based on completion length using a cosine schedule.
    The model's response is assumed to be the last message in the `messages` list.

    Inspired by the OpenR1 implementation (https://github.com/OpenRL-Lab/open-r1) and
    Kimi Technical Report (https://arxiv.org/abs/2501.12599).

    Shorter correct solutions are rewarded more than longer ones.
    Longer incorrect solutions are penalized less than shorter ones.

    Args:
        messages: List of conversation messages, where `messages[-1]` is the model's response.
        ground_truth: Optional. Expected assistant response trajectory. Not directly used by this length reward.
        correctness: Optional float (0-1) indicating solution correctness.
        is_correct: Optional boolean indicating if the solution is correct.
        max_length: Maximum length for scaling.
        min_value_wrong: Minimum reward for wrong answers (typically negative)
        max_value_wrong: Maximum reward for wrong answers (typically negative but closer to zero)
        min_value_correct: Minimum reward for correct answers (typically positive)
        max_value_correct: Maximum reward for correct answers (typically more positive)
        token_method: Method to count tokens
        **kwargs: Additional arguments

    Returns:
        EvaluateResult with score based on cosine-scaled length evaluation
    """
    if not messages or len(messages) == 0:
        return EvaluateResult(
            score=0.0,
            reason="No messages provided",
            metrics={"cosine_length": MetricResult(score=0.0, is_score_valid=False, reason="No messages provided")},
        )

    response = messages[-1]

    if isinstance(response, Message):
        if response.role != "assistant" or not response.content:
            return EvaluateResult(
                score=0.0,
                reason="No assistant response found",
                metrics={
                    "cosine_length": MetricResult(
                        score=0.0,
                        is_score_valid=False,
                        reason="Message not from assistant or has no content",
                    )
                },
            )
        text = response.content if isinstance(response.content, str) else ""
    elif isinstance(response, dict):
        if response.get("role") != "assistant" or not response.get("content"):
            return EvaluateResult(
                score=0.0,
                reason="No assistant response found",
                metrics={
                    "cosine_length": MetricResult(
                        score=0.0,
                        is_score_valid=False,
                        reason="Message not from assistant or has no content",
                    )
                },
            )
        text_val = response.get("content", "")
        text = text_val if isinstance(text_val, str) else ""
    else:
        return EvaluateResult(
            score=0.0,
            reason="Last message is of unexpected type.",
            metrics={
                "cosine_length": MetricResult(
                    score=0.0,
                    is_score_valid=False,
                    reason="Invalid message type in messages.",
                )
            },
        )

    token_count = count_tokens(text, method=token_method)

    solution_is_correct = False
    if is_correct is not None:
        solution_is_correct = is_correct
    elif correctness is not None:
        solution_is_correct = correctness >= 0.9

    progress = min(1.0, token_count / max_length)
    cosine_factor = math.cos(progress * math.pi)

    if solution_is_correct:
        min_value = min_value_correct
        max_value = max_value_correct
    else:
        min_value = max_value_wrong
        max_value = min_value_wrong

    score = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine_factor)

    if solution_is_correct:
        success = True
        reason = f"Correct solution with length penalty: {token_count} tokens"
    else:
        success = False
        reason = f"Incorrect solution with length consideration: {token_count} tokens"

    detailed_reason = (
        f"Length-based {'reward' if solution_is_correct else 'penalty'}: "
        f"{token_count}/{max_length} tokens, cosine factor: {cosine_factor:.2f}"
    )

    metrics = {
        "cosine_length": MetricResult(
            score=score,
            is_score_valid=success,
            reason=detailed_reason,  # Use detailed_reason here
        ),
        "token_count": MetricResult(
            score=min(1.0, float(token_count) / max_length),
            is_score_valid=success,
            reason=f"Token count: {token_count}/{max_length}",
        ),
        "correctness": MetricResult(
            score=1.0 if solution_is_correct else 0.0,
            is_score_valid=solution_is_correct,
            reason=f"Solution is {'correct' if solution_is_correct else 'incorrect'}",
        ),
    }

    return EvaluateResult(score=score, reason=reason, metrics=metrics)
