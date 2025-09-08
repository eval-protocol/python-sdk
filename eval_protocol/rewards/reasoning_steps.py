"""
Reward functions for evaluating reasoning steps.

This module provides reward functions that evaluate whether a model's response
contains adequate step-by-step reasoning, rewarding structured thinking.
"""

import re
from typing import Any, Dict, List, Optional, Pattern, Set, Union

from ..models import EvaluateResult, Message, MetricResult
from ..typed_interface import reward_function


@reward_function
def reasoning_steps_reward(
    messages: List[Message],
    pattern: Optional[str] = None,
    min_steps: int = 3,
    max_steps: Optional[int] = None,
    exclusive_patterns: bool = False,
    **kwargs: Any,
) -> EvaluateResult:
    """
    Reward function that evaluates step-by-step reasoning in model responses.

    This function checks if the model's response contains indicators of structured
    reasoning, such as numbered steps, bullet points, or transitional phrases.

    Args:
        messages: List of conversation messages
        pattern: Optional custom regex pattern to use for detecting reasoning steps
        min_steps: Minimum number of steps required for full score
        max_steps: Optional maximum number of steps (default: None)
        exclusive_patterns: Whether to use only the custom pattern (True) or
                           combine it with default patterns (False)
        **kwargs: Additional arguments

    Returns:
        EvaluateResult with score based on the number of reasoning steps detected
    """
    if not messages or len(messages) == 0:
        return EvaluateResult(
            score=0.0,
            reason="No messages provided",
            metrics={"reasoning_steps": MetricResult(score=0.0, is_score_valid=False, reason="No messages provided")},
        )

    response = messages[-1]

    if response.role != "assistant" or not response.content:
        return EvaluateResult(
            score=0.0,
            reason="No assistant response found or response has no content",
            metrics={
                "reasoning_steps": MetricResult(
                    score=0.0,
                    is_score_valid=False,
                    reason="Message not from assistant or has no content",
                )
            },
        )
    text: str = response.content if isinstance(response.content, str) else ""

    # Default patterns for detecting reasoning steps
    default_patterns = [
        r"Step\s+\d+[:.]\s+",
        r"^\s*\d+\.\s+",
        r"\n\s*\d+\.\s+",
        r"\n\s*-\s+",
        r"\n\s*\*\s+",
        r"\b(?:First|Second|Third|Fourth|Fifth|Next|Then|Finally)[,:]",
        r"\b(?:Let's|I will|To solve this|To begin)[,:]",
    ]

    patterns_to_use = []
    if pattern and exclusive_patterns:
        patterns_to_use = [pattern]
    elif pattern:
        patterns_to_use = [pattern] + default_patterns
    else:
        patterns_to_use = default_patterns

    combined_pattern = "|".join(f"(?:{p})" for p in patterns_to_use)
    matches = re.findall(combined_pattern, text, re.MULTILINE)
    num_steps = len(matches)

    if num_steps == 0:
        score = 0.0
    elif max_steps is not None:
        score = min(
            1.0,
            max(0.0, (num_steps - min_steps + 1) / (max_steps - min_steps + 1)),
        )
    else:
        score = min(1.0, num_steps / min_steps)

    success = num_steps >= min_steps
    step_metrics = {}

    explicit_steps = len(re.findall(r"Step\s+\d+[:.]\s+", text, re.MULTILINE))
    if explicit_steps > 0:
        step_metrics["explicit_steps"] = MetricResult(
            score=min(1.0, explicit_steps / min_steps),
            is_score_valid=explicit_steps >= min_steps,
            reason=f"Found {explicit_steps} explicit steps",
        )

    numbered_lists = len(re.findall(r"(?:^|\n)\s*\d+\.\s+", text, re.MULTILINE))
    if numbered_lists > 0:
        step_metrics["numbered_lists"] = MetricResult(
            score=min(1.0, numbered_lists / min_steps),
            is_score_valid=numbered_lists >= min_steps,
            reason=f"Found {numbered_lists} numbered list items",
        )

    bullets = len(re.findall(r"(?:^|\n)\s*[-*]\s+", text, re.MULTILINE))
    if bullets > 0:
        step_metrics["bullet_points"] = MetricResult(
            score=min(1.0, bullets / min_steps),
            is_score_valid=bullets >= min_steps,
            reason=f"Found {bullets} bullet points",
        )

    transitions = len(
        re.findall(
            r"\b(?:First|Second|Third|Next|Then|Finally)[,:]",
            text,
            re.MULTILINE,
        )
    )
    if transitions > 0:
        step_metrics["transition_phrases"] = MetricResult(
            score=min(1.0, transitions / min_steps),
            is_score_valid=transitions >= min_steps,
            reason=f"Found {transitions} transition phrases",
        )

    metrics = {
        "reasoning_steps": MetricResult(
            score=score,
            is_score_valid=success,
            reason=f"Found {num_steps} reasoning steps (minimum required: {min_steps})",
        ),
        **step_metrics,
    }

    reason = f"Detected {num_steps} reasoning steps (required: {min_steps})"
    if max_steps:
        reason += f", max: {max_steps}"

    return EvaluateResult(score=score, reason=reason, metrics=metrics)


@reward_function
def sequence_reward(
    messages: List[Message],
    sequence_terms: Optional[List[str]] = None,
    min_matches: int = 3,
    case_sensitive: bool = False,
    **kwargs: Any,
) -> EvaluateResult:
    """
    Reward function that evaluates sequential reasoning in model responses.

    This function checks if the model's response follows a specific sequence
    of reasoning steps or includes a minimum number of required terms in order.

    Args:
        messages: List of conversation messages
        sequence_terms: List of terms that should appear in sequence
        min_matches: Minimum number of sequence terms required for full score
        case_sensitive: Whether matching should be case-sensitive
        **kwargs: Additional arguments

    Returns:
        EvaluateResult with score based on sequence matching
    """
    if not messages or len(messages) == 0:
        return EvaluateResult(
            score=0.0,
            reason="No messages provided",
            metrics={
                "sequence_reasoning": MetricResult(score=0.0, is_score_valid=False, reason="No messages provided")
            },
        )

    response = messages[-1]

    if response.role != "assistant" or not response.content:
        return EvaluateResult(
            score=0.0,
            reason="No assistant response found or response has no content",
            metrics={
                "sequence_reasoning": MetricResult(
                    score=0.0,
                    is_score_valid=False,
                    reason="Message not from assistant or has no content",
                )
            },
        )
    text: str = response.content if isinstance(response.content, str) else ""

    if not sequence_terms:
        sequence_terms = [
            "First",
            "Second",
            "Third",
            "Fourth",
            "Fifth",
            "Next",
            "Then",
            "Finally",
            "Conclusion",
        ]

    found_terms = []
    last_position = -1

    if not case_sensitive:
        text = text.lower()
        sequence_terms = [term.lower() for term in sequence_terms]

    for term in sequence_terms:
        position = text.find(term, last_position + 1)
        if position > last_position:
            found_terms.append(term)
            last_position = position

    num_matches = len(found_terms)
    score = min(1.0, num_matches / min_matches)
    success = num_matches >= min_matches

    metrics = {
        "sequence_reasoning": MetricResult(
            score=score,
            is_score_valid=success,
            reason=f"Found {num_matches} sequential terms (minimum required: {min_matches})",
        ),
        "sequential_terms_found": MetricResult(
            score=score,
            is_score_valid=success,
            reason=f"Sequential terms found: {', '.join(found_terms)}",
        ),
    }

    reason = f"Detected {num_matches} sequential reasoning terms in order (required: {min_matches})"

    return EvaluateResult(score=score, reason=reason, metrics=metrics)
