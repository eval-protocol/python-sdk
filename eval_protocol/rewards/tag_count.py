"""
Reward functions for counting tags in text responses.

This module provides reward functions that evaluate if responses contain
specified XML/HTML-like tags in correct quantities.
"""

import re
from typing import Any, Dict, List, Set, Union

from ..models import EvaluateResult, Message, MetricResult, ChatCompletionContentPartTextParam


def _to_text(content: Union[str, List[ChatCompletionContentPartTextParam], None]) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return "\n".join(part.text for part in content)
    except Exception:
        return ""


from ..typed_interface import reward_function


@reward_function  # type: ignore[arg-type]
def tag_count_reward(
    messages: List[Message],
    *,  # Make subsequent parameters keyword-only
    required_tags: List[str],
    score_per_tag: float = 0.25,
    require_balanced: bool = True,
    **kwargs: Any,
) -> EvaluateResult:
    """
    Reward function that checks for presence of specific tags in response.

    For each tag found in required_tags, adds score_per_tag to the score.
    Optionally requires tags to be balanced (equal opening and closing tags).

    Args:
        messages: List of conversation messages
        required_tags: List of tag names to check for (without < > brackets)
        score_per_tag: Score to award per correctly found tag (default: 0.25)
        require_balanced: If True, requires equal opening and closing tags
        **kwargs: Additional arguments

    Returns:
        EvaluateResult with score based on tags found
    """
    if not messages or len(messages) == 0:
        return EvaluateResult(
            score=0.0,
            reason="No messages provided",
            metrics={"tag_count": MetricResult(score=0.0, is_score_valid=False, reason="No messages provided")},
        )

    response = messages[-1]

    if response.role != "assistant" or response.content is None:
        return EvaluateResult(
            score=0.0,
            reason="No assistant response found or response has no content",
            metrics={
                "tag_count": MetricResult(
                    score=0.0,
                    is_score_valid=False,
                    reason="Message not from assistant or has no content",
                )
            },
        )
    text: str = _to_text(response.content)

    tag_metrics = {}
    found_tags: Set[str] = set()
    mismatched_tags: Set[str] = set()
    total_found = 0

    for tag in required_tags:
        opening_pattern = f"<{tag}[^>]*>"
        closing_pattern = f"</{tag}>"

        opening_count = len(re.findall(opening_pattern, text))
        closing_count = len(re.findall(closing_pattern, text))

        if require_balanced:
            is_found = opening_count > 0 and closing_count > 0 and opening_count == closing_count
        else:
            is_found = opening_count > 0 or closing_count > 0

        is_balanced = opening_count == closing_count

        if is_found:
            found_tags.add(tag)
            total_found += 1

        if require_balanced and not is_balanced and (opening_count > 0 or closing_count > 0):
            mismatched_tags.add(tag)

        if require_balanced:
            tag_score = 1.0 if (opening_count > 0 and closing_count > 0 and is_balanced) else 0.0
            tag_success = opening_count > 0 and closing_count > 0 and is_balanced
        else:
            has_tags = opening_count > 0 or closing_count > 0
            tag_score = 1.0 if has_tags else 0.0
            tag_success = opening_count > 0 or closing_count > 0

        tag_metrics[f"tag_{tag}"] = MetricResult(
            score=tag_score,
            is_score_valid=tag_success,
            reason=_get_tag_reason(tag, opening_count, closing_count, require_balanced),
        )

    total_score = min(1.0, len(found_tags) * score_per_tag)

    if require_balanced and mismatched_tags:
        penalty = len(mismatched_tags) * score_per_tag
        total_score = max(0.0, total_score - penalty)

    success = len(found_tags) == len(required_tags) and (not require_balanced or not mismatched_tags)

    reason = _get_overall_reason(required_tags, found_tags, mismatched_tags, require_balanced)
    tag_metrics["overall"] = MetricResult(score=total_score, is_score_valid=success, reason=reason)

    return EvaluateResult(score=total_score, reason=reason, metrics=tag_metrics, is_score_valid=success)


def _get_tag_reason(tag: str, opening_count: int, closing_count: int, require_balanced: bool) -> str:
    """Generate a descriptive reason for a tag's evaluation."""
    if opening_count == 0 and closing_count == 0:
        return f"Tag '{tag}' not found in response"
    elif opening_count > 0 and closing_count == 0:
        return f"Found {opening_count} opening <{tag}> tag(s) but no closing"
    elif opening_count == 0 and closing_count > 0:
        return f"Found {closing_count} closing </{tag}> tag(s) but no opening"
    elif opening_count == closing_count:
        return f"Found {opening_count} balanced '{tag}' tag(s)"
    else:
        if require_balanced:
            return f"Unbalanced tags: {opening_count} opening vs {closing_count} closing '{tag}' tags"
        else:
            return f"Found '{tag}' tags (unbalanced: {opening_count} opening, {closing_count} closing)"


def _get_overall_reason(
    required_tags: List[str],
    found_tags: Set[str],
    mismatched_tags: Set[str],
    require_balanced: bool,
) -> str:
    """Generate an overall reason for the evaluation."""
    if not found_tags:
        return "No required tags found in response"

    missing_tags = set(required_tags) - found_tags

    if not missing_tags and not mismatched_tags:
        return f"All {len(required_tags)} required tags found and balanced"

    reason_parts = []

    if found_tags:
        reason_parts.append(f"Found {len(found_tags)}/{len(required_tags)} required tags")

    if missing_tags:
        tags_str = ", ".join([f"'{tag}'" for tag in missing_tags])
        reason_parts.append(f"Missing tags: {tags_str}")

    if require_balanced and mismatched_tags:
        tags_str = ", ".join([f"'{tag}'" for tag in mismatched_tags])
        reason_parts.append(f"Unbalanced tags: {tags_str}")

    return ". ".join(reason_parts)
