"""
Reward functions for evaluating repetition in model responses.

This module provides reward functions that penalize repetitive text in model responses,
encouraging more diverse and information-rich outputs.
"""

import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from ..models import EvaluateResult, Message, MetricResult, ChatCompletionContentPartTextParam


def _to_text(content: Optional[Union[str, List[ChatCompletionContentPartTextParam]]]) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return "\n".join(part.text for part in content)
    except Exception:
        return ""


from ..typed_interface import reward_function


def get_ngrams(text: str, n: int, language: str = "en") -> Tuple[List[Tuple[str, ...]], int]:
    """
    Extract n-grams from text based on language.

    Args:
        text: The text to extract n-grams from
        n: Size of the n-grams
        language: Language of the text (affects tokenization)

    Returns:
        Tuple of (list of n-grams, total n-gram count)
    """
    if language == "en":
        words = text.lower().split()
    elif language == "zh":
        try:
            import jieba

            words = list(jieba.cut(text))
        except ImportError:
            words = list(text)
    else:
        words = text.lower().split()

    ngrams = []
    for i in range(len(words) - n + 1):
        ngrams.append(tuple(words[i : i + n]))

    return ngrams, len(ngrams)


@reward_function
def repetition_penalty_reward(
    messages: Union[List[Message], List[Dict[str, Any]]],
    ground_truth: Optional[
        Union[List[Message], List[Dict[str, Any]]]
    ] = None,  # Not used by this function but part of standard signature
    ngram_size: int = 3,
    max_penalty: float = 0.5,
    language: str = "en",
    **kwargs: Any,
) -> EvaluateResult:
    """
    Reward function that penalizes repetitive text in model responses.
    The model's response is assumed to be the last message in the `messages` list.

    This function computes repetition by examining unique n-grams in the response
    and penalizes texts with a high proportion of repeated phrases.

    Args:
        messages: List of conversation messages, where `messages[-1]` is the model's response.
        ground_truth: Optional. Expected assistant response trajectory. Not directly used by this reward.
        ngram_size: Size of n-grams to check for repetition.
        max_penalty: Maximum penalty to apply for repetitive text.
        language: Language of the text (affects tokenization).
        **kwargs: Additional arguments.

    Returns:
        EvaluateResult with score penalizing repetition
    """
    if not messages or len(messages) == 0:
        return EvaluateResult(
            score=0.0,
            reason="No messages provided",
            metrics={"repetition": MetricResult(score=0.0, is_score_valid=False, reason="No messages provided")},
        )

    response = messages[-1]

    if isinstance(response, Message):
        if response.role != "assistant":
            return EvaluateResult(
                score=0.0,
                reason="No assistant response found",
                metrics={
                    "repetition": MetricResult(
                        score=0.0,
                        is_score_valid=False,
                        reason="Message not from assistant",
                    )
                },
            )
        text = _to_text(response.content)
    elif isinstance(response, dict):
        if response.get("role") != "assistant":
            return EvaluateResult(
                score=0.0,
                reason="No assistant response found",
                metrics={
                    "repetition": MetricResult(
                        score=0.0,
                        is_score_valid=False,
                        reason="Message not from assistant",
                    )
                },
            )
        text = response.get("content", "")
    else:
        return EvaluateResult(
            score=0.0,
            reason="Last message is of unexpected type.",
            metrics={
                "repetition": MetricResult(
                    score=0.0,
                    is_score_valid=False,
                    reason="Invalid message type in messages.",
                )
            },
        )

    if not text.strip():
        return EvaluateResult(
            score=1.0,
            reason="Empty response, no repetition to penalize",
            metrics={
                "repetition": MetricResult(
                    score=1.0,
                    is_score_valid=True,
                    reason="Empty response",
                ),
                "unique_ngram_ratio": MetricResult(
                    score=1.0,
                    is_score_valid=True,
                    reason="Empty response",
                ),
                "repetition_penalty": MetricResult(
                    score=1.0,
                    is_score_valid=True,
                    reason="No penalty applied to empty response",
                ),
            },
        )

    ngrams, total = get_ngrams(text, ngram_size, language)

    if total < 1:
        return EvaluateResult(
            score=1.0,
            reason=f"Text too short for {ngram_size}-gram analysis",
            metrics={
                "repetition": MetricResult(
                    score=1.0,
                    is_score_valid=True,
                    reason=f"Text too short for {ngram_size}-gram analysis",
                )
            },
        )

    unique_ngrams = len(set(ngrams))
    repetition_ratio = 1.0 - (unique_ngrams / total)
    penalty = repetition_ratio * max_penalty
    score = max(0.0, 1.0 - penalty)
    success = repetition_ratio < 0.2

    reason = f"Repetition ratio: {repetition_ratio:.2f}, Unique {ngram_size}-grams: {unique_ngrams}/{total}"
    metrics = {
        "repetition": MetricResult(score=score, is_score_valid=success, reason=reason),
        "unique_ngram_ratio": MetricResult(
            score=1.0 - repetition_ratio,
            is_score_valid=success,
            reason=f"Unique {ngram_size}-gram ratio: {1.0 - repetition_ratio:.2f}",
        ),
        "repetition_penalty": MetricResult(
            score=1.0 - penalty,
            is_score_valid=success,
            reason=f"Applied repetition penalty: {penalty:.2f}",
        ),
    }

    return EvaluateResult(score=score, reason=reason, metrics=metrics, is_score_valid=score > 0.0)


@reward_function
def diversity_reward(
    messages: Union[List[Message], List[Dict[str, Any]]],
    ground_truth: Optional[
        Union[List[Message], List[Dict[str, Any]]]
    ] = None,  # Not used by this function but part of standard signature
    ngram_sizes: List[int] = [1, 2, 3],
    weights: Optional[List[float]] = None,
    language: str = "en",
    **kwargs: Any,
) -> EvaluateResult:
    """
    Reward function that measures lexical diversity in model responses.
    The model's response is assumed to be the last message in the `messages` list.

    This function computes diversity across multiple n-gram sizes and combines them
    into a weighted score to encourage varied vocabulary and phrasing.

    Args:
        messages: List of conversation messages, where `messages[-1]` is the model's response.
        ground_truth: Optional. Expected assistant response trajectory. Not directly used by this reward.
        ngram_sizes: List of n-gram sizes to evaluate.
        weights: Optional list of weights for each n-gram size (normalized if provided).
        language: Language of the text (affects tokenization).
        **kwargs: Additional arguments.

    Returns:
        EvaluateResult with score based on lexical diversity
    """
    if not messages or len(messages) == 0:
        return EvaluateResult(
            score=0.0,
            reason="No messages provided",
            metrics={"diversity": MetricResult(score=0.0, is_score_valid=False, reason="No messages provided")},
        )

    response = messages[-1]

    if isinstance(response, Message):
        if response.role != "assistant":
            return EvaluateResult(
                score=0.0,
                reason="No assistant response found",
                metrics={
                    "diversity": MetricResult(
                        score=0.0,
                        is_score_valid=False,
                        reason="Message not from assistant",
                    )
                },
            )
        text = _to_text(response.content)
    elif isinstance(response, dict):
        if response.get("role") != "assistant":
            return EvaluateResult(
                score=0.0,
                reason="No assistant response found",
                metrics={
                    "diversity": MetricResult(
                        score=0.0,
                        is_score_valid=False,
                        reason="Message not from assistant",
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
                "diversity": MetricResult(
                    score=0.0,
                    is_score_valid=False,
                    reason="Invalid message type in messages.",
                )
            },
        )

    if not text.strip():
        return EvaluateResult(
            score=0.0,
            reason="Empty response",
            metrics={
                "diversity": MetricResult(
                    score=0.0,
                    is_score_valid=False,
                    reason="Empty response",
                )
            },
        )

    if weights is None:
        weights = [0.2, 0.3, 0.5][: len(ngram_sizes)]

    if len(weights) != len(ngram_sizes):
        if len(weights) > len(ngram_sizes):
            weights = weights[: len(ngram_sizes)]
        else:
            missing_weight = (
                (1.0 - sum(weights)) / (len(ngram_sizes) - len(weights))
                if (len(ngram_sizes) - len(weights)) > 0
                else 0
            )
            weights.extend([missing_weight] * (len(ngram_sizes) - len(weights)))

    total_weight = sum(weights)
    if total_weight != 1.0 and total_weight > 0:  # Avoid division by zero if total_weight is 0
        weights = [w / total_weight for w in weights]
    elif total_weight == 0 and len(weights) > 0:  # If all weights are zero, distribute equally
        weights = [1.0 / len(weights)] * len(weights)

    diversity_scores = {}
    ratios = {}

    for size, weight in zip(ngram_sizes, weights):
        ngrams, total = get_ngrams(text, size, language)

        if total < 1:
            diversity_scores[f"ngram_{size}"] = 1.0
            ratios[f"ngram_{size}"] = 1.0
            continue

        unique_count = len(set(ngrams))
        ratio = unique_count / total
        diversity_scores[f"ngram_{size}"] = ratio * weight
        ratios[f"ngram_{size}"] = ratio

    final_score = sum(diversity_scores.values())
    success = final_score > 0.6

    size_metric_items: List[Tuple[str, MetricResult]] = []
    for size_key, ratio_val in ratios.items():
        metric_for_size = MetricResult(
            score=ratio_val,
            is_score_valid=ratio_val > 0.7,
            reason=f"Diversity ratio for {size_key}: {ratio_val:.2f}",
        )
        size_metric_items.append((size_key, metric_for_size))

    size_metrics: Dict[str, MetricResult] = dict(size_metric_items)

    metrics: Dict[str, MetricResult] = {
        "diversity": MetricResult(
            score=final_score,
            is_score_valid=success,
            reason=f"Overall weighted diversity score: {final_score:.2f}",
        ),
        **size_metrics,
    }

    return EvaluateResult(
        score=final_score,
        reason=f"Lexical diversity score: {final_score:.2f}",
        metrics=metrics,
    )
