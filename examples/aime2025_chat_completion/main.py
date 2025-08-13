"""
Eval Protocol example: AIME2025 chat completion evaluation

This example mirrors gpt-oss's AIME 2025 evaluation using OpenAI-compatible
chat completions. It evaluates whether the assistant's final answer matches the
ground-truth integer, extracting answers from \\boxed{...} or fallback digits.
"""

import re
from typing import Any, Dict, List, Optional, Union

from eval_protocol import EvaluateResult, MetricResult, reward_function
from eval_protocol.models import Message


def _extract_boxed_text(text: str) -> str:
    """
    Extract the last occurrence of a boxed answer (\\boxed{...} or \\framebox{...}).
    If none found, fall back to the last integer found in the text.
    """
    if not text:
        return ""

    pattern_boxed = r"boxed{(.*?)}|framebox{(.*?)}"
    matches = re.findall(pattern_boxed, text, re.DOTALL)
    if matches:
        # Iterate from the end to prioritize the final boxed answer
        for match in matches[::-1]:
            for group in match:
                if group:
                    return group.split(",")[-1].strip()

    # Fallback: last integer in the text
    matches_digits = re.findall(r"\d+", text, re.DOTALL)
    if matches_digits:
        return matches_digits[-1]
    return ""


def _normalize_to_int_or_none(s: str) -> Optional[int]:
    if s is None:
        return None
    # Only take leading digits
    m = re.match(r"\d+", str(s).strip())
    if not m:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None


@reward_function(id="aime2025_exact_match")
def evaluate(
    messages: Union[List[Message], List[Dict[str, Any]]],
    ground_truth: Optional[str] = None,
    **kwargs,
) -> EvaluateResult:
    """
    Score 1.0 if extracted final answer equals the ground-truth integer, else 0.0.
    """
    if not messages:
        return EvaluateResult(
            score=0.0,
            reason="No messages provided",
            is_score_valid=False,
            metrics={"parse_status": MetricResult(score=0.0, is_score_valid=False, reason="empty messages")},
        )

    last_msg = messages[-1]
    content = last_msg["content"] if isinstance(last_msg, dict) else (last_msg.content or "")

    extracted_text = _extract_boxed_text(content)
    extracted_int = _normalize_to_int_or_none(extracted_text)
    gt_int = _normalize_to_int_or_none(ground_truth if ground_truth is not None else "")

    is_valid = extracted_int is not None and gt_int is not None
    score = 1.0 if (is_valid and extracted_int == gt_int) else 0.0

    metrics: Dict[str, MetricResult] = {
        "exact_match": MetricResult(
            score=score,
            is_score_valid=is_valid,
            reason=(
                "Parsed both integers and they matched"
                if score == 1.0
                else (
                    "Parsed integers did not match"
                    if is_valid
                    else "Failed to parse integer from prediction or ground truth"
                )
            ),
            data={
                "extracted_text": extracted_text,
                "extracted_int": extracted_int,
                "ground_truth_int": gt_int,
            },
        )
    }

    return EvaluateResult(
        score=score,
        reason=("Answer correct" if score == 1.0 else "Answer incorrect"),
        is_score_valid=is_valid,
        metrics=metrics,
    )
