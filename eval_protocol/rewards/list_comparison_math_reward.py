"""
Reward function for comparing lists of numbers, often found in math answers
like sets of divisors, roots, etc.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..models import EvaluateResult, Message, MetricResult
from ..typed_interface import reward_function


def parse_number_list_from_string(s: str) -> Optional[List[float]]:
    """
    Parses a string potentially containing a comma-separated list of numbers.
    Handles integers and simple decimals.
    e.g., "1, 2, 3.5, 4" -> [1.0, 2.0, 3.5, 4.0]
    """
    numbers = []
    s = s.replace("$", "").strip()
    parts = re.split(r"\s*,\s*", s)
    if not parts or not any(p.strip() for p in parts):
        return None

    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            numbers.append(float(part))
        except ValueError:
            return None
    return numbers if numbers else None


def extract_number_list(text: str) -> List[List[float]]:
    """
    Extracts lists of numbers from text.
    Prioritizes content within \\boxed{} or $...$.
    If multiple such expressions exist, each valid list is returned.
    If no such delimiters, tries to parse the whole text.

    Args:
        text: The text to extract number lists from.

    Returns:
        A list of extracted number lists. Each inner list contains floats.
        Example: "\\boxed{1,2,3}, $4,5$" -> [[1.0, 2.0, 3.0], [4.0, 5.0]]
    """
    extracted_lists: List[List[float]] = []

    # Priority 1: Boxed LaTeX expressions
    boxed_contents = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", text)
    if boxed_contents:
        for content in boxed_contents:
            parsed_list = parse_number_list_from_string(content)
            if parsed_list:
                extracted_lists.append(parsed_list)
        if extracted_lists:
            return extracted_lists

    # Priority 2: Content within $...$ or $$...$$
    dollar_contents = re.findall(r"\$\$(.*?)\$\$|\$(.*?)\$", text, re.DOTALL)
    if dollar_contents:
        for group_match in dollar_contents:
            content = group_match[0] if group_match[0] else group_match[1]
            if content:
                parsed_list = parse_number_list_from_string(content.strip())
                if parsed_list:
                    extracted_lists.append(parsed_list)
        if extracted_lists:
            return extracted_lists

    # Priority 3: Try parsing the whole text as a list if no delimiters found
    # This is a fallback and might be less reliable.
    if not extracted_lists:
        full_text_parsed_list = parse_number_list_from_string(text)
        if full_text_parsed_list:
            extracted_lists.append(full_text_parsed_list)

    return extracted_lists


@reward_function  # type: ignore[arg-type]
def list_comparison_math_reward(
    messages: List[Message],
    *,
    ground_truth: str,
    order_matters: bool = False,
    **kwargs: Any,
) -> EvaluateResult:
    """
    Evaluate answers that are lists/sets of numbers.

    Extracts lists of numbers from the model's response (messages[-1].content)
    and the ground_truth string, then compares them.
    By default, order does not matter (set comparison).

    Args:
        messages: List of conversation messages. The last message is the assistant's response.
        ground_truth: String representation of the expected list of numbers.
        order_matters: If True, compares lists directly (order and count matter).
                       If False (default), compares as sets (order and duplicates
                       within a list don't matter beyond presence).
        **kwargs: Additional keyword arguments.

    Returns:
        EvaluateResult with score and metrics.
    """
    metrics: Dict[str, MetricResult] = {}

    if (
        not messages
        or not isinstance(messages[-1], Message)
        or messages[-1].role != "assistant"
        or messages[-1].content is None
    ):
        return EvaluateResult(
            score=0.0,
            reason="Invalid or missing assistant response in messages.",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    is_score_valid=False,
                    reason="Last message not a valid assistant response.",
                )
            },
        )

    gen_content_raw = messages[-1].content
    gen_content = (
        gen_content_raw
        if isinstance(gen_content_raw, str)
        else "".join([getattr(p, "text", "") for p in (gen_content_raw or [])])
    )
    orig_content = ground_truth

    if not gen_content:
        return EvaluateResult(
            score=0.0,
            reason="Assistant response content is empty.",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    is_score_valid=False,
                    reason="Empty generated message content.",
                )
            },
        )
    if not orig_content:
        return EvaluateResult(
            score=0.0,
            reason="Ground truth string (expected list) is empty.",
            metrics={"error": MetricResult(score=0.0, is_score_valid=False, reason="Empty ground truth string.")},
        )

    gen_lists = extract_number_list(gen_content)
    orig_lists = extract_number_list(orig_content)

    metrics["extracted_original_lists"] = MetricResult(
        score=1.0 if orig_lists else 0.0,
        is_score_valid=bool(orig_lists),
        reason=f"Original lists: {orig_lists}",
    )
    metrics["extracted_generated_lists"] = MetricResult(
        score=1.0 if gen_lists else 0.0,
        is_score_valid=bool(gen_lists),
        reason=f"Generated lists: {gen_lists}",
    )

    if not orig_lists:
        return EvaluateResult(
            score=0.0,
            reason="Could not extract any number list from original message (ground truth).",
            metrics=metrics,
        )
    if not gen_lists:
        return EvaluateResult(
            score=0.0,
            reason="Could not extract any number list from generated message.",
            metrics=metrics,
        )

    # For simplicity, compare the first valid list found in each.
    # Future improvement: handle multiple lists (e.g., if solution has multiple boxed lists)
    orig_list_to_compare = orig_lists[0]
    gen_list_to_compare = gen_lists[0]

    score = 0.0
    match_reason = ""

    if order_matters:
        # Note: To be robust against float precision, comparison element-wise with tolerance might be needed.
        if gen_list_to_compare == orig_list_to_compare:
            score = 1.0
            match_reason = (
                f"Exact list match (order matters). Gen: {gen_list_to_compare} vs Orig: {orig_list_to_compare}"
            )
        else:
            score = 0.0
            match_reason = f"List mismatch (order matters). Gen: {gen_list_to_compare} vs Orig: {orig_list_to_compare}"
    else:
        # Note: float precision can be an issue with sets. A more robust set comparison would involve tolerance.
        gen_set = set(gen_list_to_compare)
        orig_set = set(orig_list_to_compare)

        if gen_set == orig_set:
            score = 1.0
            match_reason = (
                f"Set match (order does not matter). Gen: {sorted(list(gen_set))} vs Orig: {sorted(list(orig_set))}"
            )
        else:
            score = 0.0
            missing_in_gen = orig_set - gen_set
            extra_in_gen = gen_set - orig_set
            match_reason_parts = [
                f"Set mismatch (order does not matter). Gen: {sorted(list(gen_set))} vs Orig: {sorted(list(orig_set))}."
            ]
            if missing_in_gen:
                match_reason_parts.append(f"Missing in generated: {sorted(list(missing_in_gen))}.")
            if extra_in_gen:
                match_reason_parts.append(f"Extra in generated: {sorted(list(extra_in_gen))}.")
            match_reason = " ".join(match_reason_parts)

    metrics["list_comparison"] = MetricResult(score=score, is_score_valid=(score == 1.0), reason=match_reason)
    return EvaluateResult(score=score, reason=match_reason, metrics=metrics)
