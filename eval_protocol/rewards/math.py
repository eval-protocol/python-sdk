"""
Math reward function for evaluating mathematical answer correctness.

This module provides functions to evaluate the correctness of mathematical
answers by extracting numerical values from text using regex patterns and
comparing them with expected answers.
"""

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union, cast

from ..models import EvaluateResult, Message, MetricResult
from ..typed_interface import reward_function

# Types used throughout this module to clearly express allowed answer values.
# Include both float and int since extraction may yield either at analysis time.
Numeric = Union[int, float]
AnswerValue = Union[Numeric, str]

_ALGEBRAIC_VARS_SET: Set[str] = {
    "x",
    "y",
    "z",
    "a",
    "b",
    "c",
    "n",
    "t",
    "q",
    "p",
    "r",
    "u",
    "v",
    "w",
}


def _parse_numeric_string(s: str) -> Optional[float]:
    s = s.strip()
    try:
        if re.fullmatch(r"-?\d+(\.\d+)?", s):
            return float(s)
        m_frac = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)", s)
        if m_frac:
            num = float(m_frac.group(1))
            den = float(m_frac.group(2))
            return num / den if den != 0 else None
    except (ValueError, ZeroDivisionError):
        return None
    return None


def _is_coefficient(
    text_content: str,
    match_obj: re.Match,
    num_group_idx: int = 1,
    unit_group_idx: Optional[int] = None,
) -> bool:
    """
    Checks if a number identified by match_obj in text_content is likely a coefficient.
    """
    unit_candidate = ""
    if unit_group_idx is not None and len(match_obj.groups()) >= unit_group_idx and match_obj.group(unit_group_idx):
        unit_candidate = match_obj.group(unit_group_idx).strip()

    if unit_candidate and len(unit_candidate) == 1 and unit_candidate.lower() in _ALGEBRAIC_VARS_SET:
        return True

    idx_after_num_str = match_obj.end(num_group_idx)

    if idx_after_num_str < len(text_content) and text_content[idx_after_num_str].lower() in _ALGEBRAIC_VARS_SET:
        if idx_after_num_str + 1 == len(text_content) or not text_content[idx_after_num_str + 1].isalnum():
            return True

    if (
        idx_after_num_str + 1 < len(text_content)
        and text_content[idx_after_num_str] == " "
        and text_content[idx_after_num_str + 1].lower() in _ALGEBRAIC_VARS_SET
    ):
        if idx_after_num_str + 2 == len(text_content) or not text_content[idx_after_num_str + 2].isalnum():
            return True
    return False


def _extract_html_tag_answers(text: str) -> List[Tuple[str, AnswerValue]]:
    """Extracts answers from <answer> or <ans> HTML-like tags."""
    html_tag_answers: List[Tuple[str, AnswerValue]] = []
    tag_re = re.compile(
        r"<(?P<tag>answer|ans)\b[^>]*>(?P<inner>.*?)</(?P=tag)>",
        re.IGNORECASE | re.DOTALL,
    )
    for m in tag_re.finditer(text):
        raw = m.group(0)
        inner = m.group("inner").strip()
        inner = re.sub(r"^\$+|^\(+|^\[+|(\$|\)|\])+?$", "", inner).strip()

        val = _parse_numeric_string(inner)
        if val is not None:
            html_tag_answers.append((raw, val))
            continue

        m_frac = re.fullmatch(r"\\frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}", inner)
        if m_frac:
            try:
                num, den = float(m_frac.group(1)), float(m_frac.group(2))
                if den != 0:
                    html_tag_answers.append((raw, num / den))
                    continue
            except (ValueError, ZeroDivisionError):
                pass

        sci = re.fullmatch(r"([-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?)", inner)
        if sci:
            try:
                cleaned = sci.group(1).replace(",", "")
                html_tag_answers.append((raw, float(cleaned)))
                continue
            except ValueError:
                pass

        m_num_unit = re.fullmatch(r"(-?\d+(?:\.\d+)?)[ ]*[a-zA-Z%]+", inner)
        if m_num_unit:
            try:
                html_tag_answers.append((raw, float(m_num_unit.group(1))))
                continue
            except ValueError:
                pass
    return html_tag_answers


def _extract_boxed_latex_answers(
    text: str,
) -> Tuple[List[Tuple[str, AnswerValue]], bool]:
    """
    Extracts answers from \\boxed{} LaTeX expressions.
    Returns a tuple: (list of answers, boolean indicating if any boxed expr was found).
    """
    boxed_answers: List[Tuple[str, AnswerValue]] = []
    found_any_boxed_expr = False
    for m_boxed in re.finditer(r"\\boxed\s*\{\s*((?:[^{}]|\{[^{}]*\})*?)\s*\}", text):
        found_any_boxed_expr = True
        original_boxed_expr = m_boxed.group(0)
        content = m_boxed.group(1).strip()

        if not content:
            continue

        if " or " in content.lower():
            boxed_answers.append((original_boxed_expr, content))
            continue
        if re.fullmatch(r"[A-Ea-e]", content):
            boxed_answers.append((original_boxed_expr, content.upper()))
            continue

        m_latex_frac = re.fullmatch(r"\\frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}", content)
        if m_latex_frac:
            try:
                num = float(m_latex_frac.group(1))
                den = float(m_latex_frac.group(2))
                if den != 0:
                    boxed_answers.append((original_boxed_expr, num / den))
                    continue
            except (ValueError, ZeroDivisionError):
                pass

        numeric_val = _parse_numeric_string(content)
        if numeric_val is not None:
            boxed_answers.append((original_boxed_expr, numeric_val))
            continue

        m_num_unit = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*([a-zA-Z%]+)", content)
        if m_num_unit:
            try:
                num_val = float(m_num_unit.group(1))
                boxed_answers.append((original_boxed_expr, num_val))
                continue
            except ValueError:
                pass

    if found_any_boxed_expr and boxed_answers:
        if len(boxed_answers) > 1:
            numeric_values_only = all(isinstance(val, (float, int)) for _, val in boxed_answers)
            if numeric_values_only and len(boxed_answers) > 1:
                first_val_candidate = boxed_answers[0][1]
                if isinstance(first_val_candidate, (float, int)):
                    first_numeric_value = float(first_val_candidate)
                    all_other_values_identical = True
                    if len(boxed_answers) > 1:
                        all_other_values_identical = all(
                            math.isclose(val, first_numeric_value, rel_tol=1e-9, abs_tol=1e-9)
                            for _, val in boxed_answers[1:]
                            if isinstance(val, (float, int))
                        )
                    if all_other_values_identical:
                        boxed_answers = [boxed_answers[0]]
    return boxed_answers, found_any_boxed_expr


def extract_numbers(text: str) -> List[Tuple[str, AnswerValue]]:
    """
    Extracts mathematical answers from text based on a hierarchical priority:
    1. HTML <answer>/<ans> tags
    2. Boxed LaTeX expressions (e.g., \\boxed{answer})
    3. GSM8K-style final answer markers (e.g., #### 123)
    4. General numeric or LaTeX-formatted numbers as a fallback.

    Args:
        text: The text to extract answers from.

    Returns:
        A list of tuples, where each tuple contains the original matched
        string and its normalized value (float for numbers, str for MCQs
        or specific string expressions like "A or B").
        Returns an empty list if no answer is confidently extracted.
    """
    html_answers = _extract_html_tag_answers(text)
    if html_answers:
        return html_answers

    boxed_answers, found_any_boxed = _extract_boxed_latex_answers(text)
    if found_any_boxed:
        return boxed_answers

    gsm8k_answers = _extract_gsm8k_answers(text)
    if gsm8k_answers:
        return gsm8k_answers

    general_answers = _extract_general_numeric_answers(text)
    if general_answers:
        return general_answers

    return []


def _extract_gsm8k_answers(text: str) -> List[Tuple[str, AnswerValue]]:
    """Extracts answers from GSM8K-style final answer markers (#### ...)."""
    final_marker_answers: List[Tuple[str, Union[float, str]]] = []
    GSM8K_NUM_CONTENT_PATTERN = r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?"
    for m_final in re.finditer(rf"####\s*({GSM8K_NUM_CONTENT_PATTERN})", text):
        original_marker_expr = m_final.group(0)
        num_str_from_regex = m_final.group(1)
        cleaned_num_str = num_str_from_regex.replace(",", "")
        try:
            final_marker_answers.append((original_marker_expr, float(cleaned_num_str)))
        except ValueError:
            pass
    return final_marker_answers


def _extract_general_numeric_answers(text: str) -> List[Tuple[str, AnswerValue]]:
    """Extracts general numeric or LaTeX-formatted numbers as a fallback."""
    potential_general_matches: List[Dict[str, Any]] = []

    for latex_block_match in re.finditer(r"\$\$(.*?)\$\$|\$(.*?)\$", text, re.DOTALL):
        content = latex_block_match.group(1) if latex_block_match.group(1) is not None else latex_block_match.group(2)
        offset = latex_block_match.start(1) if latex_block_match.group(1) is not None else latex_block_match.start(2)
        if not content:
            continue
        if content.strip().startswith("\\boxed{") and content.strip().endswith("}"):
            continue

        for m_frac_latex in re.finditer(r"\\frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}", content):
            try:
                num, den = float(m_frac_latex.group(1)), float(m_frac_latex.group(2))
                if den != 0:
                    potential_general_matches.append(
                        {
                            "text": m_frac_latex.group(0),
                            "value": num / den,
                            "span": (
                                m_frac_latex.start(0) + offset,
                                m_frac_latex.end(0) + offset,
                            ),
                            "type_priority": 1,
                        }
                    )
            except (ValueError, ZeroDivisionError):
                pass

        for m_sci_latex in re.finditer(r"(-?\d+(?:\.\d+)?)\s*\\times\s*10\^\{(.*?)\}", content):
            try:
                base, exp = float(m_sci_latex.group(1)), float(m_sci_latex.group(2))
                potential_general_matches.append(
                    {
                        "text": m_sci_latex.group(0),
                        "value": base * (10**exp),
                        "span": (
                            m_sci_latex.start(0) + offset,
                            m_sci_latex.end(0) + offset,
                        ),
                        "type_priority": 2,
                    }
                )
            except ValueError:
                pass

        for m_plain_latex in re.finditer(r"(?<![a-zA-Z0-9_])(-?\d+(?:\.\d+)?)(?![a-zA-Z0-9_])", content):
            if _is_coefficient(text_content=content, match_obj=m_plain_latex, num_group_idx=1):
                continue
            try:
                potential_general_matches.append(
                    {
                        "text": m_plain_latex.group(1),
                        "value": float(m_plain_latex.group(1)),
                        "span": (
                            m_plain_latex.start(1) + offset,
                            m_plain_latex.end(1) + offset,
                        ),
                        "type_priority": 3,
                    }
                )
            except ValueError:
                pass

    sci_pattern = r"(?<![a-zA-Z0-9_])(-?\d+\.?\d*[eE][-+]?\d+)(?:\s*([a-zA-Z%]+))?"
    for m in re.finditer(sci_pattern, text):
        if _is_coefficient(text_content=text, match_obj=m, num_group_idx=1, unit_group_idx=2):
            continue
        try:
            potential_general_matches.append(
                {
                    "text": m.group(0),
                    "value": float(m.group(1)),
                    "span": m.span(),
                    "type_priority": 4,
                }
            )
        except ValueError:
            pass

    frac_pattern = r"(?<!\d/)(?<!\d)(?<!\.)(-?\d+)\s*/\s*(-?\d+)(?!\.\d)(?!\d*/)(?:\s+(?!(?:and|or)\b)([a-zA-Z%]+)\b)?"
    for m in re.finditer(frac_pattern, text):
        if _is_coefficient(text_content=text, match_obj=m, num_group_idx=1, unit_group_idx=3):
            continue
        try:
            num, den = float(m.group(1)), float(m.group(2))
            if den == 0:
                continue
            num_str_clean, den_str_clean = m.group(1), m.group(2)
            unit_str_clean = m.group(3) or ""
            display_text = f"{num_str_clean}/{den_str_clean}"
            if unit_str_clean:
                display_text += f" {unit_str_clean}"
            potential_general_matches.append(
                {
                    "text": display_text,
                    "value": num / den,
                    "span": m.span(),
                    "type_priority": 5,
                }
            )
        except (ValueError, ZeroDivisionError):
            pass

    comma_num_pattern = r"(?<![a-zA-Z0-9_])(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?)(?:\s*([a-zA-Z%]+))?"
    for m in re.finditer(comma_num_pattern, text):
        if _is_coefficient(text_content=text, match_obj=m, num_group_idx=1, unit_group_idx=2):
            continue
        try:
            potential_general_matches.append(
                {
                    "text": m.group(0),
                    "value": float(m.group(1).replace(",", "")),
                    "span": m.span(),
                    "type_priority": 6,
                }
            )
        except ValueError:
            pass

    decimal_pattern = r"(?<![a-zA-Z0-9_])(?<!,\d{3})(-?\d+\.\d+)(?!\d*[eE])(?:\s*([a-zA-Z%]+))?"
    for m in re.finditer(decimal_pattern, text):
        if _is_coefficient(text_content=text, match_obj=m, num_group_idx=1, unit_group_idx=2):
            continue
        try:
            potential_general_matches.append(
                {
                    "text": m.group(0),
                    "value": float(m.group(1)),
                    "span": m.span(),
                    "type_priority": 7,
                }
            )
        except ValueError:
            pass

    integer_pattern = (
        r"(?<![a-zA-Z0-9_])(?<!\d\.)(-?\d+)(?!\.\d)(?![eE][-+]?\d+)(?!,\d{3})(?!\s*/\s*\d+)(?:\s*([a-zA-Z%]+))?"
    )
    for m in re.finditer(integer_pattern, text):
        if _is_coefficient(text_content=text, match_obj=m, num_group_idx=1, unit_group_idx=2):
            continue
        try:
            potential_general_matches.append(
                {
                    "text": m.group(0),
                    "value": float(m.group(1)),
                    "span": m.span(),
                    "type_priority": 8,
                }
            )
        except ValueError:
            pass

    potential_general_matches.sort(key=lambda x: (x["span"][0], -(x["span"][1] - x["span"][0]), x["type_priority"]))
    filtered_general_answers: List[Tuple[str, AnswerValue]] = []
    last_covered_end = -1
    for item in potential_general_matches:
        start, end = item["span"]
        if start >= last_covered_end:
            value_to_append = item["value"]
            if isinstance(value_to_append, (int, float)):
                filtered_general_answers.append((item["text"], float(value_to_append)))
            last_covered_end = end

    if filtered_general_answers:
        return filtered_general_answers

    return []


def compare_numbers(
    expected: float,
    actual: float,
    relative_tolerance: float = 1e-5,
    absolute_tolerance: float = 1e-8,
) -> Tuple[bool, float]:
    is_close = math.isclose(expected, actual, rel_tol=relative_tolerance, abs_tol=absolute_tolerance)
    if is_close:
        return True, 1.0
    try:
        if expected == 0:
            error = abs(actual)
            similarity = max(0.0, 1.0 - min(1.0, error / absolute_tolerance))
        else:
            rel_error = abs((expected - actual) / expected)
            similarity = max(0.0, 1.0 - min(1.0, rel_error / relative_tolerance))
    except (ZeroDivisionError, OverflowError):
        similarity = 0.0
    return False, similarity


def _has_unit_text(full_extracted_text: str, numeric_value: float) -> bool:
    """Checks if the extracted text for a number likely contains a unit."""
    content_to_check = full_extracted_text
    if content_to_check.startswith("\\boxed{") and content_to_check.endswith("}"):
        content_to_check = content_to_check[7:-1].strip()

    num_str_float = str(numeric_value)
    num_str_int = str(int(numeric_value)) if numeric_value == int(numeric_value) else None
    search_terms = [num_str_float]
    if num_str_int and num_str_int != num_str_float:
        search_terms.append(num_str_int)

    for term in search_terms:
        found_at = content_to_check.find(term)
        if found_at != -1:
            suffix_start = found_at + len(term)
            if suffix_start < len(content_to_check):
                suffix = content_to_check[suffix_start:].strip().split(" ")[0]
                if suffix and not suffix.replace(".", "", 1).isdigit() and suffix.lower() != "or":
                    return True
    return False


def _check_unboxed_or_strictness(
    model_response_content: str,
    gen_answers_extracted: Sequence[Tuple[str, AnswerValue]],
    metrics: Dict[str, MetricResult],
) -> Optional[EvaluateResult]:
    """Checks for 'unboxed or' strictness violation."""
    raw_extracted_numbers = extract_numbers(model_response_content)
    if (
        " or " in model_response_content.lower()
        and sum(1 for _, val_check in raw_extracted_numbers if isinstance(val_check, (float, int))) > 1
        and not (
            len(gen_answers_extracted) == 1
            and isinstance(gen_answers_extracted[0][1], str)
            and " or " in gen_answers_extracted[0][1].lower()
        )
    ):
        specific_reason_detail = (
            "Generated answer offers multiple numeric alternatives with an unboxed 'or' in the raw response."
        )
        full_reason = f"Strictness fail (Issue #1 - Unboxed 'or'): {specific_reason_detail}"
        metrics["strictness_penalty_unboxed_or"] = MetricResult(
            score=0.0, is_score_valid=False, reason=specific_reason_detail
        )
        return EvaluateResult(score=0.0, reason=full_reason, metrics=metrics)
    return None


def _check_ambiguity_strictness(
    orig_answers_extracted: Sequence[Tuple[str, AnswerValue]],
    gen_answers_extracted: Sequence[Tuple[str, AnswerValue]],
    metrics: Dict[str, MetricResult],
) -> Optional[EvaluateResult]:
    """Checks for ambiguity strictness violation."""
    if len(orig_answers_extracted) == 1 and len(gen_answers_extracted) > 1:
        specific_reason_detail = "Ground truth is specific (one answer), but generated answer is ambiguous (multiple answers extracted, even after potential leniency)."
        full_reason = f"Strictness fail (Issue #2 - Ambiguity): {specific_reason_detail}"
        metrics["strictness_penalty_ambiguity"] = MetricResult(
            score=0.0, is_score_valid=False, reason=specific_reason_detail
        )
        return EvaluateResult(score=0.0, reason=full_reason, metrics=metrics)
    return None


def _check_conflicting_answers_strictness(
    orig_answers_extracted: Sequence[Tuple[str, AnswerValue]],
    gen_answers_extracted: Sequence[Tuple[str, AnswerValue]],
    best_match_score: float,
    match_found_flag: bool,
    is_single_orig_boxed_truth: bool,
    has_matching_gen_boxed_answer: bool,
    tolerance: float,
    absolute_tolerance: float,
    current_best_reason: str,
    metrics: Dict[str, MetricResult],
) -> Tuple[float, bool, str]:
    """Checks for conflicting answers strictness violation."""
    if not (match_found_flag and best_match_score > 0.75):
        return best_match_score, match_found_flag, current_best_reason

    conflicting_extra_numeric_values = []
    if not (is_single_orig_boxed_truth and has_matching_gen_boxed_answer):
        for _, gen_val in gen_answers_extracted:
            if not isinstance(gen_val, (float, int)):
                continue
            is_gen_val_a_match_to_an_orig_val = False
            for _, orig_val_comp in orig_answers_extracted:
                if isinstance(orig_val_comp, (float, int)):
                    if math.isclose(
                        gen_val,
                        orig_val_comp,
                        rel_tol=tolerance,
                        abs_tol=absolute_tolerance,
                    ):
                        is_gen_val_a_match_to_an_orig_val = True
                        break
            if not is_gen_val_a_match_to_an_orig_val:
                conflicting_extra_numeric_values.append(gen_val)

        if conflicting_extra_numeric_values:
            formatted_conflicting = ", ".join(map(str, sorted(list(set(conflicting_extra_numeric_values)))))
            specific_reason_detail = (
                f"Generated answer, while containing a match for an original answer, "
                f"also includes other distinct numerical values not matching any original answer: [{formatted_conflicting}]"
            )
            metrics["strictness_penalty_conflicting_answers"] = MetricResult(
                score=0.0, is_score_valid=False, reason=specific_reason_detail
            )
            return (
                0.0,
                False,
                f"Strictness fail (Conflicting Answers): {specific_reason_detail}. Initial match was: {current_best_reason}",
            )

    return best_match_score, match_found_flag, current_best_reason


@reward_function
def math_reward(
    messages: List[Message],
    *,
    ground_truth: str,
    tolerance: float = 0.001,
    absolute_tolerance: float = 1e-8,
    require_units: bool = False,
    **kwargs: Any,
) -> EvaluateResult:
    """
    NOTE: This is the deprecated/old way of creating an eval in Eval Protocol.
    What use to be the @reward_function decorator is now the @evaluation_test
    decorator with the mode="pointwise" parameter.
    """
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
    model_response_content = messages[-1].content if isinstance(messages[-1].content, str) else ""
    if ground_truth is None or ground_truth == "":
        return EvaluateResult(
            score=0.0,
            reason="Missing or empty ground_truth (expected math answer string).",
            metrics={
                "error": MetricResult(
                    score=0.0,
                    is_score_valid=False,
                    reason="Invalid ground_truth string.",
                )
            },
        )

    gen_answers_extracted_initial = extract_numbers(model_response_content)
    orig_answers_extracted = extract_numbers(ground_truth)
    gen_answers_extracted: List[Tuple[str, AnswerValue]] = list(gen_answers_extracted_initial)
    metrics: Dict[str, MetricResult] = {}

    def format_extracted(items: List[Tuple[str, Union[float, str]]]) -> str:
        if not items:
            return "None"
        return ", ".join([f"'{i[0]}' ({i[1]})" for i in items])

    metrics["extracted_original_answers"] = MetricResult(
        score=0.0,
        is_score_valid=bool(orig_answers_extracted),
        reason=f"Extracted from original: {format_extracted(orig_answers_extracted)}",
    )
    metrics["extracted_generated_answers"] = MetricResult(
        score=0.0,
        is_score_valid=bool(gen_answers_extracted_initial),
        reason=f"Extracted from generated (initial): {format_extracted(gen_answers_extracted_initial)}",
    )

    if not orig_answers_extracted:
        return EvaluateResult(
            score=0.0,
            reason="Could not extract answers from original message (ground truth).",
            metrics=metrics,
        )
    if not gen_answers_extracted_initial:
        return EvaluateResult(
            score=0.0,
            reason="Could not extract answers from generated message, but original message has answers.",
            metrics=metrics,
        )

    # --- DEMO Leniency Modification START ---
    is_single_orig_boxed_truth = False
    orig_boxed_value = None
    if len(orig_answers_extracted) == 1 and orig_answers_extracted[0][0].startswith("\\boxed{"):
        if isinstance(orig_answers_extracted[0][1], (float, int)):
            is_single_orig_boxed_truth = True
            orig_boxed_value = orig_answers_extracted[0][1]

    has_matching_gen_boxed_answer = False
    if is_single_orig_boxed_truth and orig_boxed_value is not None:
        for gen_text, gen_val in gen_answers_extracted_initial:
            if gen_text.startswith("\\boxed{") and isinstance(gen_val, (float, int)):
                if math.isclose(
                    gen_val,
                    orig_boxed_value,
                    rel_tol=tolerance,
                    abs_tol=absolute_tolerance,
                ):
                    has_matching_gen_boxed_answer = True
                    gen_answers_extracted = [(gen_text, cast(AnswerValue, gen_val))]
                    metrics["demo_leniency_info"] = MetricResult(
                        score=1.0,
                        is_score_valid=True,
                        reason=f"Demo Leniency: Matching boxed answer '{gen_text}' found. Simplified gen_answers to this match.",
                    )
                    break
    # --- DEMO Leniency Modification END ---

    unboxed_or_result = _check_unboxed_or_strictness(model_response_content, gen_answers_extracted, metrics)
    if unboxed_or_result:
        return unboxed_or_result

    ambiguity_result = _check_ambiguity_strictness(orig_answers_extracted, gen_answers_extracted, metrics)
    if ambiguity_result:
        return ambiguity_result

    best_match_score = 0.0
    best_match_reason = "No matching answer found"
    match_found_flag = False
    first_comparison_details_for_no_match = ""

    for orig_text, orig_value in orig_answers_extracted:
        for gen_text, gen_value in gen_answers_extracted:
            current_match = False
            current_similarity = 0.0
            comparison_details = ""
            if isinstance(orig_value, (float, int)) and isinstance(gen_value, (float, int)):
                if require_units:
                    orig_has_unit = _has_unit_text(orig_text, float(orig_value))
                    gen_has_unit = _has_unit_text(gen_text, float(gen_value))
                    if orig_has_unit != gen_has_unit:
                        comparison_details = f"Unit presence mismatch (require_units=True). Orig_text: '{orig_text}', Gen_text: '{gen_text}'"
                    else:
                        current_match, current_similarity = compare_numbers(
                            float(orig_value),
                            float(gen_value),
                            tolerance,
                            absolute_tolerance,
                        )
                        comparison_details = (
                            f"Numeric match: {'Yes' if current_match else 'No'}, Similarity: {current_similarity:.3f}"
                        )
                else:
                    current_match, current_similarity = compare_numbers(
                        float(orig_value),
                        float(gen_value),
                        tolerance,
                        absolute_tolerance,
                    )
                    comparison_details = (
                        f"Numeric match: {'Yes' if current_match else 'No'}, Similarity: {current_similarity:.3f}"
                    )
            elif isinstance(orig_value, str) and isinstance(gen_value, str):
                if orig_value.lower() == gen_value.lower():
                    current_match = True
                    current_similarity = 1.0
                comparison_details = (
                    f"String match: {'Yes' if current_match else 'No'} (value: '{gen_value}' vs '{orig_value}')"
                )
            else:
                comparison_details = (
                    f"Type mismatch: Gen({type(gen_value).__name__}) vs Orig({type(orig_value).__name__})"
                )

            if not first_comparison_details_for_no_match:
                first_comparison_details_for_no_match = (
                    f"Initial comparison: Gen='{gen_text}' ({gen_value}) vs Orig='{orig_text}' ({orig_value}).\n"
                    f"{comparison_details}"
                )

            if current_similarity > best_match_score:
                best_match_score = current_similarity
                match_found_flag = current_match
                best_match_reason = (
                    f"Best match: Gen='{gen_text}' ({gen_value}) vs Orig='{orig_text}' ({orig_value}).\n"
                    f"{comparison_details}"
                )
            elif best_match_score == 0 and not match_found_flag and current_similarity == 0:
                best_match_reason = (
                    f"No score match: Gen='{gen_text}' ({gen_value}) vs Orig='{orig_text}' ({orig_value}).\n"
                    f"{comparison_details}"
                )

    if (
        best_match_score == 0
        and not match_found_flag
        and first_comparison_details_for_no_match
        and best_match_reason == "No matching answer found"
    ):
        best_match_reason = first_comparison_details_for_no_match

    best_match_score, match_found_flag, best_match_reason = _check_conflicting_answers_strictness(
        orig_answers_extracted,
        gen_answers_extracted,
        best_match_score,
        match_found_flag,
        is_single_orig_boxed_truth,
        has_matching_gen_boxed_answer,
        tolerance,
        absolute_tolerance,
        best_match_reason,
        metrics,
    )

    metrics["answer_comparison"] = MetricResult(
        score=best_match_score,
        is_score_valid=match_found_flag and best_match_score > 0,
        reason=best_match_reason,
    )
    return EvaluateResult(score=best_match_score, reason=best_match_reason, metrics=metrics)
