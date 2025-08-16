"""
Tests for math reward functions.
"""

import re
from typing import List, Optional, Tuple, Union  # Added Tuple

import pytest

# Removed: from eval_protocol.rewards.advanced_math import advanced_math_reward
from eval_protocol.models import (
    EvaluateResult,
    Message,  # Added Message import
    MetricResult,
)
from eval_protocol.rewards.math import compare_numbers, extract_numbers, math_reward


class TestExtractNumbers:
    # --- PRIORITY 1: Boxed LaTeX ---
    def test_P1_boxed_simple_number(self):
        text = "The answer is \\boxed{42}."
        expected = [("\\boxed{42}", 42.0)]
        assert extract_numbers(text) == expected

    def test_P1_boxed_simple_fraction_string(self):
        text = "The answer is \\boxed{1/2}."
        expected = [("\\boxed{1/2}", 0.5)]
        assert extract_numbers(text) == expected

    def test_P1_boxed_latex_fraction(self):
        text = "The answer is \\boxed{\\frac{3}{4}}."
        expected = [("\\boxed{\\frac{3}{4}}", 0.75)]
        assert extract_numbers(text) == expected

    def test_P1_boxed_mcq_letter(self):
        text = "The answer is \\boxed{A}."
        expected = [("\\boxed{A}", "A")]
        assert extract_numbers(text) == expected

    def test_P1_boxed_or_expression(self):
        text = "The roots are \\boxed{x=1 \\text{ or } x=2}."
        expected = [("\\boxed{x=1 \\text{ or } x=2}", "x=1 \\text{ or } x=2")]
        assert extract_numbers(text) == expected

    def test_P1_boxed_complex_content_as_string_or_ignored(self):
        text_no_or = "The answer is \\boxed{x^2+1}."
        # P1: found_any_boxed_expr = True. boxed_answers is empty. Returns []. Correct.
        assert extract_numbers(text_no_or) == []

        text_with_or = "The answer is \\boxed{x^2+1 \\text{ or } y=0}."
        expected_with_or = [("\\boxed{x^2+1 \\text{ or } y=0}", "x^2+1 \\text{ or } y=0")]
        assert extract_numbers(text_with_or) == expected_with_or

    def test_P1_multiple_boxed_expressions(self):
        text = "Answers: \\boxed{1}, then \\boxed{B}, and finally \\boxed{1/2 or 3/2}"
        # Order of extraction depends on finditer, so check presence and content
        results = extract_numbers(text)
        assert len(results) == 3
        assert ("\\boxed{1}", 1.0) in results
        assert ("\\boxed{B}", "B") in results
        assert ("\\boxed{1/2 or 3/2}", "1/2 or 3/2") in results

    # --- PRIORITY 2: GSM8K-style (#### ...) ---
    def test_P2_gsm8k_simple_number(self):
        text = "The final answer is #### 123."  # Input with period
        expected = [("#### 123", 123.0)]  # Regex for num content won't include trailing period in m.group(0)
        assert extract_numbers(text) == expected

    def test_P2_gsm8k_with_commas_and_decimal(self):
        text = "The final answer is #### 1,234.56."  # Input with period
        expected = [("#### 1,234.56", 1234.56)]  # Expect m.group(0) to be "#### 1,234.56"
        assert extract_numbers(text) == expected

    def test_P2_gsm8k_negative_number(self):
        text = "The final answer is #### -7."  # Input with period
        expected = [("#### -7", -7.0)]
        assert extract_numbers(text) == expected

    def test_P2_gsm8k_takes_priority_over_mcq_and_general(self):
        text = "Answer is (A). The number is 10. Final: #### 100."  # Input with period
        expected = [("#### 100", 100.0)]
        assert extract_numbers(text) == expected

    def test_P2_multiple_gsm8k_markers(self):
        text = "First #### 1. Second #### 2."  # Input with periods
        results = extract_numbers(text)
        assert len(results) == 2
        assert ("#### 1", 1.0) in results
        assert ("#### 2", 2.0) in results

    # --- PRIORITY 3: Multiple Choice Question (MCQ) ---
    # MCQ extraction is now handled by multiple_choice_math_reward.py
    # These tests verify that extract_numbers from math.py NO LONGER extracts MCQs.
    def test_P3_mcq_various_formats(self):
        assert extract_numbers("(A)") == []
        assert extract_numbers("B.") == []
        assert extract_numbers("[C]") == []
        assert extract_numbers("{D}") == []
        assert extract_numbers(" E ") == []

    def test_P3_mcq_multiple_different_mcqs(self):
        text = "Options: (A), B., [C]"
        results = extract_numbers(text)
        assert results == []

    def test_P3_mcq_takes_priority_over_general_numbers(self):
        # Since MCQ is no longer extracted by math.extract_numbers, general numbers should be found.
        text = "The number is 10, but the choice is (A)."
        expected = [("10", 10.0)]
        assert extract_numbers(text) == expected

    # --- PRIORITY 4: General Fallback Extraction ---
    def test_P4_general_integers(self):
        text = "The answer is 42. Another value is -17."
        results = extract_numbers(text)
        assert ("42", 42.0) in results
        assert ("-17", -17.0) in results
        assert len(results) == 2

    def test_P4_general_decimals(self):
        text = "The value of pi is 3.14159."
        expected = [("3.14159", 3.14159)]
        assert extract_numbers(text) == expected

    def test_P4_general_scientific_notation(self):
        text = "Avogadro's number is approximately 6.022e23."
        expected = [("6.022e23", 6.022e23)]
        assert extract_numbers(text) == expected

    def test_P4_general_fractions_plain(self):
        text = "One half is 1/2 and three quarters is 3/4."
        results = extract_numbers(text)
        # m.group(0) for "1/2 and" with the current frac_pattern should be "1/2"
        assert ("1/2", 0.5) in results
        assert ("3/4", 0.75) in results
        assert len(results) == 2

    def test_P4_general_latex_fraction_outside_box(self):
        text = "Answer: $\\frac{1}{2}$"
        expected = [("\\frac{1}{2}", 0.5)]
        assert extract_numbers(text) == expected

    def test_P4_general_latex_sci_notation_outside_box(self):
        text = "Value: $3 \\times 10^{8}$"
        expected = [("3 \\times 10^{8}", 3e8)]
        assert extract_numbers(text) == expected

    def test_P4_general_numbers_with_units_no_algebraic_vars(self):
        text = "The distance is 42 km and the weight is 3.5 kg."
        results = extract_numbers(text)
        # Order might vary, check presence
        assert ("42 km", 42.0) in results
        assert ("3.5 kg", 3.5) in results
        assert len(results) == 2

    def test_P4_general_avoids_coefficients_plain_text(self):
        assert extract_numbers("4y") == []
        assert extract_numbers("4 y") == []
        results = extract_numbers("answer = 4x + 2")
        assert len(results) == 1
        assert results[0] == ("2", 2.0)  # Only "2" should be extracted

    def test_P4_general_avoids_coefficients_latex(self):
        assert extract_numbers("$4y$") == []
        assert extract_numbers("$4 y$") == []
        assert extract_numbers("$x=4a+2b$") == []
        results_eq = extract_numbers("$x=4$")  # "4" is extracted from "$x=4$"
        assert len(results_eq) == 1
        assert results_eq[0] == ("4", 4.0)

    def test_P4_general_multiple_formats_fallback(self):
        text = "Values: 42, then 3.14, also $1/4$, and $10 \\text{ m}$, finally $5.5 \\times 10^{6} \\text{ Hz}$"
        results = extract_numbers(text)
        extracted_values = {r[1] for r in results}
        expected_values = {42.0, 3.14, 0.25, 10.0, 5.5e6}
        assert extracted_values == expected_values
        # Check original texts for some to ensure they are reasonable
        # This can be tricky due to overlapping regexes and filtering in fallback.
        # For example, "$10 \\text{ m}$" might extract "10" with original text "10".
        # For "$1/4$", original text should be "1/4".
        # For "$5.5 \\times 10^{6} \\text{ Hz}$", original text "5.5 \\times 10^{6}".
        assert any(r[0] == "1/4" and r[1] == 0.25 for r in results)
        assert any(r[0] == "5.5 \\times 10^{6}" and r[1] == 5.5e6 for r in results)

    # --- Overall Priority Tests ---
    def test_priority_boxed_over_all_else(self):
        text = "The answer is (A), also 10, and #### 20, but finally \\boxed{30}."
        expected = [("\\boxed{30}", 30.0)]
        assert extract_numbers(text) == expected

    def test_priority_gsm8k_over_mcq_and_general(self):
        text = "The answer is (A), also 10, and finally #### 20."  # Period after 20
        expected = [("#### 20", 20.0)]
        assert extract_numbers(text) == expected

    def test_priority_mcq_over_general(self):
        # MCQ logic removed, so general number '10' should be extracted.
        text = "The answer is 10, but choose (A)."
        expected = [("10", 10.0)]
        assert extract_numbers(text) == expected

    def test_no_answer_found(self):
        text = "This is just some text without any clear answer."
        assert extract_numbers(text) == []

    def test_empty_string(self):
        assert extract_numbers("") == []

    def test_extract_numbers_issue_3_scenario_recheck(self):
        text_coeff = "This is $4 y$."
        assert extract_numbers(text_coeff) == []

        text_valid_num_in_latex = "This is $v_R = 4 \\mathrm{km/h}$."
        results_valid = extract_numbers(text_valid_num_in_latex)
        assert len(results_valid) == 1
        assert results_valid[0] == ("4", 4.0)

        text_combined = "Solution with $4 y$ and also $v_R = 4 \\mathrm{km/h}$."
        results_combined = extract_numbers(text_combined)
        assert len(results_combined) == 1
        assert results_combined[0] == ("4", 4.0)

    # --- PRIORITY 0: HTML/XML tag–wrapped numeric answers ---
    def test_P0_html_tag_simple_number(self):
        text = "Some text <answer>77</answer> more text."
        expected = [("<answer>77</answer>", 77.0)]
        assert extract_numbers(text) == expected

    def test_P0_html_tag_with_whitespace(self):
        text = "<answer>   3.14   </answer>"
        expected = [("<answer>   3.14   </answer>", 3.14)]
        assert extract_numbers(text) == expected

    def test_P0_html_tag_string_content(self):
        text = "<answer>Not a number</answer>"
        expected: List[Tuple[str, Union[str, float]]] = []
        assert extract_numbers(text) == expected

    def test_P0_html_tag_multiple_answer_tags(self):
        text = "<answer>1</answer> and <answer>2.5</answer>"
        results = extract_numbers(text)
        assert ("<answer>1</answer>", 1.0) in results
        assert ("<answer>2.5</answer>", 2.5) in results
        assert len(results) == 2

    def test_P0_html_tag_priority_over_general(self):
        text = "The answer is <answer>42</answer> but also 99."
        expected = [("<answer>42</answer>", 42.0)]
        assert extract_numbers(text) == expected


class TestCompareNumbers:
    def test_exact_match(self):
        is_match, similarity = compare_numbers(42.0, 42.0)
        assert is_match is True
        assert similarity == 1.0

    def test_close_match(self):
        is_match, similarity = compare_numbers(3.14159, 3.14, relative_tolerance=0.01)
        assert is_match is True
        assert similarity == 1.0

    def test_not_close_match(self):
        is_match, similarity = compare_numbers(10.0, 11.0, relative_tolerance=0.01)
        assert is_match is False
        assert similarity < 1.0

    def test_very_different(self):
        is_match, similarity = compare_numbers(100.0, 200.0, relative_tolerance=0.01)
        assert is_match is False
        assert similarity == 0.0

    def test_zero_expected(self):
        is_match, similarity = compare_numbers(0.0, 0.00001, absolute_tolerance=0.0001)
        assert is_match is True
        assert similarity == 1.0

        is_match, similarity = compare_numbers(0.0, 0.0001, absolute_tolerance=0.0001)
        assert is_match is True
        assert similarity == 1.0

        is_match, similarity = compare_numbers(0.0, 0.001, absolute_tolerance=0.0001)
        assert is_match is False
        assert similarity < 1.0


class TestMathReward:
    # Helper to create a common prompt
    def _get_prompt(
        self,
    ) -> Optional[List[Message]]:  # Ensure return type matches Optional if used as such
        return [Message(role="user", content="Solve the math problem.")]

    def test_basic_match_boxed(self):
        user_messages = self._get_prompt() or []
        assistant_message = Message(role="assistant", content="It is \\boxed{4}.")
        messages_arg = user_messages + [assistant_message]
        ground_truth_arg = "Answer is \\boxed{4}."
        result = math_reward(messages=messages_arg, ground_truth=ground_truth_arg)
        assert isinstance(result, EvaluateResult)
        assert result.score == 1.0
        # assert result.score == 1.0 # Corrected access

    def test_basic_match_gsm8k(self):
        user_messages = self._get_prompt() or []
        assistant_message = Message(role="assistant", content="The result is #### 4")
        messages_arg = user_messages + [assistant_message]
        ground_truth_arg = "Final answer: #### 4"
        result = math_reward(messages=messages_arg, ground_truth=ground_truth_arg)
        assert isinstance(result, EvaluateResult)
        assert result.score == 1.0
        # assert result.score == 1.0 # Corrected access

    def test_basic_match_mcq(self):
        user_messages = self._get_prompt() or []
        assistant_message = Message(role="assistant", content="My answer is (A).")
        messages_arg = user_messages + [assistant_message]
        ground_truth_arg = "Choice (A)."
        result = math_reward(messages=messages_arg, ground_truth=ground_truth_arg)
        assert isinstance(result, EvaluateResult)
        assert result.score == 0.0
        # assert result.score == 0.0 # Duplicate line removed
        assert (
            result.reason is not None
            and "Could not extract answers from original message (ground truth)" in result.reason
        )

    def test_type_mismatch_mcq_vs_number(self):
        user_messages = self._get_prompt() or []
        assistant_message = Message(role="assistant", content="It is 4.")
        messages_arg = user_messages + [assistant_message]
        ground_truth_arg = "The number is 4."
        result = math_reward(messages=messages_arg, ground_truth=ground_truth_arg)
        assert isinstance(result, EvaluateResult)
        assert result.score == 1.0
        # assert result.score == 1.0 # Corrected access

    def test_close_match_tolerance(self):
        user_messages = self._get_prompt() or []
        assistant_message = Message(role="assistant", content="Pi is \\boxed{3.14}.")
        messages_arg = user_messages + [assistant_message]
        ground_truth_arg = "Pi is \\boxed{3.14159}."
        result = math_reward(messages=messages_arg, ground_truth=ground_truth_arg, tolerance=0.01)
        assert isinstance(result, EvaluateResult)
        assert result.score == 1.0
        # assert result.score == 1.0 # Corrected access

    def test_wrong_answer_numeric(self):
        user_messages = self._get_prompt() or []
        assistant_message = Message(role="assistant", content="Answer: \\boxed{5}.")
        messages_arg = user_messages + [assistant_message]
        ground_truth_arg = "Answer: \\boxed{4}."
        result = math_reward(messages=messages_arg, ground_truth=ground_truth_arg)
        assert isinstance(result, EvaluateResult)
        assert result.score < 0.1
        # assert result.score < 0.1 # Corrected access

    def test_wrong_answer_mcq(self):
        user_messages = self._get_prompt() or []
        assistant_message = Message(role="assistant", content="Choice (B).")
        messages_arg = user_messages + [assistant_message]
        ground_truth_arg = "Choice (A)."
        result = math_reward(messages=messages_arg, ground_truth=ground_truth_arg)
        assert isinstance(result, EvaluateResult)
        assert result.score == 0.0
        # The reason should be about the original message (ground_truth) if it's unparseable
        assert (
            result.reason is not None
            and "Could not extract answers from original message (ground truth)" in result.reason
        )

    def test_no_answer_in_generated(self):
        user_messages = self._get_prompt() or []
        assistant_message = Message(role="assistant", content="Answer is (A).")
        messages_arg = user_messages + [assistant_message]
        ground_truth_arg = "Answer is \\boxed{1}."
        result = math_reward(messages=messages_arg, ground_truth=ground_truth_arg)
        assert isinstance(result, EvaluateResult)
        assert result.score == 0.0
        # assert result.score == 0.0 # Duplicate line removed
        assert result.reason is not None and "Could not extract answers from generated message" in result.reason

    def test_no_answer_in_original_when_gen_is_unparseable(
        self,
    ):  # Renamed to avoid conflict and be more descriptive
        user_messages = self._get_prompt() or []
        assistant_message = Message(role="assistant", content="I don't know.")  # Gen unparseable
        messages_arg = user_messages + [assistant_message]
        ground_truth_arg = "Answer is \\boxed{1}."  # GT parseable
        result = math_reward(messages=messages_arg, ground_truth=ground_truth_arg)
        assert isinstance(result, EvaluateResult)
        assert result.score == 0.0
        assert result.reason is not None and "Could not extract answers from generated message" in result.reason

    def test_unparseable_ground_truth(
        self,
    ):  # Renamed from the second test_no_answer_in_original
        user_messages = self._get_prompt() or []
        assistant_message = Message(role="assistant", content="Answer is \\boxed{1}.")  # Gen parseable
        messages_arg = user_messages + [assistant_message]
        ground_truth_arg = "What is it?"  # GT unparseable
        result = math_reward(messages=messages_arg, ground_truth=ground_truth_arg)
        assert isinstance(result, EvaluateResult)
        assert result.score == 0.0
        # assert result.score == 0.0 # Duplicate line removed
        assert (
            result.reason is not None
            and "Could not extract answers from original message (ground truth)" in result.reason
        )

    # --- Strictness Penalty Tests ---
    def test_penalty_unboxed_or_issue1(self):
        user_messages = self._get_prompt() or []
        assistant_message = Message(role="assistant", content="The answer is $1/2 \\text{ or } 1$.")
        messages_arg = user_messages + [assistant_message]
        ground_truth_arg = "The answer is $\\boxed{1/2}$."
        result = math_reward(messages=messages_arg, ground_truth=ground_truth_arg)
        assert isinstance(result, EvaluateResult)
        assert result.score == 0.0
        # assert result.score == 0.0 # Duplicate line removed
        assert result.reason is not None and result.reason.startswith("Strictness fail (Issue #1")
        assert (
            result.metrics["strictness_penalty_unboxed_or"].reason
            == "Generated answer offers multiple numeric alternatives with an unboxed 'or' in the raw response."
        )

    def test_no_penalty_if_gen_is_single_boxed_or_expr(self):
        user_messages = self._get_prompt() or []
        assistant_message = Message(role="assistant", content="The answer is $\\boxed{1/2 \\text{ or } 1}$.")
        messages_arg = user_messages + [assistant_message]
        ground_truth_arg = "The answer is $\\boxed{1/2 \\text{ or } 1}$."
        result = math_reward(messages=messages_arg, ground_truth=ground_truth_arg)
        assert isinstance(result, EvaluateResult)
        assert result.score == 1.0
        # assert result.score == 1.0 # Duplicate line removed
        assert result.reason is None or ("Strictness fail" not in result.reason)

    def test_penalty_ambiguity_issue2(self):
        user_messages = self._get_prompt() or []
        assistant_message = Message(
            role="assistant",
            content="The equation is $x=\\frac{1}{4}$. Some other numbers are 1, 2, 3.",
        )
        messages_arg = user_messages + [assistant_message]
        ground_truth_arg = "The answer is $\\boxed{1/4}$."
        result = math_reward(messages=messages_arg, ground_truth=ground_truth_arg)
        assert isinstance(result, EvaluateResult)
        assert result.score == 0.0
        # assert result.score == 0.0 # Duplicate line removed
        assert result.reason is not None and result.reason.startswith("Strictness fail (Issue #2")
        assert (
            result.metrics["strictness_penalty_ambiguity"].reason
            == "Ground truth is specific (one answer), but generated answer is ambiguous (multiple answers extracted, even after potential leniency)."
        )

    def test_issue_false_mcq_match_on_v_B(self):
        user_content_str = "## Task B-1.3.\n\nA ship traveling along a river has covered $24 \\mathrm{~km}$ upstream and $28 \\mathrm{~km}$ downstream. For this journey, it took half an hour less than for traveling $30 \\mathrm{~km}$ upstream and $21 \\mathrm{~km}$ downstream, or half an hour more than for traveling $15 \\mathrm{~km}$ upstream and $42 \\mathrm{~km}$ downstream, assuming that both the ship and the river move uniformly.\n\nDetermine the speed of the ship in still water and the speed of the river."
        assistant_content_generated_str = "## Solution.\n\nLet $t$ be the time required for the boat to travel $24 \\mathrm{~km}$ upstream and $28 \\mathrm{~km}$ downstream, $v_{R}$ the speed of the river, and $v_{B}$ the speed of the boat. When the boat is traveling upstream, its speed is $v_{B}-v_{R}$, and when it is traveling downstream, its speed is $v_{B}+v_{R}$.\n\nSince $t=\\frac{s}{v}$, from the given data, we obtain the following system of equations:\n\n$\\left\\{\\begin{array}{l}t=\\frac{24}{v_{B}-v_{R}}+\\frac{28}{v_{B}+v_{R}} \\\\ t+0.5=\\frac{30}{v_{B}-v_{R}}+\\frac{21}{v_{B}+v_{R}} \\\\ t-0.5=\\frac{15}{v_{B}-v_{R}}+\\frac{42}{v_{B}+v_{R}}\\end{array}\\right.$\n\nBy introducing new variables $x=\\frac{3}{v_{B}-v_{R}}, y=\\frac{7}{v_{B}+v_{R}}$, the system transforms into:\n\n$\\left\\{\\begin{array}{l}t=8 x+4 y \\\\ t+0.5=10 x+3 y \\\\ t-0.5=5 x+6 y\\end{array}\\right.$\n\nSubstituting $t$ from the first equation into the remaining two, we get:\n\n$\\left\\{\\begin{array}{l}8 x+4 y+0.5=10 x+3 y \\\\ 8 x+4 y-0.5=5 x+6 y\\end{array}\\right.$\n\n$\\left\\{\\begin{array}{l}2 x-y=0.5 \\\\ 3 x-2 y=0.5\\end{array}\\right.$\n\nThe solution to the last system is (0.5, 0.5). Then we have:\n\n$\\frac{3}{v_{B}-v_{R}}=0.5$, hence, $v_{B}-v_{R}=6 \\mathrm{~and}$\n\n$\\frac{7}{v_{B}+v_{R}}=0.5$, hence, $v_{B}+v_{R}=14$.\n\nThe speed of the river is $v_{R}=4 \\mathrm{~km} / \\mathrm{h}$, and the speed of the boat is $v_{B}=10 \\mathrm{~km} / \\mathrm{h}$.\n\n## Note:\n\nBy substituting $x=\\frac{1}{v_{B}-v_{R}}, y=\\frac{1}{v_{B}+v_{R}} \\mathrm{~and}$ following the same procedure, the initial system transforms into the system $\\left\\{\\begin{array}{l}6 x-7 y=0.5 \\\\ 9 x-14 y=0.5\\end{array}\\right.$\n\nThe solution to this system is $\\left(\\frac{1}{6}, \\frac{1}{14}\\right)$."
        ground_truth_arg = (
            "The speed of the river is $v_R=4 \\mathrm{km/h}$, and the speed of the boat is $v_B=10 \\mathrm{km/h}$."
        )

        user_message = Message(role="user", content=user_content_str)
        assistant_message = Message(role="assistant", content=assistant_content_generated_str)
        messages_arg = [user_message, assistant_message]

        result = math_reward(messages=messages_arg, ground_truth=ground_truth_arg)
        assert isinstance(result, EvaluateResult)
        extracted_gen_reason = result.metrics["extracted_generated_answers"].reason
        extracted_orig_reason = result.metrics["extracted_original_answers"].reason
        assert "'B'" not in extracted_gen_reason
        assert "'B'" not in extracted_orig_reason
        assert result.score == 0.0
        assert result.reason is not None and "Strictness fail (Conflicting Answers)" in result.reason
        assert result.reason is not None and (
            ("0.5" in result.reason) or ("6.0" in result.reason) or ("14.0" in result.reason)
        )

    def test_no_penalty_ambiguity_if_gt_is_also_ambiguous(self):
        user_messages = self._get_prompt() or []
        assistant_message = Message(
            role="assistant",
            content="Solutions: $\\boxed{1}$, $\\boxed{2}$, $\\boxed{3}$.",
        )
        messages_arg = user_messages + [assistant_message]
        ground_truth_arg = "Answers: $\\boxed{1}$, $\\boxed{2}$."
        result = math_reward(messages=messages_arg, ground_truth=ground_truth_arg)
        assert isinstance(result, EvaluateResult)
        assert result.score == 0.0
        assert result.reason is not None and "Strictness fail (Conflicting Answers)" in result.reason
        assert (
            result.reason is not None
            and "includes other distinct numerical values not matching any original answer: [3.0]" in result.reason
        )

    def test_issue3_scenario_correct_handling(self):
        user_messages = self._get_prompt() or []
        ground_truth_arg = (
            "The speed of the river is $v_R=4 \\mathrm{km/h}$, and the speed of the boat is $v_B=10 \\mathrm{km/h}$."
        )
        response_content_str = "Let $x=\\frac{3}{v_{B}-v_{R}}, y=\\frac{7}{v_{B}+v_{R}}$. The solution to the last system is (0.5, 0.5). Then $v_R=4 \\mathrm{km} / \\mathrm{h}$, and $v_B=10 \\mathrm{km} / \\mathrm{h}$."
        assistant_message = Message(role="assistant", content=response_content_str)
        messages_arg = user_messages + [assistant_message]

        result = math_reward(messages=messages_arg, ground_truth=ground_truth_arg)
        assert isinstance(result, EvaluateResult)
        assert result.score == 0.0
        assert result.reason is not None and "Strictness fail (Conflicting Answers)" in result.reason
        assert result.reason is not None and (
            ("0.5" in result.reason) or ("3.0" in result.reason) or ("7.0" in result.reason)
        )

    def test_require_units_basic_functionality(self):
        user_messages = self._get_prompt() or []
        ground_truth_arg = "Answer is \\boxed{10 km}."

        assistant_message_match_units = Message(role="assistant", content="Answer is \\boxed{10 km}.")
        messages_match_units_arg = user_messages + [assistant_message_match_units]

        assistant_message_no_units = Message(role="assistant", content="Answer is \\boxed{10}.")
        messages_no_units_arg = user_messages + [assistant_message_no_units]

        result_match = math_reward(
            messages=messages_match_units_arg,
            ground_truth=ground_truth_arg,
            require_units=True,
        )
        assert isinstance(result_match, EvaluateResult)
        assert result_match.score == 1.0
        assert result_match.reason is None or ("Unit presence mismatch" not in result_match.reason)

        result_no_unit = math_reward(
            messages=messages_no_units_arg,
            ground_truth=ground_truth_arg,
            require_units=True,
        )
        assert isinstance(result_no_unit, EvaluateResult)
        assert result_no_unit.score == 0.0
        assert result_no_unit.reason is not None and "Unit presence mismatch" in result_no_unit.reason
