"""
Tests for the Solver-Judge evaluation example (Countdown task).

These tests verify the parsing and utility functions work correctly.
To run actual LLM-based tests, use pytest on main.py with appropriate API keys.
"""

import pytest
from eval_protocol.models import EvaluationRow, Message, EvaluateResult

# Import functions from the main module
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    extract_solution,
    validate_equation,
    evaluate_equation,
    compute_score,
    countdown_reward,
    create_judge_prompt,
    parse_judge_selection,
    solver_judge_reward,
    get_countdown_prompt,
)


class TestExtractSolution:
    """Tests for extract_solution function."""

    def test_extract_valid_answer(self):
        response = "Let me solve this: <answer>1 + 2 + 3 * 4</answer>"
        result = extract_solution(response)
        assert result == "1 + 2 + 3 * 4"

    def test_extract_answer_with_whitespace(self):
        response = "<answer>  (1 + 2) * 3  </answer>"
        result = extract_solution(response)
        assert result == "(1 + 2) * 3"

    def test_extract_multiple_answers_takes_last(self):
        response = "<answer>wrong</answer> wait, let me recalculate <answer>1 * 2 * 3 * 4</answer>"
        result = extract_solution(response)
        assert result == "1 * 2 * 3 * 4"

    def test_extract_no_answer(self):
        response = "I think the answer is 24"
        result = extract_solution(response)
        assert result is None

    def test_extract_with_assistant_prefix(self):
        response = "Assistant: Here's the solution: <answer>1 + 2 + 3</answer>"
        result = extract_solution(response)
        assert result == "1 + 2 + 3"


class TestValidateEquation:
    """Tests for validate_equation function."""

    def test_valid_equation_all_numbers(self):
        assert validate_equation("1 + 2 + 3 + 4", [1, 2, 3, 4]) is True

    def test_valid_equation_with_operators(self):
        assert validate_equation("(1 + 2) * 3 * 4", [1, 2, 3, 4]) is True

    def test_invalid_equation_missing_number(self):
        assert validate_equation("1 + 2 + 3", [1, 2, 3, 4]) is False

    def test_invalid_equation_extra_number(self):
        assert validate_equation("1 + 2 + 3 + 4 + 5", [1, 2, 3, 4]) is False

    def test_invalid_equation_repeated_number(self):
        assert validate_equation("1 + 1 + 2 + 3", [1, 2, 3, 4]) is False

    def test_valid_equation_different_order(self):
        assert validate_equation("4 * 3 + 2 + 1", [1, 2, 3, 4]) is True


class TestEvaluateEquation:
    """Tests for evaluate_equation function."""

    def test_simple_addition(self):
        assert evaluate_equation("1 + 2") == 3

    def test_multiplication(self):
        assert evaluate_equation("2 * 3") == 6

    def test_complex_expression(self):
        assert evaluate_equation("(1 + 2) * 3 * 4") == 36

    def test_division(self):
        result = evaluate_equation("10 / 2")
        assert abs(result - 5.0) < 1e-6

    def test_parentheses_order(self):
        assert evaluate_equation("1 + 2 * 3") == 7  # 1 + 6 = 7
        assert evaluate_equation("(1 + 2) * 3") == 9  # 3 * 3 = 9

    def test_invalid_characters(self):
        assert evaluate_equation("1 + 2; import os") is None

    def test_malformed_expression(self):
        assert evaluate_equation("1 + +") is None


class TestComputeScore:
    """Tests for compute_score function."""

    def test_correct_solution(self):
        ground_truth = {"target": 24, "numbers": [1, 2, 3, 4]}
        solution = "<answer>(1 + 2 + 3) * 4</answer>"
        score = compute_score(solution, ground_truth, do_print=False)
        assert score == 1.0

    def test_wrong_result(self):
        ground_truth = {"target": 24, "numbers": [1, 2, 3, 4]}
        solution = "<answer>1 + 2 + 3 + 4</answer>"  # = 10, not 24
        score = compute_score(solution, ground_truth, do_print=False)
        assert score == 0.1  # format_score

    def test_invalid_numbers(self):
        ground_truth = {"target": 24, "numbers": [1, 2, 3, 4]}
        solution = "<answer>5 + 6 + 7 + 8</answer>"  # wrong numbers
        score = compute_score(solution, ground_truth, do_print=False)
        assert score == 0.1  # format_score

    def test_no_answer_tag(self):
        ground_truth = {"target": 24, "numbers": [1, 2, 3, 4]}
        solution = "The answer is (1+2+3)*4 = 24"
        score = compute_score(solution, ground_truth, do_print=False)
        assert score == 0.0


class TestCountdownReward:
    """Tests for countdown_reward function."""

    def test_correct_solution_full_score(self):
        ground_truth = {"target": 24, "numbers": [1, 2, 3, 4]}
        solution = "Let me calculate: <answer>(1 + 2 + 3) * 4</answer>"
        result = countdown_reward(solution, ground_truth)
        assert result.score == 1.0
        assert "correctness" in result.metrics
        assert result.metrics["correctness"].score == 1.0

    def test_wrong_result_partial_score(self):
        ground_truth = {"target": 24, "numbers": [1, 2, 3, 4]}
        solution = "<answer>1 + 2 + 3 + 4</answer>"  # = 10
        result = countdown_reward(solution, ground_truth)
        assert result.score == 0.1
        assert result.metrics["format"].score == 1.0
        assert result.metrics["validation"].score == 1.0
        assert result.metrics["correctness"].score == 0.0

    def test_no_answer_zero_score(self):
        ground_truth = {"target": 24, "numbers": [1, 2, 3, 4]}
        solution = "I don't know how to solve this"
        result = countdown_reward(solution, ground_truth)
        assert result.score == 0.0
        assert result.metrics["format"].score == 0.0

    def test_missing_ground_truth(self):
        result = countdown_reward("any solution", {})
        assert result.score == 0.0
        assert result.is_score_valid is False


class TestParseJudgeSelection:
    """Tests for parse_judge_selection function."""

    def test_valid_selection(self):
        response = "Based on my analysis, <answer>2</answer>"
        assert parse_judge_selection(response, 3) == 1  # 0-indexed

    def test_selection_out_of_range_high(self):
        response = "<answer>5</answer>"
        assert parse_judge_selection(response, 3) == -1

    def test_selection_zero_means_none_correct(self):
        response = "<answer>0</answer>"
        assert parse_judge_selection(response, 3) == -1

    def test_no_answer_tag(self):
        response = "I select solution 2"
        assert parse_judge_selection(response, 3) == -1

    def test_invalid_number(self):
        response = "<answer>abc</answer>"
        assert parse_judge_selection(response, 3) == -1


class TestCreateJudgePrompt:
    """Tests for create_judge_prompt function."""

    def test_prompt_contains_target(self):
        prompt = create_judge_prompt(24, [1, 2, 3, 4], ["sol1", "sol2"])
        assert "24" in prompt
        assert "Target" in prompt

    def test_prompt_contains_numbers(self):
        prompt = create_judge_prompt(24, [1, 2, 3, 4], ["sol1"])
        assert "[1, 2, 3, 4]" in prompt

    def test_prompt_contains_all_solutions(self):
        solutions = ["Solution A content", "Solution B content", "Solution C content"]
        prompt = create_judge_prompt(100, [25, 50, 2], solutions)
        for sol in solutions:
            assert sol in prompt

    def test_prompt_contains_instructions(self):
        prompt = create_judge_prompt(24, [1, 2, 3, 4], ["a", "b"])
        assert "<answer>" in prompt.lower()
        assert "correct" in prompt.lower()


class TestSolverJudgeReward:
    """Tests for solver_judge_reward function (message-based wrapper)."""

    def test_correct_answer_with_ground_truth(self):
        messages = [
            Message(role="user", content=get_countdown_prompt(24, [1, 2, 3, 4])),
            Message(role="assistant", content="<answer>(1 + 2 + 3) * 4</answer>"),
        ]
        ground_truth = {"target": 24, "numbers": [1, 2, 3, 4]}
        result = solver_judge_reward(messages=messages, ground_truth=ground_truth)
        assert result.score == 1.0

    def test_incorrect_answer_with_ground_truth(self):
        messages = [
            Message(role="user", content=get_countdown_prompt(24, [1, 2, 3, 4])),
            Message(role="assistant", content="<answer>1 + 2 + 3 + 4</answer>"),
        ]
        ground_truth = {"target": 24, "numbers": [1, 2, 3, 4]}
        result = solver_judge_reward(messages=messages, ground_truth=ground_truth)
        assert result.score == 0.1  # partial credit for format

    def test_empty_messages(self):
        result = solver_judge_reward(messages=[], ground_truth={"target": 24, "numbers": [1, 2, 3, 4]})
        assert result.score == 0.0
        assert result.is_score_valid is False

    def test_invalid_ground_truth_format(self):
        messages = [
            Message(role="user", content="some prompt"),
            Message(role="assistant", content="<answer>something</answer>"),
        ]
        # Invalid ground_truth (not a dict with target/numbers)
        result = solver_judge_reward(messages=messages, ground_truth="42")
        assert result.score == 1.0  # just checks format
        assert "format" in result.metrics


class TestGetCountdownPrompt:
    """Tests for get_countdown_prompt function."""

    def test_prompt_contains_target(self):
        prompt = get_countdown_prompt(24, [1, 2, 3, 4])
        assert "24" in prompt
        assert "Target" in prompt

    def test_prompt_contains_numbers(self):
        prompt = get_countdown_prompt(100, [25, 50, 2, 1])
        assert "25" in prompt
        assert "50" in prompt
        assert "Available numbers" in prompt

    def test_prompt_contains_instructions(self):
        prompt = get_countdown_prompt(42, [6, 7])
        assert "<answer>" in prompt
        assert "exactly once" in prompt.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
