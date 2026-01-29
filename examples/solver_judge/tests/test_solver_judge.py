"""
Tests for the Solver-Judge evaluation example.

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
    parse_solver_answer,
    check_answer_correct,
    create_judge_prompt,
    parse_judge_selection,
    solver_judge_reward,
)


class TestParseSolverAnswer:
    """Tests for parse_solver_answer function."""

    def test_parse_valid_answer(self):
        response = "Let me solve this: 15 + 27 = 42. <answer>42</answer>"
        result = parse_solver_answer(response)
        assert result == "<answer>42</answer>"

    def test_parse_answer_with_whitespace(self):
        response = "<answer>  42  </answer>"
        result = parse_solver_answer(response)
        assert result == "<answer>42</answer>"

    def test_parse_answer_case_insensitive(self):
        response = "<ANSWER>42</ANSWER>"
        result = parse_solver_answer(response)
        assert result == "<answer>42</answer>"

    def test_parse_answer_multiline(self):
        response = """Here's my solution:
        <answer>
        42
        </answer>
        Done!"""
        result = parse_solver_answer(response)
        assert "42" in result

    def test_parse_no_answer(self):
        response = "I think the answer is 42 but I'm not sure."
        result = parse_solver_answer(response)
        assert result == "No solution found"


class TestCheckAnswerCorrect:
    """Tests for check_answer_correct function."""

    def test_numeric_match_integer(self):
        solution = "<answer>42</answer>"
        assert check_answer_correct(solution, "42") is True

    def test_numeric_match_float(self):
        solution = "<answer>3.14</answer>"
        assert check_answer_correct(solution, "3.14") is True

    def test_numeric_mismatch(self):
        solution = "<answer>41</answer>"
        assert check_answer_correct(solution, "42") is False

    def test_string_match(self):
        solution = "<answer>Paris</answer>"
        assert check_answer_correct(solution, "Paris") is True

    def test_string_match_case_insensitive(self):
        solution = "<answer>PARIS</answer>"
        assert check_answer_correct(solution, "paris") is True

    def test_no_answer_tag(self):
        solution = "The answer is 42"
        assert check_answer_correct(solution, "42") is False


class TestParseJudgeSelection:
    """Tests for parse_judge_selection function."""

    def test_valid_selection(self):
        response = "Based on my analysis, <answer>2</answer>"
        assert parse_judge_selection(response, 3) == 1  # 0-indexed

    def test_selection_out_of_range_high(self):
        response = "<answer>5</answer>"
        assert parse_judge_selection(response, 3) == -1

    def test_selection_zero(self):
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

    def test_prompt_contains_problem(self):
        problem = "What is 2 + 2?"
        solutions = ["4", "5"]
        prompt = create_judge_prompt(problem, solutions)
        assert problem in prompt

    def test_prompt_contains_all_solutions(self):
        problem = "Test problem"
        solutions = ["Solution A", "Solution B", "Solution C"]
        prompt = create_judge_prompt(problem, solutions)
        for sol in solutions:
            assert sol in prompt

    def test_prompt_contains_instructions(self):
        prompt = create_judge_prompt("test", ["a", "b"])
        assert "<answer>" in prompt.lower()
        assert "select" in prompt.lower() or "output" in prompt.lower()


class TestSolverJudgeReward:
    """Tests for solver_judge_reward function."""

    def test_correct_answer_with_ground_truth(self):
        messages = [
            Message(role="user", content="What is 2 + 2?"),
            Message(role="assistant", content="<answer>4</answer>"),
        ]
        result = solver_judge_reward(messages=messages, ground_truth="4")
        assert result.score == 1.0
        assert "correctness" in result.metrics

    def test_incorrect_answer_with_ground_truth(self):
        messages = [
            Message(role="user", content="What is 2 + 2?"),
            Message(role="assistant", content="<answer>5</answer>"),
        ]
        result = solver_judge_reward(messages=messages, ground_truth="4")
        assert result.score == 0.0

    def test_no_ground_truth_format_check(self):
        messages = [
            Message(role="user", content="What is 2 + 2?"),
            Message(role="assistant", content="<answer>4</answer>"),
        ]
        result = solver_judge_reward(messages=messages, ground_truth=None)
        assert result.score == 1.0  # Format is correct
        assert "format" in result.metrics

    def test_empty_messages(self):
        result = solver_judge_reward(messages=[], ground_truth="4")
        assert result.score == 0.0
        assert result.is_score_valid is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
