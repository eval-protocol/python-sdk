"""
Tests for format reward function.
"""

import os
import re
import sys
import unittest

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval_protocol.models import EvaluateResult, Message
from eval_protocol.rewards.format import format_reward


class TestFormatReward(unittest.TestCase):
    """Test the format reward function."""

    def test_think_answer_format_match(self):
        """Test that the format reward correctly identifies matching format."""
        # Create a message with the correct format
        correct_format = """<think>
This is my reasoning process.
I think the answer is 42.
</think>
<answer>
42
</answer>"""

        messages = [
            {
                "role": "user",
                "content": "What is the answer to life, the universe, and everything?",
            },
            {"role": "assistant", "content": correct_format},
        ]

        result = format_reward(messages=messages, ground_truth=None)

        self.assertIsInstance(result, EvaluateResult)
        # Check the score is 1.0 for correct format
        # Attribute access
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.metrics["format_check"].score, 1.0)
        self.assertTrue(result.metrics["format_check"].is_score_valid)
        # Dictionary access
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["metrics"]["format_check"]["score"], 1.0)
        self.assertTrue(result["metrics"]["format_check"]["is_score_valid"])

    def test_think_answer_format_mismatch(self):
        """Test that the format reward correctly identifies mismatched format."""
        # Create a message with incorrect format
        incorrect_format = """I think the answer is 42.
The answer is 42."""

        messages = [
            {
                "role": "user",
                "content": "What is the answer to life, the universe, and everything?",
            },
            {"role": "assistant", "content": incorrect_format},
        ]

        result = format_reward(messages=messages, ground_truth=None)

        self.assertIsInstance(result, EvaluateResult)
        # Check the score is 0.0 for incorrect format
        # Attribute access
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.metrics["format_check"].score, 0.0)
        self.assertFalse(result.metrics["format_check"].is_score_valid)
        # Dictionary access
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["metrics"]["format_check"]["score"], 0.0)
        self.assertFalse(result["metrics"]["format_check"]["is_score_valid"])

    def test_think_answer_format_wrong_order(self):
        """Test that the format reward fails when tags are in wrong order."""
        # Create a message with tags in wrong order
        wrong_order = """<answer>
42
</answer>
<think>
This is my reasoning process.
I think the answer is 42.
</think>"""

        messages = [
            {
                "role": "user",
                "content": "What is the answer to life, the universe, and everything?",
            },
            {"role": "assistant", "content": wrong_order},
        ]

        result = format_reward(messages=messages, ground_truth=None)

        self.assertIsInstance(result, EvaluateResult)
        # Check the score is 0.0 for incorrect order
        # Attribute access
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.metrics["format_check"].score, 0.0)
        self.assertFalse(result.metrics["format_check"].is_score_valid)
        # Dictionary access
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["metrics"]["format_check"]["score"], 0.0)
        self.assertFalse(result["metrics"]["format_check"]["is_score_valid"])

    def test_custom_format_regex(self):
        """Test that the format reward works with custom regex patterns."""
        # Create a message with a custom format
        custom_format = """[REASONING]
This is my reasoning process.
[/REASONING]
[RESULT]
42
[/RESULT]"""

        messages = [
            {
                "role": "user",
                "content": "What is the answer to life, the universe, and everything?",
            },
            {"role": "assistant", "content": custom_format},
        ]

        # Use a custom regex pattern
        custom_regex = r"^\[REASONING\].*?\[/REASONING\].*?\[RESULT\].*?\[/RESULT\]$"

        result = format_reward(messages=messages, ground_truth=None, format_regex=custom_regex)

        self.assertIsInstance(result, EvaluateResult)
        # Check the score is 1.0 for correct custom format
        # Attribute access
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.metrics["format_check"].score, 1.0)
        self.assertTrue(result.metrics["format_check"].is_score_valid)
        # Dictionary access
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["metrics"]["format_check"]["score"], 1.0)
        self.assertTrue(result["metrics"]["format_check"]["is_score_valid"])

    def test_no_messages(self):
        """Test that the format reward handles empty message list."""
        result = format_reward(messages=[], ground_truth=None)

        self.assertIsInstance(result, EvaluateResult)
        # Check the score is 0.0 for no messages
        # Attribute access
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.metrics["format_check"].score, 0.0)
        self.assertFalse(result.metrics["format_check"].is_score_valid)
        # Dictionary access
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["metrics"]["format_check"]["score"], 0.0)
        self.assertFalse(result["metrics"]["format_check"]["is_score_valid"])

    def test_non_assistant_message(self):
        """Test that the format reward handles non-assistant messages."""
        messages = [
            {
                "role": "user",
                "content": "What is the answer to life, the universe, and everything?",
            }
        ]

        result = format_reward(messages=messages, ground_truth=None)

        self.assertIsInstance(result, EvaluateResult)
        # Check the score is 0.0 for no assistant message
        # Attribute access
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.metrics["format_check"].score, 0.0)
        self.assertFalse(result.metrics["format_check"].is_score_valid)
        # Dictionary access
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["metrics"]["format_check"]["score"], 0.0)
        self.assertFalse(result["metrics"]["format_check"]["is_score_valid"])

    def test_partial_match_mode(self):
        """Test that the format reward works in partial match mode."""
        # Create a message with the format embedded in other text
        partial_format = """Here's my response:
<think>
This is my reasoning process.
I think the answer is 42.
</think>
<answer>
42
</answer>
Thanks for asking!"""

        messages = [
            {
                "role": "user",
                "content": "What is the answer to life, the universe, and everything?",
            },
            {"role": "assistant", "content": partial_format},
        ]

        # Exact match should fail
        result_exact = format_reward(messages=messages, ground_truth=None, require_exact_match=True)
        self.assertIsInstance(result_exact, EvaluateResult)
        # Attribute access
        self.assertEqual(result_exact.score, 0.0)
        # Dictionary access
        self.assertEqual(result_exact["score"], 0.0)

        # Partial match should succeed with a pattern without anchors
        pattern_without_anchors = r"<think>\n.*?</think>\n<answer>\n.*?</answer>"
        result_partial = format_reward(
            messages=messages,
            ground_truth=None,
            format_regex=pattern_without_anchors,
            require_exact_match=False,
        )
        self.assertIsInstance(result_partial, EvaluateResult)
        # Attribute access
        self.assertEqual(result_partial.score, 1.0)
        # Dictionary access
        self.assertEqual(result_partial["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
