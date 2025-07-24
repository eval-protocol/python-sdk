"""
Tests for tag count reward function.
"""

import os
import sys
import unittest

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval_protocol.models import EvaluateResult, Message
from eval_protocol.rewards.tag_count import tag_count_reward


class TestTagCountReward(unittest.TestCase):
    """Test the tag count reward function."""

    def test_basic_tag_counting(self):
        """Test basic tag counting functionality."""
        # Create a message with multiple tags
        content = """
        <think>
        Let me think about this problem.
        First, I need to understand what's being asked.
        </think>

        <answer>
        The solution is 42.
        </answer>
        """

        messages = [
            {
                "role": "user",
                "content": "What is the answer to life, the universe, and everything?",
            },
            {"role": "assistant", "content": content},
        ]

        result = tag_count_reward(
            messages=messages,
            required_tags=["think", "answer"],
            score_per_tag=0.5,
        )

        self.assertIsInstance(result, EvaluateResult)
        # Check the overall score (0.5 * 2 = 1.0)
        # Attribute access
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.metrics["overall"].is_score_valid, True)
        self.assertEqual(result.metrics["tag_think"].score, 1.0)
        self.assertEqual(result.metrics["tag_answer"].score, 1.0)
        # Dictionary access
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["metrics"]["overall"]["is_score_valid"], True)
        self.assertEqual(result["metrics"]["tag_think"]["score"], 1.0)
        self.assertEqual(result["metrics"]["tag_answer"]["score"], 1.0)

    def test_missing_tags(self):
        """Test when some required tags are missing."""
        content = """
        <answer>
        The answer is 42.
        </answer>
        """

        messages = [
            {"role": "user", "content": "What is the answer?"},
            {"role": "assistant", "content": content},
        ]

        result = tag_count_reward(
            messages=messages,
            required_tags=["think", "answer"],
            score_per_tag=0.5,
        )

        self.assertIsInstance(result, EvaluateResult)
        # Check the overall score (0.5 * 1 = 0.5)
        # Attribute access
        self.assertEqual(result.score, 0.5)
        self.assertEqual(result.metrics["overall"].is_score_valid, False)
        self.assertEqual(result.metrics["tag_think"].score, 0.0)
        self.assertEqual(result.metrics["tag_answer"].score, 1.0)
        # Dictionary access
        self.assertEqual(result["score"], 0.5)
        self.assertEqual(result["metrics"]["overall"]["is_score_valid"], False)
        self.assertEqual(result["metrics"]["tag_think"]["score"], 0.0)
        self.assertEqual(result["metrics"]["tag_answer"]["score"], 1.0)

    def test_unbalanced_tags(self):
        """Test behavior with unbalanced tags."""
        content = """
        <think>
        Let me think about this problem.
        First, I need to understand what's being asked.

        <answer>
        The solution is 42.
        </answer>
        """

        messages = [
            {"role": "user", "content": "What is the answer?"},
            {"role": "assistant", "content": content},
        ]

        # With require_balanced=True (default)
        result_balanced = tag_count_reward(
            messages=messages,
            required_tags=["think", "answer"],
            score_per_tag=0.5,
        )

        self.assertIsInstance(result_balanced, EvaluateResult)
        # Only the balanced "answer" tag should count, "think" should be penalized
        # Attribute access
        self.assertEqual(result_balanced.score, 0.0)  # 0.5 - 0.5 = 0
        self.assertEqual(result_balanced.metrics["overall"].is_score_valid, False)
        self.assertEqual(result_balanced.metrics["tag_think"].score, 0.0)
        # Dictionary access
        self.assertEqual(result_balanced["score"], 0.0)
        self.assertEqual(result_balanced["metrics"]["overall"]["is_score_valid"], False)
        self.assertEqual(result_balanced["metrics"]["tag_think"]["score"], 0.0)

        # With require_balanced=False
        result_unbalanced = tag_count_reward(
            messages=messages,
            required_tags=["think", "answer"],
            score_per_tag=0.5,
            require_balanced=False,
        )

        self.assertIsInstance(result_unbalanced, EvaluateResult)
        # Both tags should be counted even though "think" is unbalanced
        # Attribute access
        self.assertEqual(result_unbalanced.score, 1.0)  # 0.5 * 2 = 1.0
        self.assertEqual(result_unbalanced.metrics["tag_think"].score, 1.0)
        # Dictionary access
        self.assertEqual(result_unbalanced["score"], 1.0)
        self.assertEqual(result_unbalanced["metrics"]["tag_think"]["score"], 1.0)

    def test_custom_tags(self):
        """Test with custom tag names."""
        content = """
        <reasoning>
        This is a complex problem that requires careful analysis.
        </reasoning>

        <conclusion>
        Based on my reasoning, I conclude that X = Y.
        </conclusion>
        """

        messages = [
            {"role": "user", "content": "Solve for X."},
            {"role": "assistant", "content": content},
        ]

        result = tag_count_reward(
            messages=messages,
            required_tags=["reasoning", "conclusion"],
            score_per_tag=0.25,
        )

        self.assertIsInstance(result, EvaluateResult)
        # Check the overall score (0.25 * 2 = 0.5)
        # Attribute access
        self.assertEqual(result.score, 0.5)
        self.assertEqual(result.metrics["overall"].is_score_valid, True)
        self.assertEqual(result.metrics["tag_reasoning"].score, 1.0)
        self.assertEqual(result.metrics["tag_conclusion"].score, 1.0)
        # Dictionary access
        self.assertEqual(result["score"], 0.5)
        self.assertEqual(result["metrics"]["overall"]["is_score_valid"], True)
        self.assertEqual(result["metrics"]["tag_reasoning"]["score"], 1.0)
        self.assertEqual(result["metrics"]["tag_conclusion"]["score"], 1.0)

    def test_multiple_occurrences(self):
        """Test when tags appear multiple times."""
        content = """
        <step>First, we calculate A.</step>
        <step>Next, we calculate B.</step>
        <step>Finally, we calculate C.</step>

        <answer>The final result is 123.</answer>
        """

        messages = [
            {"role": "user", "content": "Solve this step by step."},
            {"role": "assistant", "content": content},
        ]

        result = tag_count_reward(
            messages=messages,
            required_tags=["step", "answer"],
            score_per_tag=0.5,
        )

        self.assertIsInstance(result, EvaluateResult)
        # Check the overall score (0.5 * 2 = 1.0)
        # Attribute access
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.metrics["overall"].is_score_valid, True)
        self.assertIn("3 balanced 'step' tag(s)", result.metrics["tag_step"].reason)
        # Dictionary access
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["metrics"]["overall"]["is_score_valid"], True)
        self.assertIn("3 balanced 'step' tag(s)", result["metrics"]["tag_step"]["reason"])

    def test_no_messages(self):
        """Test behavior with empty messages list."""
        result = tag_count_reward(messages=[], required_tags=["think", "answer"])
        self.assertIsInstance(result, EvaluateResult)
        # Attribute access
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.metrics["tag_count"].score, 0.0)
        self.assertEqual(result.metrics["tag_count"].is_score_valid, False)
        # Dictionary access
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["metrics"]["tag_count"]["score"], 0.0)
        self.assertEqual(result["metrics"]["tag_count"]["is_score_valid"], False)

    def test_non_assistant_message(self):
        """Test behavior when the last message is not from the assistant."""
        messages = [{"role": "user", "content": "What is the answer?"}]

        result = tag_count_reward(messages=messages, required_tags=["think", "answer"])
        self.assertIsInstance(result, EvaluateResult)
        # Attribute access
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.metrics["tag_count"].is_score_valid, False)
        # Dictionary access
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["metrics"]["tag_count"]["is_score_valid"], False)

    def test_attributes_in_tags(self):
        """Test tags with HTML-like attributes."""
        content = """
        <think method="systematic">
        Let me analyze this step by step.
        </think>

        <answer confidence="high">
        The result is definitely 42.
        </answer>
        """

        messages = [
            {"role": "user", "content": "What is the answer?"},
            {"role": "assistant", "content": content},
        ]

        result = tag_count_reward(
            messages=messages,
            required_tags=["think", "answer"],
            score_per_tag=0.5,
        )

        self.assertIsInstance(result, EvaluateResult)
        # Both tags should be counted correctly despite having attributes
        # Attribute access
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.metrics["tag_think"].score, 1.0)
        self.assertEqual(result.metrics["tag_answer"].score, 1.0)
        # Dictionary access
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["metrics"]["tag_think"]["score"], 1.0)
        self.assertEqual(result["metrics"]["tag_answer"]["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
