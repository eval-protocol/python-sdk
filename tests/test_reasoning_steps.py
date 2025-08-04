"""
Tests for reasoning steps reward function.
"""

import os
import sys
import unittest

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval_protocol.models import EvaluateResult, Message
from eval_protocol.rewards.reasoning_steps import (
    reasoning_steps_reward,
    sequence_reward,
)


class TestReasoningStepsReward(unittest.TestCase):
    """Test the reasoning steps reward function."""

    def test_numbered_steps(self):
        """Test detection of explicitly numbered steps."""
        content = """
        To solve this problem, I'll use a step-by-step approach:

        Step 1: Identify the relevant variables
        Let's call the unknown quantity x.

        Step 2: Set up the equation
        We know that 2x + 3 = 7

        Step 3: Solve for x
        2x + 3 = 7
        2x = 4
        x = 2

        Therefore, x = 2 is the solution.
        """

        messages = [
            {"role": "user", "content": "Solve the equation 2x + 3 = 7"},
            {"role": "assistant", "content": content},
        ]

        result = reasoning_steps_reward(messages=messages)

        self.assertIsInstance(result, EvaluateResult)
        # Should be high score for explicit steps
        # Attribute access
        self.assertEqual(result.score, 1.0)
        self.assertTrue(result.metrics["reasoning_steps"].is_score_valid)
        self.assertIn("explicit_steps", result.metrics)
        # Dictionary access
        self.assertEqual(result["score"], 1.0)
        self.assertTrue(result["metrics"]["reasoning_steps"]["is_score_valid"])
        self.assertIn("explicit_steps", result["metrics"])

    def test_numbered_list(self):
        """Test detection of numbered list items."""
        content = """
        I'll break down the algorithm into steps:

        1. Initialize a counter variable to 0
        2. Iterate through each element in the array
        3. For each element, increment the counter if it meets the condition
        4. Return the final counter value

        This algorithm has a time complexity of O(n).
        """

        messages = [
            {
                "role": "user",
                "content": "Explain how to count specific elements in an array",
            },
            {"role": "assistant", "content": content},
        ]

        result = reasoning_steps_reward(messages=messages)

        self.assertIsInstance(result, EvaluateResult)
        # Should be high score for numbered lists
        # Attribute access
        self.assertEqual(result.score, 1.0)
        self.assertTrue(result.metrics["reasoning_steps"].is_score_valid)
        self.assertIn("numbered_lists", result.metrics)
        # Dictionary access
        self.assertEqual(result["score"], 1.0)
        self.assertTrue(result["metrics"]["reasoning_steps"]["is_score_valid"])
        self.assertIn("numbered_lists", result["metrics"])

    def test_bullet_points(self):
        """Test detection of bullet points."""
        content = """
        To implement this feature, follow these steps:

        * First, create a new component file
        * Add the necessary imports at the top
        * Implement the component's logic
        * Export the component
        * Import and use it in your main application

        This pattern ensures maintainability and modularity.
        """

        messages = [
            {
                "role": "user",
                "content": "How do I implement a reusable component?",
            },
            {"role": "assistant", "content": content},
        ]

        result = reasoning_steps_reward(messages=messages)

        self.assertIsInstance(result, EvaluateResult)
        # Should be high score for bullet points
        # Attribute access
        self.assertEqual(result.score, 1.0)
        self.assertTrue(result.metrics["reasoning_steps"].is_score_valid)
        self.assertIn("bullet_points", result.metrics)
        # Dictionary access
        self.assertEqual(result["score"], 1.0)
        self.assertTrue(result["metrics"]["reasoning_steps"]["is_score_valid"])
        self.assertIn("bullet_points", result["metrics"])

    def test_transition_phrases(self):
        """Test detection of transition phrases."""
        content = """
        Let me walk you through the process of solving this:

        First, I need to identify what we're looking for.
        Second, I'll analyze the constraints given in the problem.
        Third, I'll apply the formula that relates these variables.
        Finally, I'll calculate the answer and verify it matches the constraints.

        The solution is 42.
        """

        messages = [
            {"role": "user", "content": "Solve this word problem"},
            {"role": "assistant", "content": content},
        ]

        result = reasoning_steps_reward(messages=messages)

        self.assertIsInstance(result, EvaluateResult)
        # Should be high score for transition phrases
        # Attribute access
        self.assertEqual(result.score, 1.0)
        self.assertTrue(result.metrics["reasoning_steps"].is_score_valid)
        self.assertIn("transition_phrases", result.metrics)
        # Dictionary access
        self.assertEqual(result["score"], 1.0)
        self.assertTrue(result["metrics"]["reasoning_steps"]["is_score_valid"])
        self.assertIn("transition_phrases", result["metrics"])

    def test_custom_pattern(self):
        """Test with custom pattern."""
        content = """
        Let me think about this problem:

        STEP-A: Define what we're looking for
        STEP-B: Recall relevant formulas
        STEP-C: Apply the formulas to our problem
        STEP-D: Calculate the final answer

        The answer is 42.
        """

        messages = [
            {"role": "user", "content": "Solve this calculation problem"},
            {"role": "assistant", "content": content},
        ]

        # Using a custom pattern that detects "STEP-X:" format
        result = reasoning_steps_reward(
            messages=messages,
            pattern=r"STEP-[A-Z]:",
            exclusive_patterns=True,
            min_steps=3,
        )

        self.assertIsInstance(result, EvaluateResult)
        # Should be high score with custom pattern
        # Attribute access
        self.assertEqual(result.score, 1.0)
        self.assertTrue(result.metrics["reasoning_steps"].is_score_valid)
        # Dictionary access
        self.assertEqual(result["score"], 1.0)
        self.assertTrue(result["metrics"]["reasoning_steps"]["is_score_valid"])

    def test_insufficient_steps(self):
        """Test with insufficient reasoning steps."""
        content = """
        The answer is 42.
        """

        messages = [
            {"role": "user", "content": "What is 6 times 7?"},
            {"role": "assistant", "content": content},
        ]

        result = reasoning_steps_reward(messages=messages)

        self.assertIsInstance(result, EvaluateResult)
        # Should be low score for insufficient steps
        # Attribute access
        self.assertEqual(result.score, 0.0)
        self.assertFalse(result.metrics["reasoning_steps"].is_score_valid)
        # Dictionary access
        self.assertEqual(result["score"], 0.0)
        self.assertFalse(result["metrics"]["reasoning_steps"]["is_score_valid"])

    def test_partial_score(self):
        """Test with partial reasoning steps."""
        content = """
        Step 1: Figure out what we're solving for
        Step 2: Apply the formula

        The answer is 42.
        """

        messages = [
            {"role": "user", "content": "Solve this problem"},
            {"role": "assistant", "content": content},
        ]

        result = reasoning_steps_reward(messages=messages, min_steps=3)

        self.assertIsInstance(result, EvaluateResult)
        # Should be partial score for some but not enough steps
        # Attribute access
        self.assertGreater(result.score, 0.0)
        self.assertLess(result.score, 1.0)
        self.assertFalse(result.metrics["reasoning_steps"].is_score_valid)
        # Dictionary access
        self.assertGreater(result["score"], 0.0)
        self.assertLess(result["score"], 1.0)
        self.assertFalse(result["metrics"]["reasoning_steps"]["is_score_valid"])

    def test_max_steps(self):
        """Test with maximum steps parameter."""
        content = """
        I'll use a detailed approach:

        Step 1: Identify variables
        Step 2: Set up equations
        Step 3: Simplify
        Step 4: Solve for unknowns
        Step 5: Verify the solution
        Step 6: Interpret the result
        """

        messages = [
            {"role": "user", "content": "Solve this complex equation"},
            {"role": "assistant", "content": content},
        ]

        result = reasoning_steps_reward(messages=messages, min_steps=3, max_steps=5)

        self.assertIsInstance(result, EvaluateResult)
        # Should be max score even with more than max_steps
        # Attribute access
        self.assertEqual(result.score, 1.0)
        self.assertTrue(result.metrics["reasoning_steps"].is_score_valid)
        # Dictionary access
        self.assertEqual(result["score"], 1.0)
        self.assertTrue(result["metrics"]["reasoning_steps"]["is_score_valid"])

    def test_sequence_reward_basic(self):
        """Test sequence reward with default terms."""
        content = """
        First, let's identify what we're looking for.
        Second, we need to formulate the equation.
        Third, we'll solve for the unknown.
        Finally, we verify our answer.

        The solution is correct.
        """

        messages = [
            {"role": "user", "content": "Show how to solve this step by step"},
            {"role": "assistant", "content": content},
        ]

        result = sequence_reward(messages=messages)

        self.assertIsInstance(result, EvaluateResult)
        # Should detect sequential terms correctly
        # Attribute access
        self.assertEqual(result.score, 1.0)
        self.assertTrue(result.metrics["sequence_reasoning"].is_score_valid)
        # Dictionary access
        self.assertEqual(result["score"], 1.0)
        self.assertTrue(result["metrics"]["sequence_reasoning"]["is_score_valid"])

    def test_sequence_reward_custom(self):
        """Test sequence reward with custom terms."""
        content = """
        Begin by examining the premises.
        Continue by identifying logical connections.
        Proceed to draw preliminary conclusions.
        End with a final deduction.

        The argument is valid.
        """

        messages = [
            {"role": "user", "content": "Analyze this logical argument"},
            {"role": "assistant", "content": content},
        ]

        result = sequence_reward(
            messages=messages,
            sequence_terms=["Begin", "Continue", "Proceed", "End"],
            min_matches=3,
        )

        self.assertIsInstance(result, EvaluateResult)
        # Should detect custom sequential terms correctly
        # Attribute access
        self.assertEqual(result.score, 1.0)
        self.assertTrue(result.metrics["sequence_reasoning"].is_score_valid)
        # Dictionary access
        self.assertEqual(result["score"], 1.0)
        self.assertTrue(result["metrics"]["sequence_reasoning"]["is_score_valid"])

    def test_sequence_reward_partial(self):
        """Test sequence reward with partial matches."""
        content = """
        First I'll analyze the problem.
        Then I'll formulate a solution approach.

        The answer is 42.
        """

        messages = [
            {"role": "user", "content": "Solve this problem"},
            {"role": "assistant", "content": content},
        ]

        result = sequence_reward(messages=messages, min_matches=3)

        self.assertIsInstance(result, EvaluateResult)
        # Should have partial score for some but not enough sequential terms
        # Attribute access
        self.assertGreater(result.score, 0.0)
        self.assertLess(result.score, 1.0)
        self.assertFalse(result.metrics["sequence_reasoning"].is_score_valid)
        # Dictionary access
        self.assertGreater(result["score"], 0.0)
        self.assertLess(result["score"], 1.0)
        self.assertFalse(result["metrics"]["sequence_reasoning"]["is_score_valid"])


if __name__ == "__main__":
    unittest.main()
