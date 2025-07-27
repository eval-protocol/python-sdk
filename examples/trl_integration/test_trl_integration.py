# pylint: disable=all
"""
Tests for the TRL integration functionality.

This file contains tests for:
1. The TRL adapter in RewardFunction class
2. The helper functions in trl_adapter.py
3. Basic integration with TRL's expected interfaces
"""

import os
import sys
import unittest
from typing import Any, Dict, List, Optional

# Ensure eval-protocol is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import TRL adapter utilities
from examples.trl_integration.trl_adapter import (  # Changed to absolute import from project root
    apply_reward_to_responses,
    create_combined_reward,
    create_grpo_reward,
    grpo_format_reward,
)
from eval_protocol.models import EvaluateResult, MetricResult

# Import eval-protocol components
from eval_protocol.reward_function import RewardFunction, reward_function
from eval_protocol.rewards.length import length_reward


# Define a simple example reward function (not a pytest test case)
@reward_function
def _example_trl_reward_func(  # Renamed to avoid pytest collection
    messages: List[Dict[str, Any]],
    ground_truth: Optional[List[Dict[str, Any]]] = None,  # Changed from original_messages
    **kwargs,
) -> EvaluateResult:
    """Simple test reward that returns 1.0 if text contains 'good' and 0.0 otherwise."""
    if not messages or len(messages) == 0:
        return EvaluateResult(score=0.0, reason="No messages", metrics={})

    response = messages[-1]
    if response.get("role") != "assistant" or not response.get("content"):
        return EvaluateResult(score=0.0, reason="No assistant response", metrics={})

    text = response.get("content", "")

    score = 1.0 if "good" in text.lower() else 0.0
    reason = "Contains 'good'" if score > 0 else "Does not contain 'good'"

    return EvaluateResult(score=score, reason=reason, metrics={})  # Added metrics={}


# _example_trl_reward_func.__test__ = False # Pytest should ignore functions starting with _


class TestTRLIntegration(unittest.TestCase):
    """Test cases for TRL integration."""

    def setUp(self):
        """Set up test fixtures."""
        # Create message sequences for test data
        cat_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me about cats."},
            {"role": "assistant", "content": "Cats are good pets."},
        ]

        dog_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me about dogs."},
            {"role": "assistant", "content": "Dogs are loyal companions."},
        ]

        # Message format for trl tests - prompt and completion separated
        # Prompts contain system and user (but not assistant) messages
        self.prompts = [
            cat_messages[:-1],  # System and user message
            dog_messages[:-1],  # System and user message
        ]

        # Completions are just the assistant responses
        self.completions = [
            cat_messages[-1]["content"],  # "Cats are good pets."
            dog_messages[-1]["content"],  # "Dogs are loyal companions."
        ]

        # Test messages for the adapter tests (each has full conversation)
        # This is for passing complete conversations to the adapter
        self.test_messages = [cat_messages, dog_messages]

        # GRPO test data with think/answer format
        cat_grpo_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me about cats."},
            {
                "role": "assistant",
                "content": "<think>Cats are domesticated animals.</think><answer>Cats are good pets.</answer>",
            },
        ]

        dog_no_format_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me about dogs."},
            {"role": "assistant", "content": "Dogs are loyal companions."},
        ]

        self.grpo_messages = [cat_grpo_messages, dog_no_format_messages]

        # Create reward functions
        self.test_rf = RewardFunction(func=_example_trl_reward_func)  # Updated to new name
        self.length_rf = RewardFunction(func=length_reward)
        self.format_rf = RewardFunction(func=grpo_format_reward)

    def test_basic_adapter(self):
        """Test that the basic TRL adapter works correctly."""
        adapter = self.test_rf.get_trl_adapter()

        # Prepare prompts and completions
        prompts_batch = [hist[:-1] for hist in self.test_messages]
        completions_batch = [hist[-1]["content"] for hist in self.test_messages]

        # Apply to test messages
        rewards = adapter(prompts=prompts_batch, completions=completions_batch)

        # Check results
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)  # Contains 'good'
        self.assertEqual(rewards[1], 0.0)  # Doesn't contain 'good'

    def test_combined_reward(self):
        """Test that combined rewards work correctly."""
        combined = create_combined_reward(reward_functions=[self.test_rf, self.length_rf], weights=[0.7, 0.3])

        # Prepare prompts and completions
        prompts_batch_combined = [hist[:-1] for hist in self.test_messages]
        completions_batch_combined = [hist[-1]["content"] for hist in self.test_messages]

        # Apply to test messages
        rewards = combined(prompts=prompts_batch_combined, completions=completions_batch_combined)

        # Check results
        self.assertEqual(len(rewards), 2)

        # Calculate expected results
        test_scores = [1.0, 0.0]

        # Get length scores (use the adapter directly)
        length_adapter = self.length_rf.get_trl_adapter()
        prompts_batch_len = [hist[:-1] for hist in self.test_messages]
        completions_batch_len = [hist[-1]["content"] for hist in self.test_messages]
        length_scores = length_adapter(prompts=prompts_batch_len, completions=completions_batch_len)

        # Calculate expected combined scores
        expected = [
            0.7 * test_scores[0] + 0.3 * length_scores[0],
            0.7 * test_scores[1] + 0.3 * length_scores[1],
        ]

        # Allow for small floating point differences
        self.assertAlmostEqual(rewards[0], expected[0], places=5)
        self.assertAlmostEqual(rewards[1], expected[1], places=5)

    def test_grpo_format_reward(self):
        """Test the GRPO format reward function."""
        format_adapter = self.format_rf.get_trl_adapter()

        # Prepare prompts and completions for GRPO
        grpo_prompts_batch = [hist[:-1] for hist in self.grpo_messages]
        grpo_completions_batch = [hist[-1]["content"] for hist in self.grpo_messages]

        # Apply to GRPO messages
        rewards = format_adapter(prompts=grpo_prompts_batch, completions=grpo_completions_batch)

        # Check results
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)  # Has correct format
        self.assertEqual(rewards[1], 0.0)  # Missing format tags

    def test_create_grpo_reward(self):
        """Test the GRPO reward creator."""
        grpo_reward = create_grpo_reward(content_reward=self.test_rf, format_weight=0.4, content_weight=0.6)

        # Prepare prompts and completions for GRPO
        grpo_prompts_batch_creator = [hist[:-1] for hist in self.grpo_messages]
        grpo_completions_batch_creator = [hist[-1]["content"] for hist in self.grpo_messages]

        # Apply to GRPO messages
        rewards = grpo_reward(
            prompts=grpo_prompts_batch_creator,
            completions=grpo_completions_batch_creator,
        )

        # Check results
        self.assertEqual(len(rewards), 2)

        # Calculate expected results
        grpo_prompts_batch_fmt = [hist[:-1] for hist in self.grpo_messages]
        grpo_completions_batch_fmt = [hist[-1]["content"] for hist in self.grpo_messages]

        format_adapter = self.format_rf.get_trl_adapter()
        format_scores = format_adapter(prompts=grpo_prompts_batch_fmt, completions=grpo_completions_batch_fmt)

        grpo_prompts_batch_test = [hist[:-1] for hist in self.grpo_messages]  # Can reuse if identical
        grpo_completions_batch_test = [hist[-1]["content"] for hist in self.grpo_messages]  # Can reuse

        test_adapter = self.test_rf.get_trl_adapter()
        test_scores = test_adapter(prompts=grpo_prompts_batch_test, completions=grpo_completions_batch_test)

        # Expected combined scores
        expected = [
            0.4 * format_scores[0] + 0.6 * test_scores[0],
            0.4 * format_scores[1] + 0.6 * test_scores[1],
        ]

        # Allow for small floating point differences
        self.assertAlmostEqual(rewards[0], expected[0], places=5)
        self.assertAlmostEqual(rewards[1], expected[1], places=5)

    def test_apply_reward_to_responses(self):
        """Test applying reward to text responses."""
        responses = [
            "<think>Cats are domesticated animals.</think><answer>Cats are good pets.</answer>",
            "Dogs are loyal companions.",
        ]

        # Apply test reward with a specified system prompt to ensure proper context
        rewards = apply_reward_to_responses(
            self.test_rf, responses, system_prompt="Evaluate the response for quality."
        )

        # Check results - first response contains 'good'
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)  # Contains 'good'
        self.assertEqual(rewards[1], 0.0)  # Doesn't contain 'good'

        # Apply format reward
        format_rewards = apply_reward_to_responses(self.format_rf, responses)

        # Check results
        self.assertEqual(len(format_rewards), 2)
        self.assertEqual(format_rewards[0], 1.0)  # Has correct format
        self.assertEqual(format_rewards[1], 0.0)  # Missing format tags


# Simple test function to run as a standalone test
def test_standalone_reward():
    """Test that the test_reward function works correctly."""
    # Create test message
    messages = [
        {"role": "user", "content": "Tell me about cats"},
        {"role": "assistant", "content": "Cats are good pets."},
    ]

    # Call reward function directly
    result = _example_trl_reward_func(messages)

    # Verify result
    assert result.score == 1.0
    assert result.reason is not None and "good" in result.reason


if __name__ == "__main__":
    unittest.main()
