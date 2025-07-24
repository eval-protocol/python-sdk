"""
Tests for OpenAI message type compatibility.
"""

import os
import sys
import unittest
from typing import Any, Dict, List

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from eval_protocol import EvaluateResult, Message, MetricResult, reward_function


class OpenAICompatibilityTest(unittest.TestCase):
    """Test compatibility with OpenAI message types."""

    def test_message_type_compatibility(self):
        """Test that our Message type is compatible with OpenAI's."""
        # We no longer directly alias ChatCompletionMessageParam
        # Instead verify we can use the same fields and create our Message
        openai_msg = {"role": "assistant", "content": "Hello"}
        our_msg = Message(**openai_msg)
        self.assertEqual(our_msg.role, "assistant")
        self.assertEqual(our_msg.content, "Hello")

    def test_openai_message_in_decorator(self):
        """Test that the reward_function decorator can handle OpenAI message types."""

        @reward_function
        def sample_evaluator(messages: List[Message], **kwargs: Any) -> EvaluateResult:
            """Sample evaluator that uses OpenAI message types."""
            # Check if the last message is from the assistant
            last_message = messages[-1]

            # Simple evaluation - check if the response is not empty and from the assistant
            success = last_message.role == "assistant" and last_message.content != ""

            return EvaluateResult(
                score=1.0 if success else 0.0,
                reason="Response evaluation",
                metrics={
                    "not_empty": MetricResult(
                        success=success,
                        score=1.0 if success else 0.0,
                        reason=("Response is not empty" if success else "Response is empty or not from assistant"),
                    )
                },
            )

        # Test with OpenAI message types
        system_message = ChatCompletionSystemMessageParam(role="system", content="You are a helpful assistant.")
        user_message = ChatCompletionUserMessageParam(role="user", content="Hello!")
        assistant_message = ChatCompletionAssistantMessageParam(role="assistant", content="Hi there!")

        # Convert to dict for the decorated function
        messages_dict = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        # Test the evaluator with dict messages
        result = sample_evaluator(messages=messages_dict)

        # Verify the result
        self.assertIn("metrics", result)
        self.assertIn("not_empty", result["metrics"])
        self.assertEqual(result["metrics"]["not_empty"]["score"], 1.0)
        self.assertEqual(result["metrics"]["not_empty"]["reason"], "Response is not empty")


if __name__ == "__main__":
    unittest.main()
