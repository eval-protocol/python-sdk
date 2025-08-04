import json
import sys  # Import sys
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# Ensure the module is loaded (though RewardFunction import likely does this)
import eval_protocol
from eval_protocol.models import EvaluateResult, MetricResult  # Changed

# Get a direct reference to the module object
reward_function_module_obj = sys.modules["eval_protocol.reward_function"]
from eval_protocol import RewardFunction


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_messages(self):
        """Test handling of empty messages arrays."""

        def reward_func(messages, ground_truth, **kwargs):  # Changed
            """Function that expects non-empty messages."""
            if not messages or not ground_truth:  # Changed
                raise ValueError("Messages cannot be empty")
            return EvaluateResult(score=0.5, reason="Test reason", metrics={})  # Changed

        reward_fn = RewardFunction(func=reward_func, mode="local")

        # Test with empty messages
        with pytest.raises(ValueError):
            reward_fn(
                messages=[],
                ground_truth=[{"role": "user", "content": "Hello"}],  # Changed
            )

        # Test with empty ground_truth
        with pytest.raises(ValueError):
            reward_fn(
                messages=[{"role": "user", "content": "Hello"}],
                ground_truth=[],  # Changed
            )

    def test_invalid_message_structure(self):
        """Test handling of invalid message structure."""

        def reward_func(messages, ground_truth, **kwargs):  # Changed
            """Function that validates message structure."""
            for msg in messages + ground_truth:  # Changed
                if "role" not in msg or "content" not in msg:
                    raise ValueError("Invalid message structure")
            return EvaluateResult(score=0.5, reason="Test reason", metrics={})  # Changed

        reward_fn = RewardFunction(func=reward_func, mode="local")

        # Test with missing role
        with pytest.raises(ValueError):
            reward_fn(
                messages=[{"content": "Hello"}],
                ground_truth=[{"role": "user", "content": "Hello"}],  # Changed
            )

        # Test with missing content
        with pytest.raises(ValueError):
            reward_fn(
                messages=[{"role": "user"}],
                ground_truth=[{"role": "user", "content": "Hello"}],  # Changed
            )

    def test_remote_error_handling(self):
        """Test error handling in remote mode."""
        with patch.object(reward_function_module_obj, "requests") as mock_requests_module:
            # Mock the response to simulate an error
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_requests_module.post.return_value = mock_response

            reward_fn = RewardFunction(endpoint="https://example.com/reward", mode="remote")

            # Should raise an exception due to server error
            with pytest.raises(Exception):
                reward_fn(
                    messages=[{"role": "user", "content": "Hello"}],
                    ground_truth=[{"role": "user", "content": "Hello"}],  # Changed
                )

    def test_large_message_handling(self):
        """Test handling of very large messages."""

        def reward_func(messages, ground_truth, **kwargs):  # Changed
            """Function that processes large messages."""
            # Just calculate based on message length
            content_length = len(messages[-1]["content"])
            length_score = min(content_length / 10000.0, 1.0)
            metrics = {
                "length": MetricResult(  # Changed
                    score=length_score,
                    reason=f"Message length: {content_length} chars",
                    success=length_score == 1.0,
                )
            }
            return EvaluateResult(score=0.5, reason="Large message processed", metrics=metrics)  # Changed

        reward_fn = RewardFunction(func=reward_func, mode="local")

        # Generate a large message (10KB)
        large_content = "x" * 10000

        # This should not raise any exceptions
        result = reward_fn(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": large_content},
            ],
            ground_truth=[{"role": "user", "content": "Hello"}],  # Changed
        )

        assert result.score == 0.5
        assert "length" in result.metrics
        assert result.metrics["length"].score == 1.0  # Max score due to length

    def test_unicode_handling(self):
        """Test handling of unicode characters in messages."""

        def reward_func(messages, ground_truth, **kwargs):  # Changed
            """Function that handles Unicode."""
            # Just return a simple score and the message
            content = messages[-1]["content"]
            metrics = {
                "content": MetricResult(  # Changed
                    score=0.5,
                    reason=f"Processed: {content}",
                    success=True,  # Assuming success for this metric
                )
            }
            return EvaluateResult(score=0.5, reason="Unicode message processed", metrics=metrics)  # Changed

        reward_fn = RewardFunction(func=reward_func, mode="local")

        # Test with various Unicode characters
        unicode_message = "Hello 你好 こんにちは Привет 👋 🌍 ☀️"

        result = reward_fn(
            messages=[
                {"role": "user", "content": "Greet me in different languages"},
                {"role": "assistant", "content": unicode_message},
            ],
            ground_truth=[{"role": "user", "content": "Greet me in different languages"}],  # Changed
        )

        assert result.score == 0.5
        assert "content" in result.metrics
        assert unicode_message in result.metrics["content"].reason

        # Ensure the output can be serialized to JSON and back
        json_str = json.dumps(result.model_dump())  # Changed to model_dump()
        parsed = json.loads(json_str)
        assert parsed["metrics"]["content"]["reason"] == f"Processed: {unicode_message}"
