"""
Unit tests for rollout processor error handling.

Tests that rollout processors properly set rollout_status.status = "error" when exceptions occur.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from eval_protocol.dataset_logger import default_logger
from eval_protocol.models import EvaluationRow, Message, RolloutStatus
from eval_protocol.pytest.default_agent_rollout_processor import default_agent_rollout_processor
from eval_protocol.pytest.default_single_turn_rollout_process import default_single_turn_rollout_processor
from eval_protocol.pytest.types import RolloutProcessorConfig


class TestRolloutErrorHandling:
    """Test that rollout processors handle errors correctly."""

    @pytest.mark.asyncio
    async def test_agent_rollout_processor_429_error(self):
        """Test that agent rollout processor handles 429 rate limit errors correctly."""

        # Create test row with initialized rollout_status
        test_row = EvaluationRow(
            messages=[Message(role="user", content="Hello")], rollout_status=RolloutStatus(status="running")
        )

        config = RolloutProcessorConfig(
            model="gpt-4", input_params={}, mcp_config_path="", logger=default_logger  # Empty to avoid MCP setup
        )

        # Mock the LiteLLM policy to raise a 429 error
        with patch("eval_protocol.pytest.default_agent_rollout_processor.LiteLLMPolicy") as mock_policy_class:
            # Create a mock policy instance
            mock_policy = AsyncMock()
            mock_policy_class.return_value = mock_policy

            # Mock the _make_llm_call method to raise a 429 error
            import litellm

            mock_policy._make_llm_call.side_effect = litellm.RateLimitError(
                message="Rate limit exceeded: 429", llm_provider="openai", model="gpt-4"
            )

            # The agent rollout processor should catch the exception and set error status
            result = await default_agent_rollout_processor([test_row], config)

            assert len(result) == 1
            assert result[0].rollout_status.status == "error"
            assert result[0].rollout_status.error_message is not None
            assert (
                "429" in result[0].rollout_status.error_message
                or "rate limit" in result[0].rollout_status.error_message.lower()
            )

    @pytest.mark.asyncio
    async def test_agent_rollout_processor_bad_request_error(self):
        """Test that agent rollout processor handles BadRequest errors correctly."""

        test_row = EvaluationRow(
            messages=[Message(role="user", content="Hello")], rollout_status=RolloutStatus(status="running")
        )

        config = RolloutProcessorConfig(model="gpt-4", input_params={}, mcp_config_path="", logger=default_logger)

        # Mock the LiteLLM policy to raise a BadRequest error like the one in your example
        with patch("eval_protocol.pytest.default_agent_rollout_processor.LiteLLMPolicy") as mock_policy_class:
            mock_policy = AsyncMock()
            mock_policy_class.return_value = mock_policy

            import openai

            mock_policy._make_llm_call.side_effect = openai.BadRequestError(
                "Invalid value for 'content': expected a string, got null.", response=MagicMock(), body=None
            )

            result = await default_agent_rollout_processor([test_row], config)

            assert len(result) == 1
            assert result[0].rollout_status.status == "error"
            assert result[0].rollout_status.error_message is not None
            assert (
                "content" in result[0].rollout_status.error_message
                or "BadRequest" in result[0].rollout_status.error_message
            )

    @pytest.mark.asyncio
    async def test_single_turn_rollout_processor_429_error(self):
        """Test that single turn rollout processor handles 429 rate limit errors correctly."""

        test_row = EvaluationRow(
            messages=[Message(role="user", content="Hello")], rollout_status=RolloutStatus(status="running")
        )

        config = RolloutProcessorConfig(model="gpt-4", input_params={}, mcp_config_path="", logger=default_logger)

        # Mock litellm.acompletion to raise a 429 error
        with patch("importlib.import_module") as mock_import:
            mock_litellm = MagicMock()
            mock_import.return_value = mock_litellm

            import litellm

            mock_litellm.acompletion.side_effect = litellm.RateLimitError(
                message="Rate limit exceeded: 429", llm_provider="openai", model="gpt-4"
            )

            result = await default_single_turn_rollout_processor([test_row], config)

            assert len(result) == 1
            assert result[0].rollout_status.status == "error"
            assert result[0].rollout_status.error_message is not None
            assert (
                "429" in result[0].rollout_status.error_message
                or "rate limit" in result[0].rollout_status.error_message.lower()
            )

    @pytest.mark.asyncio
    async def test_single_turn_rollout_processor_bad_request_error(self):
        """Test that single turn rollout processor handles BadRequest errors correctly."""

        test_row = EvaluationRow(
            messages=[Message(role="user", content="Hello")], rollout_status=RolloutStatus(status="running")
        )

        config = RolloutProcessorConfig(model="gpt-4", input_params={}, mcp_config_path="", logger=default_logger)

        # Mock litellm.acompletion to raise a BadRequest error
        with patch("importlib.import_module") as mock_import:
            mock_litellm = MagicMock()
            mock_import.return_value = mock_litellm

            import openai

            mock_litellm.acompletion.side_effect = openai.BadRequestError(
                "Invalid value for 'content': expected a string, got null.", response=MagicMock(), body=None
            )

            result = await default_single_turn_rollout_processor([test_row], config)

            assert len(result) == 1
            assert result[0].rollout_status.status == "error"
            assert result[0].rollout_status.error_message is not None
            assert (
                "content" in result[0].rollout_status.error_message
                or "BadRequest" in result[0].rollout_status.error_message
            )

    @pytest.mark.asyncio
    async def test_multiple_rows_with_mixed_errors(self):
        """Test that when some rows get 429 errors and some succeed, each gets the correct status."""

        # Create test rows
        row1 = EvaluationRow(
            messages=[Message(role="user", content="Hello 1")], rollout_status=RolloutStatus(status="running")
        )

        row2 = EvaluationRow(
            messages=[Message(role="user", content="Hello 2")], rollout_status=RolloutStatus(status="running")
        )

        config = RolloutProcessorConfig(model="gpt-4", input_params={}, mcp_config_path="", logger=default_logger)

        # Mock litellm.acompletion to raise 429 for both rows (simulating rate limiting)
        with patch("importlib.import_module") as mock_import:
            mock_litellm = MagicMock()
            mock_import.return_value = mock_litellm

            import litellm

            mock_litellm.acompletion.side_effect = litellm.RateLimitError(
                message="Rate limit exceeded: 429", llm_provider="openai", model="gpt-4"
            )

            result = await default_single_turn_rollout_processor([row1, row2], config)

            assert len(result) == 2
            # Both should have error status due to 429 errors
            for row in result:
                assert row.rollout_status.status == "error"
                assert row.rollout_status.error_message is not None
                assert (
                    "429" in row.rollout_status.error_message
                    or "rate limit" in row.rollout_status.error_message.lower()
                )

    @pytest.mark.asyncio
    async def test_rollout_status_preserves_original_row_data_on_api_error(self):
        """Test that when API errors occur, the original row data is preserved."""

        original_message = Message(role="user", content="Original message")
        test_row = EvaluationRow(messages=[original_message], rollout_status=RolloutStatus(status="running"))

        config = RolloutProcessorConfig(model="gpt-4", input_params={}, mcp_config_path="", logger=default_logger)

        # Mock the LiteLLM policy to raise an API error
        with patch("eval_protocol.pytest.default_agent_rollout_processor.LiteLLMPolicy") as mock_policy_class:
            mock_policy = AsyncMock()
            mock_policy_class.return_value = mock_policy

            import litellm

            mock_policy._make_llm_call.side_effect = litellm.RateLimitError(
                message="Rate limit exceeded: 429", llm_provider="openai", model="gpt-4"
            )

            result = await default_agent_rollout_processor([test_row], config)

            assert len(result) == 1
            assert result[0].rollout_status.status == "error"
            # Original message should be preserved
            assert len(result[0].messages) == 1
            assert result[0].messages[0].content == "Original message"

    def test_rollout_status_initialization(self):
        """Test that RolloutStatus initializes with correct default values."""

        # Test default initialization
        status = RolloutStatus()
        assert status.status == "finished"  # Default from the model
        assert status.error_message is None

        # Test explicit initialization
        status = RolloutStatus(status="error", error_message="Test error")
        assert status.status == "error"
        assert status.error_message == "Test error"
