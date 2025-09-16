import pytest
from typing import Any, Dict
from unittest.mock import Mock
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from eval_protocol.models import EvaluationRow, InputMetadata, ExecutionMetadata
from eval_protocol.pytest.default_pydantic_ai_rollout_processor import PydanticAgentRolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig


def test_pydantic_ai_metadata_only_stored_for_responses_model():
    """Test that PydanticAI metadata is only stored in extra_body for ResponsesModel, not for ChatModel."""

    # Create a test row with metadata
    row = EvaluationRow(
        input_metadata=InputMetadata(row_id="test-row-123"),
        execution_metadata=ExecutionMetadata(
            invocation_id="test-invocation-456",
            rollout_id="test-rollout-789",
            run_id="test-run-101",
            experiment_id="test-experiment-202",
        ),
        messages=[],
    )

    # Test with OpenAIChatModel (should NOT store metadata)
    chat_model = OpenAIChatModel("gpt-4")
    chat_agent = Agent(model=chat_model)
    processor = PydanticAgentRolloutProcessor(lambda config: chat_agent)

    settings = processor.construct_model_settings(chat_agent, row)

    # ChatModel should not have metadata in extra_body
    extra_body = settings.get("extra_body", {})
    assert isinstance(extra_body, dict), "extra_body should be a dict"
    assert "metadata" not in extra_body, "ChatModel should not store metadata in extra_body"

    # Test with OpenAIResponsesModel (should store metadata)
    responses_model = OpenAIResponsesModel("gpt-5")
    responses_agent = Agent(model=responses_model)
    processor_responses = PydanticAgentRolloutProcessor(lambda config: responses_agent)

    settings_responses = processor_responses.construct_model_settings(responses_agent, row)

    # ResponsesModel should have metadata in extra_body
    extra_body_responses = settings_responses.get("extra_body", {})
    assert isinstance(extra_body_responses, dict), "extra_body should be a dict"
    assert "metadata" in extra_body_responses, "ResponsesModel should store metadata in extra_body"

    metadata = extra_body_responses["metadata"]
    assert isinstance(metadata, dict), "metadata should be a dict"
    assert metadata["row_id"] == "test-row-123"
    assert metadata["invocation_id"] == "test-invocation-456"
    assert metadata["rollout_id"] == "test-rollout-789"
    assert metadata["run_id"] == "test-run-101"
    assert metadata["experiment_id"] == "test-experiment-202"


def test_pydantic_ai_metadata_handling_with_string_model():
    """Test that PydanticAI string models don't cause issues with metadata handling."""

    # Create a test row
    row = EvaluationRow(
        input_metadata=InputMetadata(row_id="test-row-123"),
        execution_metadata=ExecutionMetadata(
            invocation_id="test-invocation-456",
            rollout_id="test-rollout-789",
            run_id="test-run-101",
            experiment_id="test-experiment-202",
        ),
        messages=[],
    )

    # Create agent with string model (should not store metadata)
    agent = Agent(model="gpt-4")
    processor = PydanticAgentRolloutProcessor(lambda config: agent)

    settings = processor.construct_model_settings(agent, row)

    # String model should not have metadata in extra_body
    extra_body = settings.get("extra_body", {})
    assert isinstance(extra_body, dict), "extra_body should be a dict"
    assert "metadata" not in extra_body, "String model should not store metadata in extra_body"


def test_pydantic_ai_metadata_handling_with_none_model():
    """Test that PydanticAI None model doesn't cause issues with metadata handling."""

    # Create a test row
    row = EvaluationRow(
        input_metadata=InputMetadata(row_id="test-row-123"),
        execution_metadata=ExecutionMetadata(
            invocation_id="test-invocation-456",
            rollout_id="test-rollout-789",
            run_id="test-run-101",
            experiment_id="test-experiment-202",
        ),
        messages=[],
    )

    # Create agent with None model (should not store metadata)
    agent = Agent(model=None)
    processor = PydanticAgentRolloutProcessor(lambda config: agent)

    settings = processor.construct_model_settings(agent, row)

    # None model should not have metadata in extra_body
    extra_body = settings.get("extra_body", {})
    assert isinstance(extra_body, dict), "extra_body should be a dict"
    assert "metadata" not in extra_body, "None model should not store metadata in extra_body"
