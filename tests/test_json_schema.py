import json
from typing import Any, Dict  # Added Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from eval_protocol.models import EvaluateResult  # Changed
from eval_protocol.rewards.json_schema import (
    json_schema_reward,
    json_schema_reward_with_llm_judge,
)


class TestJsonSchemaReward:
    """Tests for the json_schema reward module."""

    def test_json_schema_reward_exact_match(self):
        """Test JSON schema reward with exact match."""
        expected_schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "number"},
                        "isActive": {"type": "boolean"},
                    },
                },
                "products": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "price": {"type": "number"},
                        },
                    },
                },
            },
        }

        json_content = {
            "user": {"name": "John Doe", "age": 30, "isActive": True},
            "products": [{"id": "prod-1", "price": 29.99}],
        }

        result = json_schema_reward(
            messages=[
                {"role": "user", "content": "Give me user information"},
                {"role": "assistant", "content": "Here's the user information"},
            ],
            ground_truth=None,
            json_content=json_content,
            expected_schema=expected_schema,
        )

        assert isinstance(result, EvaluateResult)
        # Attribute access
        assert result.score > 0.9  # Should be very high
        assert "schema_similarity" in result.metrics
        assert result.metrics["schema_similarity"].score > 0.9
        # Dictionary access
        assert result["score"] > 0.9
        assert "schema_similarity" in result["metrics"]
        assert result["metrics"]["schema_similarity"]["score"] > 0.9

    def test_json_schema_reward_partial_match(self):
        """Test JSON schema reward with partial match."""
        expected_schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "number"},
                        "isActive": {"type": "boolean"},
                        "address": {
                            "type": "object",
                            "properties": {
                                "street": {"type": "string"},
                                "city": {"type": "string"},
                            },
                        },
                    },
                }
            },
        }

        json_content = {
            "user": {
                "name": "John Doe",
                "age": 30,
                "email": "john@example.com",  # Extra property not in schema
                # Missing isActive and address
            }
        }

        result = json_schema_reward(
            messages=[
                {"role": "user", "content": "Give me user information"},
                {"role": "assistant", "content": "Here's the user information"},
            ],
            ground_truth=None,
            json_content=json_content,
            expected_schema=expected_schema,
        )

        assert isinstance(result, EvaluateResult)
        # Attribute access
        assert 0.3 < result.score < 0.8  # Should be in middle range
        assert "schema_similarity" in result.metrics
        assert 0.3 < result.metrics["schema_similarity"].score < 0.8
        assert (
            result.metrics["schema_similarity"].reason is not None
            and "Missing properties" in result.metrics["schema_similarity"].reason
        )
        assert (
            result.metrics["schema_similarity"].reason is not None
            and "Extra properties" in result.metrics["schema_similarity"].reason
        )
        # Dictionary access
        assert 0.3 < result["score"] < 0.8
        assert "schema_similarity" in result["metrics"]
        assert 0.3 < result["metrics"]["schema_similarity"]["score"] < 0.8
        assert (
            result["metrics"]["schema_similarity"]["reason"] is not None
            and "Missing properties" in result["metrics"]["schema_similarity"]["reason"]
        )
        assert (
            result["metrics"]["schema_similarity"]["reason"] is not None
            and "Extra properties" in result["metrics"]["schema_similarity"]["reason"]
        )

    def test_json_schema_reward_extract_from_message(self):
        """Test JSON schema reward with JSON extracted from message content."""
        expected_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
            },
        }

        message_with_json = {
            "role": "assistant",
            "content": 'Here\'s the user information:\n```json\n{\n  "name": "John Doe",\n  "age": 30\n}\n```',
        }

        result = json_schema_reward(
            messages=[
                {"role": "user", "content": "Give me user information"},
                message_with_json,
            ],
            ground_truth=None,
            expected_schema=expected_schema,
        )

        assert isinstance(result, EvaluateResult)
        # Attribute access
        assert result.score > 0.9  # Should be very high
        assert "schema_similarity" in result.metrics
        assert result.metrics["schema_similarity"].score > 0.9
        # Dictionary access
        assert result["score"] > 0.9
        assert "schema_similarity" in result["metrics"]
        assert result["metrics"]["schema_similarity"]["score"] > 0.9

    def test_json_schema_reward_mismatched_types(self):
        """Test JSON schema reward with mismatched property types."""
        expected_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
                "tags": {"type": "array"},
            },
        }

        json_content = {
            "name": "John Doe",
            "age": "30",  # String instead of number
            "tags": {"tag1": "value1"},  # Object instead of array
        }

        result = json_schema_reward(
            messages=[
                {"role": "user", "content": "Give me user information"},
                {"role": "assistant", "content": "Here's the user information"},
            ],
            ground_truth=None,
            json_content=json_content,
            expected_schema=expected_schema,
        )

        assert isinstance(result, EvaluateResult)
        # Attribute access
        assert result.score < 0.7  # Should be lower due to type mismatches
        assert "schema_similarity" in result.metrics
        assert result.metrics["schema_similarity"].score < 0.7
        assert (
            result.metrics["schema_similarity"].reason is not None
            and "Matching properties" in result.metrics["schema_similarity"].reason
        )
        # Dictionary access
        assert result["score"] < 0.7
        assert "schema_similarity" in result["metrics"]
        assert result["metrics"]["schema_similarity"]["score"] < 0.7
        assert (
            result["metrics"]["schema_similarity"]["reason"] is not None
            and "Matching properties" in result["metrics"]["schema_similarity"]["reason"]
        )

    def test_json_schema_reward_with_json_string(self):
        """Test JSON schema reward with JSON string instead of object."""
        expected_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
            },
        }

        json_string = '{"name": "John Doe", "age": 30}'

        result = json_schema_reward(
            messages=[
                {"role": "user", "content": "Give me user information"},
                {"role": "assistant", "content": "Here's the user information"},
            ],
            ground_truth=None,
            json_content=json_string,
            expected_schema=expected_schema,
        )

        assert isinstance(result, EvaluateResult)
        # Attribute access
        assert result.score > 0.9  # Should be very high
        assert "schema_similarity" in result.metrics
        assert result.metrics["schema_similarity"].score > 0.9
        # Dictionary access
        assert result["score"] > 0.9
        assert "schema_similarity" in result["metrics"]
        assert result["metrics"]["schema_similarity"]["score"] > 0.9

    def test_json_schema_reward_empty_properties(self):
        """Test JSON schema reward with empty properties."""
        expected_schema = {"type": "object", "properties": {}}

        json_content: Dict[str, Any] = {}

        result = json_schema_reward(
            messages=[
                {"role": "user", "content": "Give me an empty object"},
                {"role": "assistant", "content": "Here's an empty object"},
            ],
            ground_truth=None,
            json_content=json_content,
            expected_schema=expected_schema,
        )

        assert isinstance(result, EvaluateResult)
        # Attribute access
        assert result.score == 1.0  # Should be perfect match
        assert "schema_similarity" in result.metrics
        assert result.metrics["schema_similarity"].score == 1.0
        # Dictionary access
        assert result["score"] == 1.0
        assert "schema_similarity" in result["metrics"]
        assert result["metrics"]["schema_similarity"]["score"] == 1.0

    def test_json_schema_reward_invalid_json(self):
        """Test JSON schema reward with invalid JSON."""
        expected_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
            },
        }

        invalid_json = '{"name": "John Doe", "age": }'  # Invalid JSON

        result = json_schema_reward(
            messages=[
                {"role": "user", "content": "Give me user information"},
                {"role": "assistant", "content": "Here's the user information"},
            ],
            ground_truth=None,
            json_content=invalid_json,
            expected_schema=expected_schema,
        )

        assert isinstance(result, EvaluateResult)
        # Attribute access
        assert result.score == 0.0
        assert "error" in result.metrics
        assert result.metrics["error"].reason is not None and "Invalid JSON content" in result.metrics["error"].reason
        # Dictionary access
        assert result["score"] == 0.0
        assert "error" in result["metrics"]
        assert (
            result["metrics"]["error"]["reason"] is not None
            and "Invalid JSON content" in result["metrics"]["error"]["reason"]
        )

    @patch("openai.OpenAI")
    def test_json_schema_reward_with_llm_judge(self, mock_openai_class):
        """Test JSON schema reward with LLM judge integration."""
        # Mock the OpenAI client response
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = (
            "SCORE: 0.85\nEXPLANATION: This is a good JSON structure that matches the expected schema."
        )
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        expected_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
            },
        }

        json_content = {"name": "John Doe", "age": 30}

        result = json_schema_reward_with_llm_judge(
            messages=[
                {"role": "user", "content": "Give me user information"},
                {"role": "assistant", "content": "Here's the user information"},
            ],
            ground_truth=None,
            json_content=json_content,
            expected_schema=expected_schema,
            expected_behavior="Provide a user object with name and age",
            openai_api_key="fake_key_for_testing",
        )

        assert isinstance(result, EvaluateResult)
        # Attribute access
        assert "schema_score" in result.metrics
        assert "llm_score" in result.metrics
        assert "weights" in result.metrics
        assert 0.9 < result.score < 1.0
        # Dictionary access
        assert "schema_score" in result["metrics"]
        assert "llm_score" in result["metrics"]
        assert "weights" in result["metrics"]
        assert 0.9 < result["score"] < 1.0

        # Check that the LLM was called with the expected schema and content
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        assert "messages" in call_args
        assert any("John Doe" in str(msg.get("content", "")) for msg in call_args["messages"])

    def test_json_schema_reward_with_llm_judge_custom_weights(self):
        """Test JSON schema reward with LLM judge using custom weights."""
        with patch("openai.OpenAI") as mock_openai_class:
            # Mock the OpenAI client response
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client

            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = "SCORE: 0.60\nEXPLANATION: This JSON is acceptable but has some issues."
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response

            expected_schema = {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "number"},
                },
            }

            json_content = {"name": "John Doe", "age": 30}

            # Use custom weights: schema=0.3, llm=0.7
            result = json_schema_reward_with_llm_judge(
                messages=[
                    {"role": "user", "content": "Give me user information"},
                    {
                        "role": "assistant",
                        "content": "Here's the user information",
                    },
                ],
                ground_truth=None,
                json_content=json_content,
                expected_schema=expected_schema,
                expected_behavior="Provide a user object with name and age",
                openai_api_key="fake_key_for_testing",
                weights={"schema": 0.3, "llm": 0.7},
            )

            assert isinstance(result, EvaluateResult)
            # Attribute access
            # Schema score should be ~1.0, LLM score is mocked at 0.60
            # Expected final score ≈ 0.3*1.0 + 0.7*0.60 ≈ 0.72
            assert 0.7 < result.score < 0.75
            assert (
                result.metrics["weights"].reason is not None and "0.30" in result.metrics["weights"].reason
            )  # Schema weight
            assert (
                result.metrics["weights"].reason is not None and "0.70" in result.metrics["weights"].reason
            )  # LLM weight
            # Dictionary access
            assert 0.7 < result["score"] < 0.75
            assert (
                result["metrics"]["weights"]["reason"] is not None and "0.30" in result["metrics"]["weights"]["reason"]
            )
            assert (
                result["metrics"]["weights"]["reason"] is not None and "0.70" in result["metrics"]["weights"]["reason"]
            )
