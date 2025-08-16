import json
from typing import Any, Dict, List, Tuple, cast
from unittest.mock import MagicMock, patch

import pytest

from eval_protocol.models import (
    EvaluateResult,  # Changed
    Message,  # Added import
)
from eval_protocol.rewards.function_calling import (
    calculate_jaccard_similarity,
    composite_function_call_reward,
    exact_tool_match_reward,
    extract_schema_properties,
    llm_judge_reward,
    match_function_call,
    schema_jaccard_reward,
)


class TestFunctionCalling:
    """Tests for the function_calling reward module."""

    def test_exact_match(self):
        """Test exact match of function name and arguments."""
        expected_schema = {
            "name": "get_weather",
            "arguments": {
                "location": {"type": "string"},
                "unit": {"type": "string"},
            },
        }

        parsed_name = "get_weather"
        parsed_args = {"location": "New York", "unit": "celsius"}

        result = match_function_call(
            messages=cast(
                List[Dict[str, Any]],
                [
                    {"role": "user", "content": "What's the weather?"},
                    {"role": "assistant", "content": "Let me check the weather."},
                ],
            ),
            function_name=parsed_name,
            parsed_arguments=parsed_args,
            expected_call_schema=expected_schema,
            argument_match_strictness="exact",
        )

        assert isinstance(result, EvaluateResult)
        # Attribute access
        assert result.score == 1.0
        assert "function_name_match" in result.metrics
        assert "arguments_match" in result.metrics
        assert result.metrics["function_name_match"].score == 1.0
        assert result.metrics["arguments_match"].score == 1.0
        # Dictionary access
        assert result["score"] == 1.0
        assert result["metrics"]["function_name_match"]["score"] == 1.0
        assert result["metrics"]["arguments_match"]["score"] == 1.0

    def test_wrong_function_name(self):
        """Test with incorrect function name."""
        expected_schema = {
            "name": "get_weather",
            "arguments": {
                "location": {"type": "string"},
                "unit": {"type": "string"},
            },
        }

        parsed_name = "fetch_weather"  # Wrong name
        parsed_args = {"location": "New York", "unit": "celsius"}

        result = match_function_call(
            messages=cast(
                List[Dict[str, Any]],
                [
                    {"role": "user", "content": "What's the weather?"},
                    {"role": "assistant", "content": "Let me check the weather."},
                ],
            ),
            function_name=parsed_name,
            parsed_arguments=parsed_args,
            expected_call_schema=expected_schema,
            argument_match_strictness="exact",
        )

        assert isinstance(result, EvaluateResult)
        # Attribute access
        assert result.score < 1.0
        assert "function_name_match" in result.metrics
        assert result.metrics["function_name_match"].score == 0.0
        assert (
            result.metrics["function_name_match"].reason is not None
            and "Function name does not match" in result.metrics["function_name_match"].reason  # type: ignore[operator]
        )
        # Dictionary access
        assert result["score"] < 1.0
        assert result["metrics"]["function_name_match"]["score"] == 0.0
        assert (
            result["metrics"]["function_name_match"]["reason"] is not None
            and "Function name does not match" in result["metrics"]["function_name_match"]["reason"]  # type: ignore[operator]
        )

    def test_missing_required_argument(self):
        """Test with missing required argument."""
        expected_schema = {
            "name": "get_weather",
            "arguments": {
                "location": {"type": "string"},
                "unit": {"type": "string"},
            },
        }

        parsed_name = "get_weather"
        parsed_args = {
            "location": "New York"
            # Missing "unit" argument
        }

        result = match_function_call(
            messages=cast(
                List[Dict[str, Any]],
                [
                    {"role": "user", "content": "What's the weather?"},
                    {"role": "assistant", "content": "Let me check the weather."},
                ],
            ),
            function_name=parsed_name,
            parsed_arguments=parsed_args,
            expected_call_schema=expected_schema,
            argument_match_strictness="exact",
        )

        assert isinstance(result, EvaluateResult)
        # Attribute access
        assert result.score < 1.0
        assert "arguments_match" in result.metrics
        assert result.metrics["arguments_match"].score < 1.0
        assert (
            result.metrics["arguments_match"].reason is not None
            and "Missing argument" in result.metrics["arguments_match"].reason  # type: ignore[operator]
        )
        # Dictionary access
        assert result["score"] < 1.0
        assert result["metrics"]["arguments_match"]["score"] < 1.0
        assert (
            result["metrics"]["arguments_match"]["reason"] is not None
            and "Missing argument" in result.metrics["arguments_match"]["reason"]  # type: ignore[operator]
        )

    def test_extra_argument(self):
        """Test with extra argument not in schema."""
        expected_schema = {
            "name": "get_weather",
            "arguments": {
                "location": {"type": "string"},
                "unit": {"type": "string"},
            },
        }

        parsed_name = "get_weather"
        parsed_args = {
            "location": "New York",
            "unit": "celsius",
            "extra_param": "value",  # Extra argument
        }

        result = match_function_call(
            messages=cast(
                List[Dict[str, Any]],
                [
                    {"role": "user", "content": "What's the weather?"},
                    {"role": "assistant", "content": "Let me check the weather."},
                ],
            ),
            function_name=parsed_name,
            parsed_arguments=parsed_args,
            expected_call_schema=expected_schema,
            argument_match_strictness="exact",
        )

        assert isinstance(result, EvaluateResult)
        # Attribute access
        assert result.score < 1.0
        assert "arguments_match" in result.metrics
        assert result.metrics["arguments_match"].score < 1.0
        assert (
            result.metrics["arguments_match"].reason is not None
            and "Unexpected argument" in result.metrics["arguments_match"].reason  # type: ignore[operator]
        )
        # Dictionary access
        assert result["score"] < 1.0
        assert result["metrics"]["arguments_match"]["score"] < 1.0
        assert (
            result["metrics"]["arguments_match"]["reason"] is not None
            and "Unexpected argument" in result.metrics["arguments_match"]["reason"]  # type: ignore[operator]
        )

    def test_permissive_mode(self):
        """Test permissive mode with extra arguments."""
        expected_schema = {
            "name": "get_weather",
            "arguments": {
                "location": {"type": "string"},
                "unit": {"type": "string"},
            },
        }

        parsed_name = "get_weather"
        parsed_args = {
            "location": "New York",
            "unit": "celsius",
            "extra_param": "value",  # Extra argument
        }

        result = match_function_call(
            messages=cast(
                List[Dict[str, Any]],
                [
                    {"role": "user", "content": "What's the weather?"},
                    {"role": "assistant", "content": "Let me check the weather."},
                ],
            ),
            function_name=parsed_name,
            parsed_arguments=parsed_args,
            expected_call_schema=expected_schema,
            argument_match_strictness="permissive",  # Permissive mode
        )

        assert isinstance(result, EvaluateResult)
        # In permissive mode, extra arguments are allowed
        # Attribute access
        assert result.score == 1.0
        assert "function_name_match" in result.metrics
        assert "arguments_match" in result.metrics
        assert result.metrics["function_name_match"].score == 1.0
        assert result.metrics["arguments_match"].score == 1.0
        # Dictionary access
        assert result["score"] == 1.0
        assert result["metrics"]["function_name_match"]["score"] == 1.0
        assert result["metrics"]["arguments_match"]["score"] == 1.0

    def test_wrong_argument_value_type(self):
        """Test with wrong argument value type."""
        expected_schema = {
            "name": "get_weather",
            "arguments": {
                "location": {"type": "string"},
                "temperature": {"type": "number"},
            },
        }

        parsed_name = "get_weather"
        parsed_args = {
            "location": "New York",
            "temperature": "25",  # String instead of number
        }

        result = match_function_call(
            messages=cast(
                List[Dict[str, Any]],
                [
                    {"role": "user", "content": "What's the weather?"},
                    {"role": "assistant", "content": "Let me check the weather."},
                ],
            ),
            function_name=parsed_name,
            parsed_arguments=parsed_args,
            expected_call_schema=expected_schema,
            argument_match_strictness="exact",
        )

        assert isinstance(result, EvaluateResult)
        # Attribute access
        assert result.score < 1.0
        assert "arguments_match" in result.metrics
        assert result.metrics["arguments_match"].score < 1.0
        assert (
            result.metrics["arguments_match"].reason is not None
            and "Type mismatch" in result.metrics["arguments_match"].reason  # type: ignore[operator]
        )
        # Dictionary access
        assert result["score"] < 1.0
        assert result["metrics"]["arguments_match"]["score"] < 1.0
        assert (
            result["metrics"]["arguments_match"]["reason"] is not None
            and "Type mismatch" in result.metrics["arguments_match"]["reason"]  # type: ignore[operator]
        )

    def test_calculate_jaccard_similarity(self):
        """Test Jaccard similarity calculation."""
        # Perfect match
        set1 = {"a", "b", "c"}
        set2 = {"a", "b", "c"}
        similarity = calculate_jaccard_similarity(set1, set2)
        assert similarity == 1.0

        # No overlap
        set1 = {"a", "b", "c"}
        set2 = {"d", "e", "f"}
        similarity = calculate_jaccard_similarity(set1, set2)
        assert similarity == 0.0

        # Partial overlap
        set1 = {"a", "b", "c"}
        set2 = {"b", "c", "d"}
        similarity = calculate_jaccard_similarity(set1, set2)
        assert similarity == 0.5  # 2/4 = 0.5

        # Empty sets
        set1 = set()
        set2 = set()
        similarity = calculate_jaccard_similarity(set1, set2)
        assert similarity == 1.0  # Both empty should be perfect match

    def test_extract_schema_properties(self):
        """Test extraction of properties from JSON schema."""
        # Simple schema
        schema = {
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
            }
        }
        properties = extract_schema_properties(schema)
        assert len(properties) == 2
        assert ("name", "string") in properties
        assert ("age", "number") in properties

        # Nested schema
        nested_schema: Dict[str, Any] = {
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "firstName": {"type": "string"},
                        "lastName": {"type": "string"},
                    },
                }
            }
        }
        properties = extract_schema_properties(nested_schema)
        assert len(properties) == 3
        assert ("user", "object") in properties
        assert ("user.firstName", "string") in properties
        assert ("user.lastName", "string") in properties

    def test_schema_jaccard_reward_exact_match(self):
        """Test schema_jaccard_reward now delegates to exact_tool_match_reward - Perfect Match."""
        # This test now verifies exact_tool_match_reward's behavior via schema_jaccard_reward
        assistant_message_content_with_tool_call = {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "New York", "unit": "celsius"}),
                    },
                }
            ],
        }
        ground_truth_data = {
            "role": "assistant",  # Role for ground_truth is illustrative, exact_tool_match_reward uses tool_calls from it
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "New York", "unit": "celsius"}),
                    },
                }
            ],
        }

        result = schema_jaccard_reward(  # This now calls exact_tool_match_reward
            messages=cast(
                List[Dict[str, Any]],
                [
                    {"role": "user", "content": "What's the weather?"},
                    assistant_message_content_with_tool_call,
                ],
            ),
            ground_truth=ground_truth_data,
            # expected_schema is no longer used by the delegated function
        )

        assert isinstance(result, EvaluateResult)
        assert result.score == 1.0
        assert isinstance(result.reason, str)
        assert result.reason is not None and "Exact tool match evaluation score: 1.0" in result.reason

    def test_schema_jaccard_reward_mismatch(self):
        """Test schema_jaccard_reward now delegates to exact_tool_match_reward - Mismatch."""
        assistant_message_content_with_tool_call = {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "New York", "unit": "fahrenheit"}),
                    },  # Different unit
                }
            ],
        }
        ground_truth_data = {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "New York", "unit": "celsius"}),
                    },
                }
            ]
        }

        result = schema_jaccard_reward(
            messages=cast(
                List[Dict[str, Any]],
                [
                    {"role": "user", "content": "What's the weather?"},
                    assistant_message_content_with_tool_call,
                ],
            ),
            ground_truth=ground_truth_data,
        )
        assert isinstance(result, EvaluateResult)
        assert result.score == 0.0
        assert isinstance(result.reason, str)
        assert result.reason is not None and "Exact tool match evaluation score: 0.0" in result.reason

    def test_schema_jaccard_reward_wrong_function_name(self):
        """Test schema_jaccard_reward (delegating) with wrong function name."""
        assistant_message_content_with_tool_call = {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "fetch_weather_data",
                        "arguments": json.dumps({"location": "New York", "unit": "celsius"}),
                    },
                }
            ],
        }
        ground_truth_data = {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "New York", "unit": "celsius"}),
                    },
                }
            ]
        }
        result = schema_jaccard_reward(
            messages=cast(
                List[Dict[str, Any]],
                [
                    {"role": "user", "content": "What's the weather?"},
                    assistant_message_content_with_tool_call,
                ],
            ),
            ground_truth=ground_truth_data,
        )
        assert isinstance(result, EvaluateResult)
        assert result.score == 0.0
        assert isinstance(result.reason, str)
        assert result.reason is not None and "Exact tool match evaluation score: 0.0" in result.reason

    def test_nested_schema_exact_match(self):  # Renamed for clarity
        """Test exact_tool_match_reward (via schema_jaccard_reward) with nested objects."""
        assistant_message_content_with_tool_call = {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "create_user",
                        "arguments": json.dumps(
                            {
                                "user": {
                                    "firstName": "John",
                                    "lastName": "Doe",
                                    "age": 30,
                                }
                            }
                        ),
                    },
                }
            ],
        }
        ground_truth_data = {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "create_user",
                        "arguments": json.dumps(
                            {
                                "user": {
                                    "firstName": "John",
                                    "lastName": "Doe",
                                    "age": 30,
                                }
                            }
                        ),
                    },
                }
            ]
        }
        result = schema_jaccard_reward(  # This now calls exact_tool_match_reward
            messages=cast(
                List[Dict[str, Any]],
                [
                    {"role": "user", "content": "Create a user for John Doe"},
                    assistant_message_content_with_tool_call,
                ],
            ),
            ground_truth=ground_truth_data,
        )
        assert isinstance(result, EvaluateResult)
        assert result.score == 1.0
        assert isinstance(result.reason, str)
        assert result.reason is not None and "Exact tool match evaluation score: 1.0" in result.reason

    # Remove @patch for OpenAI as llm_judge_reward now delegates
    def test_llm_judge_reward_delegation(self):  # Renamed and simplified
        """Test llm_judge_reward now delegates to exact_tool_match_reward."""
        assistant_message_content_with_tool_call = {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "New York", "unit": "celsius"}),
                    },
                }
            ],
        }
        ground_truth_data = {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "New York", "unit": "celsius"}),
                    },
                }
            ]
        }

        result = llm_judge_reward(  # This now calls exact_tool_match_reward
            messages=cast(
                List[Dict[str, Any]],
                [
                    {"role": "user", "content": "What's the weather?"},
                    assistant_message_content_with_tool_call,
                ],
            ),
            ground_truth=ground_truth_data,
            # Other params like expected_schema, expected_behavior, openai_api_key are no longer used by the core logic
        )

        assert isinstance(result, EvaluateResult)
        assert result.score == 1.0
        assert isinstance(result.reason, str)
        assert result.reason is not None and "Exact tool match evaluation score: 1.0" in result.reason
        # Ensure no LLM-specific metrics are present if the delegation is clean
        assert "llm_judge" not in result.metrics

    # Remove @patch for OpenAI as composite_function_call_reward now delegates
    def test_composite_function_call_reward_delegation(self):  # Renamed and simplified
        """Test composite_function_call_reward now delegates to exact_tool_match_reward."""
        assistant_message_content_with_tool_call = {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "New York", "unit": "celsius"}),
                    },
                }
            ],
        }
        ground_truth_data = {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "New York", "unit": "celsius"}),
                    },
                }
            ]
        }

        result = composite_function_call_reward(  # This now calls exact_tool_match_reward
            messages=cast(
                List[Dict[str, Any]],
                [
                    {"role": "user", "content": "What's the weather?"},
                    assistant_message_content_with_tool_call,
                ],
            ),
            ground_truth=ground_truth_data,
            # Other params like expected_schema, expected_behavior, weights are no longer used by the core logic
        )

        assert isinstance(result, EvaluateResult)
        assert result.score == 1.0
        assert isinstance(result.reason, str)
        assert result.reason is not None and "Exact tool match evaluation score: 1.0" in result.reason
        # Ensure no composite-specific metrics (like schema_score, llm_score, weights) are present
        assert "schema_score" not in result.metrics
        assert "llm_score" not in result.metrics
        assert "weights" not in result.metrics


# The JSON schema tests have been moved to tests/test_json_schema.py


class TestExactToolMatchReward:
    """Tests for the exact_tool_match_reward function."""

    def test_perfect_match_single_call(self):
        """Test perfect match with a single tool call."""
        messages = [
            {"role": "user", "content": "What's the weather in London?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"location": "London", "unit": "celsius"}),
                        },
                    }
                ],
            },
        ]
        ground_truth: Dict[str, Any] = {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "London", "unit": "celsius"}),
                    },
                }
            ]
        }
        result = exact_tool_match_reward(messages=messages, ground_truth=ground_truth)
        assert result.score == 1.0
        assert result.reason is not None and "Exact tool match evaluation score: 1.0" in result.reason

    def test_perfect_match_multiple_calls_ordered(self):
        """Test perfect match with multiple tool calls in correct order."""
        messages = [
            {
                "role": "user",
                "content": "Weather in London and book a flight to Paris.",
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"location": "London"}),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "book_flight",
                            "arguments": json.dumps({"destination": "Paris", "date": "2024-12-01"}),
                        },
                    },
                ],
            },
        ]
        ground_truth = {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "London"}),
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "book_flight",
                        "arguments": json.dumps({"destination": "Paris", "date": "2024-12-01"}),
                    },
                },
            ]
        }
        result = exact_tool_match_reward(messages=messages, ground_truth=ground_truth)
        assert result.score == 1.0

    def test_mismatch_multiple_calls_order(self):
        """Test mismatch due to incorrect order of multiple tool calls."""
        messages = [
            {
                "role": "user",
                "content": "Weather in London and book a flight to Paris.",
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "book_flight",  # Called first
                            "arguments": json.dumps({"destination": "Paris", "date": "2024-12-01"}),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",  # Called second
                            "arguments": json.dumps({"location": "London"}),
                        },
                    },
                ],
            },
        ]
        ground_truth = {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "London"}),
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "book_flight",
                        "arguments": json.dumps({"destination": "Paris", "date": "2024-12-01"}),
                    },
                },
            ]
        }
        result = exact_tool_match_reward(messages=messages, ground_truth=ground_truth)
        assert result.score == 0.0
        assert result.reason is not None and "Exact tool match evaluation score: 0.0" in result.reason

    def test_mismatch_function_name(self):
        """Test mismatch in function name."""
        messages = [
            {"role": "user", "content": "What's the weather in London?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "fetch_weather_forecast",  # Wrong name
                            "arguments": json.dumps({"location": "London", "unit": "celsius"}),
                        },
                    }
                ],
            },
        ]
        ground_truth = {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "London", "unit": "celsius"}),
                    },
                }
            ]
        }
        result = exact_tool_match_reward(messages=messages, ground_truth=ground_truth)
        assert result.score == 0.0

    def test_mismatch_argument_value(self):
        """Test mismatch in argument value."""
        messages = [
            {"role": "user", "content": "What's the weather in London?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"location": "London", "unit": "fahrenheit"}),  # Wrong unit
                        },
                    }
                ],
            },
        ]
        ground_truth = {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "London", "unit": "celsius"}),
                    },
                }
            ]
        }
        result = exact_tool_match_reward(messages=messages, ground_truth=ground_truth)
        assert result.score == 0.0

    def test_mismatch_argument_name(self):
        """Test mismatch in argument name."""
        messages = [
            {"role": "user", "content": "What's the weather in London?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"city": "London", "unit": "celsius"}),  # Wrong arg name 'city'
                        },
                    }
                ],
            },
        ]
        ground_truth = {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "London", "unit": "celsius"}),
                    },
                }
            ]
        }
        result = exact_tool_match_reward(messages=messages, ground_truth=ground_truth)
        assert result.score == 0.0

    def test_mismatch_number_of_calls_gen_more(self):
        """Test mismatch when generation has more tool calls than ground truth."""
        messages = [
            {"role": "user", "content": "What's the weather in London?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"location": "London"}),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "extra_call",
                            "arguments": json.dumps({}),
                        },
                    },
                ],
            },
        ]
        ground_truth = {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "London"}),
                    },
                }
            ]
        }
        result = exact_tool_match_reward(messages=messages, ground_truth=ground_truth)
        assert result.score == 0.0

    def test_mismatch_number_of_calls_gt_more(self):
        """Test mismatch when ground truth has more tool calls than generation."""
        messages = [
            {"role": "user", "content": "What's the weather in London?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"location": "London"}),
                        },
                    }
                ],
            },
        ]
        ground_truth = {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "London"}),
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "expected_extra_call",
                        "arguments": json.dumps({}),
                    },
                },
            ]
        }
        result = exact_tool_match_reward(messages=messages, ground_truth=ground_truth)
        assert result.score == 0.0

    def test_gen_has_calls_gt_none(self):
        """Test when generation has tool calls but ground truth expects none."""
        messages = [
            {"role": "user", "content": "Tell me a joke."},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",  # Unexpected call
                            "arguments": json.dumps({"location": "London"}),
                        },
                    }
                ],
            },
        ]
        ground_truth: Dict[str, Any] = {"tool_calls": []}  # Expects no tool calls
        result = exact_tool_match_reward(messages=messages, ground_truth=ground_truth)
        assert result.score == 0.0

    def test_gen_no_calls_gt_expects_some(self):
        """Test when generation has no tool calls but ground truth expects some."""
        messages = [
            {"role": "user", "content": "What's the weather in London?"},
            {"role": "assistant", "content": "It might be sunny."},  # No tool call
        ]
        ground_truth = {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "London"}),
                    },
                }
            ]
        }
        result = exact_tool_match_reward(messages=messages, ground_truth=ground_truth)
        assert result.score == 0.0

    def test_gen_no_calls_gt_no_calls(self):
        """Test when neither generation nor ground truth have tool calls."""
        messages = [
            {"role": "user", "content": "Tell me a joke."},
            {"role": "assistant", "content": "Why did the chicken cross the road?"},
        ]
        ground_truth: Dict[str, Any] = {"tool_calls": []}
        result = exact_tool_match_reward(messages=messages, ground_truth=ground_truth)
        assert result.score == 1.0

    def test_parse_from_content_xml(self):
        """Test parsing tool call from content if tool_calls attribute is missing."""
        messages = [
            {"role": "user", "content": "What's the weather in Berlin?"},
            {
                "role": "assistant",
                "content": '<tool_call>{"type": "function", "function": {"name": "get_weather", "arguments": "{\\"location\\": \\"Berlin\\", \\"unit\\": \\"celsius\\"}"}}</tool_call>',
            },  # No tool_calls attribute, should parse from content
        ]
        ground_truth = {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "Berlin", "unit": "celsius"}),
                    },
                }
            ]
        }
        result = exact_tool_match_reward(messages=messages, ground_truth=ground_truth)
        assert result.score == 1.0

    def test_parse_from_content_xml_mismatch(self):
        """Test parsing tool call from content with a mismatch."""
        messages = [
            {"role": "user", "content": "What's the weather in Berlin?"},
            {
                "role": "assistant",
                "content": '<tool_call>{"type": "function", "function": {"name": "get_weather", "arguments": "{\\"location\\": \\"Paris\\", \\"unit\\": \\"celsius\\"}"}}</tool_call>',
            },
        ]
        ground_truth = {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "Berlin", "unit": "celsius"}),
                    },
                }
            ]
        }
        result = exact_tool_match_reward(messages=messages, ground_truth=ground_truth)
        assert result.score == 0.0

    def test_empty_messages_list(self):
        """Test with an empty messages list."""
        result = exact_tool_match_reward(messages=[], ground_truth={})
        assert result.score == 0.0
        assert result.reason is not None and "No messages provided" in result.reason

    def test_ground_truth_none(self):
        """Test with ground_truth being None."""
        # Case 1: Generation has tool calls
        messages_with_calls = [
            {"role": "user", "content": "Query"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "some_func", "arguments": "{}"},
                    }
                ],
            },
        ]
        result1 = exact_tool_match_reward(messages=messages_with_calls, ground_truth=None)
        assert result1.score == 0.0
        assert result1.reason is not None and "Ground truth not provided" in result1.reason

        # Case 2: Generation has no tool calls
        messages_without_calls = [
            {"role": "user", "content": "Query"},
            {"role": "assistant", "content": "No calls here."},
        ]
        result2 = exact_tool_match_reward(messages=messages_without_calls, ground_truth=None)
        assert result2.score == 1.0
        assert result2.reason is not None and "Ground truth not provided" in result2.reason

    def test_ground_truth_missing_tool_calls_key(self):
        """Test with ground_truth dict missing the 'tool_calls' key."""
        messages = [
            {"role": "user", "content": "Query"},
            {"role": "assistant", "content": "Assistant response"},
        ]
        ground_truth: Dict[str, Any] = {"some_other_key": []}  # Missing 'tool_calls'
        result = exact_tool_match_reward(messages=messages, ground_truth=ground_truth)
        # This implies no tool calls are expected, so if assistant also makes no calls, score is 1.0
        assert result.score == 1.0

    def test_non_json_arguments_string(self):
        """Test with arguments string that is not valid JSON."""
        messages = [
            {"role": "user", "content": "Query"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "test_func",
                            "arguments": "not a json string",
                        },
                    }
                ],
            },
        ]
        ground_truth = {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "test_func", "arguments": "not a json string"},
                }
            ]
        }
        # The maybe_deserialize_tool_call_arguments leaves non-JSON string as is.
        # So if GT also has the same non-JSON string, it should match.
        result = exact_tool_match_reward(messages=messages, ground_truth=ground_truth)
        assert result.score == 1.0

        ground_truth_diff_string: Dict[str, Any] = {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "test_func",
                        "arguments": "another non json string",
                    },
                }
            ]
        }
        result_diff = exact_tool_match_reward(messages=messages, ground_truth=ground_truth_diff_string)
        assert result_diff.score == 0.0

    def test_message_object_input(self):
        """Test with Message objects as input."""
        messages = [
            Message(role="user", content="What's the weather in London?"),
            Message(
                role="assistant",
                tool_calls=[
                    {
                        "id": "call_test123",  # Added ID field
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"location": "London", "unit": "celsius"}),
                        },
                    }
                ],
            ),
        ]
        ground_truth: Dict[str, Any] = {
            "tool_calls": [
                {
                    # ID in ground_truth doesn't need to match the one in assistant message,
                    # as exact_tool_match_reward primarily compares function name and arguments.
                    # However, the structure for GT tool_calls also includes 'type' and 'function'.
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "London", "unit": "celsius"}),
                    },
                }
            ]
        }
        result = exact_tool_match_reward(messages=messages, ground_truth=ground_truth)
        assert result.score == 1.0
