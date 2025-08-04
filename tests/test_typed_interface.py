"""
Tests for the typed interface functionality.
"""

import asyncio
from typing import Any, Dict, List

import pytest

from eval_protocol.models import EvaluateResult, Message, MetricResult
from eval_protocol.typed_interface import reward_function


def test_typed_interface_basic():
    """Test that the typed_interface decorator works with basic inputs."""

    @reward_function
    def sample_evaluator(messages: List[Message], **kwargs) -> EvaluateResult:
        """Sample evaluator that returns a hardcoded result."""
        return EvaluateResult(
            score=0.8,
            reason="Overall test reason",
            metrics={"test": MetricResult(success=True, score=0.8, reason="Test reason")},
        )

    # Test with valid messages
    valid_messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    result = sample_evaluator(messages=valid_messages)

    # Check the output - Pydantic object access
    assert isinstance(result, EvaluateResult)
    assert result.score == 0.8
    assert result.reason == "Overall test reason"
    assert result.metrics is not None
    assert "test" in result.metrics
    metric_test = result.metrics["test"]
    assert isinstance(metric_test, MetricResult)
    assert metric_test.is_score_valid is True
    assert metric_test.score == 0.8
    assert metric_test.reason == "Test reason"

    # Check dictionary-style access
    assert result["score"] == 0.8
    assert result["reason"] == "Overall test reason"
    assert result["metrics"] is not None  # Accesses the dict of MetricResult objects
    assert "test" in result["metrics"]
    metric_test_dict_access = result["metrics"]["test"]  # This is a MetricResult object
    assert isinstance(metric_test_dict_access, MetricResult)
    assert metric_test_dict_access["is_score_valid"] is True  # MetricResult also has __getitem__
    assert metric_test_dict_access["score"] == 0.8
    assert metric_test_dict_access["reason"] == "Test reason"


def test_typed_interface_input_validation():
    """
    Test that input messages are properly validated.

    Our Message model now properly validates messages, requiring at least 'role' and 'content'.
    """

    @reward_function
    def sample_evaluator(messages: List[Message], **kwargs) -> EvaluateResult:
        # Check that we can access the messages
        assert len(messages) > 0

        return EvaluateResult(
            score=0.5,
            reason="Overall test",
            metrics={"test": MetricResult(success=True, score=0.5, reason="Test")},
        )

    # Valid messages with required fields
    valid_messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    # This should work without error
    result = sample_evaluator(messages=valid_messages)
    assert isinstance(result, EvaluateResult)
    assert result.score == 0.5
    assert result.metrics["test"].score == 0.5

    # Check dictionary-style access
    assert result["score"] == 0.5
    assert result["metrics"]["test"]["score"] == 0.5

    # Test with invalid messages should raise an error
    invalid_messages = [{"content": "Hello without role"}]

    try:
        sample_evaluator(messages=invalid_messages)
        assert False, "Should have raised a validation error"
    except ValueError:
        # Expected validation error
        pass

    # Test with other unusual formats
    unusual_role_messages = [
        {"role": "custom_role", "content": "Hello"},  # Non-standard role
        {"role": "assistant", "content": "Hi there!"},
    ]

    # This should also work without raising an error
    result = sample_evaluator(messages=unusual_role_messages)
    assert isinstance(result, EvaluateResult)
    assert result.score == 0.5
    assert result.metrics["test"].score == 0.5

    # Check dictionary-style access
    assert result["score"] == 0.5
    assert result["metrics"]["test"]["score"] == 0.5


def test_typed_interface_output_validation():
    """Test that the typed_interface validates output correctly."""

    @reward_function
    def invalid_evaluator(messages: List[Message], **kwargs) -> EvaluateResult:
        # Return an incomplete metric (missing required fields)
        # This should be caught by the output validation
        return EvaluateResult(
            score=0.5,
            reason=None,
            metrics={
                "test": {"score": 0.5},  # type: ignore[dict-item] # Missing 'success'
            },
        )  # type: ignore

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    with pytest.raises(ValueError) as excinfo:
        invalid_evaluator(messages=messages)

    # For pydantic v2, the error format has changed but should contain 'validation errors'
    assert "validation error" in str(excinfo.value).lower()


def test_typed_interface_kwargs():
    """Test that the typed_interface correctly passes through kwargs."""

    @reward_function
    def kwargs_evaluator(messages: List[Message], **kwargs) -> EvaluateResult:
        # Return the kwargs in the reason field
        return EvaluateResult(
            score=0.5,
            reason="Overall test with kwargs",
            metrics={
                "test": MetricResult(
                    success=True,
                    score=0.5,
                    reason=f"Got kwargs: {sorted([k for k in kwargs.keys()])}",
                )
            },
        )

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    result = kwargs_evaluator(messages=messages, param1="test", param2=123)
    assert isinstance(result, EvaluateResult)

    # Pydantic object access
    assert result.metrics is not None
    assert "test" in result.metrics
    assert "Got kwargs: ['param1', 'param2']" == result.metrics["test"].reason

    # Dictionary-style access
    assert result["metrics"] is not None
    assert "test" in result["metrics"]
    assert "Got kwargs: ['param1', 'param2']" == result["metrics"]["test"]["reason"]


def test_typed_interface_model_dump():
    """Test that the typed_interface works with model_dump() return format."""

    @reward_function
    def sample_evaluator(messages: List[Message], **kwargs) -> EvaluateResult:
        """Sample evaluator with error field."""
        return EvaluateResult(
            score=0.8,
            reason="Overall test reason",
            error="Sample error message",
            metrics={"test": MetricResult(success=True, score=0.8, reason="Test reason")},
        )

    valid_messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    result = sample_evaluator(messages=valid_messages)

    # Check the output format matches what's expected - Pydantic object access
    assert isinstance(result, EvaluateResult)
    assert result.score == 0.8
    assert result.reason == "Overall test reason"
    assert result.error == "Sample error message"
    assert result.metrics is not None
    assert "test" in result.metrics
    metric_test = result.metrics["test"]
    assert isinstance(metric_test, MetricResult)
    assert metric_test.is_score_valid is True
    assert metric_test.score == 0.8
    assert metric_test.reason == "Test reason"

    # Check dictionary-style access
    assert result["score"] == 0.8
    assert result["reason"] == "Overall test reason"
    assert result["error"] == "Sample error message"
    assert result["metrics"] is not None
    assert "test" in result["metrics"]
    metric_test_dict_access = result["metrics"]["test"]
    assert isinstance(metric_test_dict_access, MetricResult)
    assert metric_test_dict_access["is_score_valid"] is True
    assert metric_test_dict_access["score"] == 0.8
    assert metric_test_dict_access["reason"] == "Test reason"


def test_async_reward_function():
    """Test that the typed_interface works with async functions."""

    @reward_function
    async def async_evaluator(messages: List[Message], **kwargs) -> EvaluateResult:
        """Sample async evaluator that returns a hardcoded result."""
        return EvaluateResult(score=0.8, reason="Overall test reason", is_score_valid=True)

    async def _test_async_reward_function():
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        result = await async_evaluator(messages=messages)
        assert isinstance(result, EvaluateResult)
        assert result.score == 0.8
        assert result.reason == "Overall test reason"

    asyncio.run(_test_async_reward_function())


def test_reward_function_decorator_attributes():
    """Test that the reward_function decorator sets attributes correctly."""

    @reward_function(mode="batch", requirements=["requests", "numpy"], concurrency=10, timeout=10)
    def sample_evaluator(messages: List[Message], **kwargs) -> EvaluateResult:
        """Sample evaluator that returns a hardcoded result."""
        return EvaluateResult(score=0.8, reason="Overall test reason", metrics={})

    assert sample_evaluator._reward_function_mode == "batch"
    assert sample_evaluator._reward_function_requirements == ["requests", "numpy"]
    assert sample_evaluator._reward_function_concurrency == 10
    assert sample_evaluator._reward_function_timeout == 10
