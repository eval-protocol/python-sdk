"""
Tests for kwargs validation in the reward_function decorator.
"""

from typing import Any, Dict, List

import pytest

from eval_protocol.models import EvaluateResult, MetricResult
from eval_protocol.typed_interface import reward_function


class TestKwargsValidation:
    """Test that the @reward_function decorator validates kwargs acceptance."""

    def test_function_with_kwargs_passes_validation(self):
        """Test that a function with **kwargs passes validation."""

        @reward_function
        def valid_func_with_kwargs(
            messages: List[Dict[str, str]], ground_truth: Any = None, **kwargs
        ) -> EvaluateResult:
            """A valid function that accepts **kwargs."""
            return EvaluateResult(
                score=0.8,
                reason="Test with kwargs",
                metrics={"test": MetricResult(score=0.8, is_score_valid=True, reason="Valid")},
            )

        # Test that the function works properly
        messages = [{"role": "user", "content": "Hello"}]
        result = valid_func_with_kwargs(messages=messages, extra_param="test")

        assert isinstance(result, EvaluateResult)
        assert result.score == 0.8

    def test_function_without_kwargs_raises_error(self):
        """Test that a function without **kwargs raises a validation error."""

        with pytest.raises(
            ValueError,
            match=r"Function 'invalid_func_no_kwargs' must accept \*\*kwargs parameter",
        ):

            @reward_function
            def invalid_func_no_kwargs(messages: List[Dict[str, str]], ground_truth: Any = None) -> EvaluateResult:
                """An invalid function that doesn't accept **kwargs."""
                return EvaluateResult(score=0.5, reason="Test")

    def test_function_without_kwargs_batch_mode_raises_error(self):
        """Test that a batch function without **kwargs raises a validation error."""

        with pytest.raises(
            ValueError,
            match=r"Function 'invalid_batch_func' must accept \*\*kwargs parameter",
        ):

            @reward_function(mode="batch")
            def invalid_batch_func(
                rollouts_messages: List[List[Dict[str, str]]],
            ) -> List[EvaluateResult]:
                """An invalid batch function that doesn't accept **kwargs."""
                return [EvaluateResult(score=0.5, reason="Test")]

    def test_function_with_named_keyword_args_only_raises_error(self):
        """Test that a function with only named keyword args (no **kwargs) raises error."""

        with pytest.raises(
            ValueError,
            match=r"Function 'func_with_named_kwargs_only' must accept \*\*kwargs parameter",
        ):

            @reward_function
            def func_with_named_kwargs_only(
                messages: List[Dict[str, str]],
                ground_truth: Any = None,
                specific_param: str = "default",
            ) -> EvaluateResult:
                """Function with named keyword args but no **kwargs."""
                return EvaluateResult(score=0.5, reason="Test")

    def test_function_with_both_named_kwargs_and_var_kwargs_passes(self):
        """Test that a function with both named kwargs and **kwargs passes validation."""

        @reward_function
        def func_with_both(
            messages: List[Dict[str, str]],
            ground_truth: Any = None,
            specific_param: str = "default",
            **kwargs,
        ) -> EvaluateResult:
            """Function with both named kwargs and **kwargs."""
            return EvaluateResult(
                score=0.7,
                reason=f"Test with specific_param={specific_param}",
                metrics={"test": MetricResult(score=0.7, is_score_valid=True, reason="Valid")},
            )

        # Test that the function works properly
        messages = [{"role": "user", "content": "Hello"}]
        result = func_with_both(messages=messages, specific_param="custom", extra_param="test")

        assert isinstance(result, EvaluateResult)
        assert result.score == 0.7
        assert result.reason is not None
        assert "specific_param=custom" in result.reason

    def test_function_with_args_and_kwargs_passes(self):
        """Test that a function with *args and **kwargs passes validation."""

        @reward_function
        def func_with_args_kwargs(*args, **kwargs) -> EvaluateResult:
            """Function with *args and **kwargs."""
            return EvaluateResult(
                score=0.6,
                reason="Test with args and kwargs",
                metrics={"test": MetricResult(score=0.6, is_score_valid=True, reason="Valid")},
            )

        # Test that the function works properly
        messages = [{"role": "user", "content": "Hello"}]
        result = func_with_args_kwargs(messages=messages, extra_param="test")

        assert isinstance(result, EvaluateResult)
        assert result.score == 0.6

    def test_error_message_includes_function_name(self):
        """Test that the error message includes the specific function name."""

        with pytest.raises(ValueError) as exc_info:

            @reward_function
            def my_specific_function_name(
                messages: List[Dict[str, str]],
            ) -> EvaluateResult:
                return EvaluateResult(score=0.5, reason="Test")

        error_message = str(exc_info.value)
        assert "my_specific_function_name" in error_message
        assert "must accept **kwargs parameter" in error_message
        assert "Please add '**kwargs' to the function signature" in error_message

    def test_valid_function_can_use_kwargs(self):
        """Test that a valid function can actually use the kwargs passed to it."""

        @reward_function
        def func_using_kwargs(messages: List[Dict[str, str]], ground_truth: Any = None, **kwargs) -> EvaluateResult:
            """Function that uses kwargs in its logic."""
            custom_score = kwargs.get("custom_score", 0.5)
            custom_reason = kwargs.get("custom_reason", "Default reason")

            return EvaluateResult(
                score=custom_score,
                reason=custom_reason,
                metrics={
                    "custom": MetricResult(
                        score=custom_score,
                        is_score_valid=True,
                        reason=f"Used kwargs: {list(kwargs.keys())}",
                    )
                },
            )

        # Test that the function uses the kwargs
        messages = [{"role": "user", "content": "Hello"}]
        result = func_using_kwargs(
            messages=messages,
            custom_score=0.9,
            custom_reason="Custom test reason",
            extra_param="test_value",
        )

        assert isinstance(result, EvaluateResult)
        assert result.score == 0.9
        assert result.reason == "Custom test reason"
        assert "custom_score" in result.metrics["custom"].reason
        assert "custom_reason" in result.metrics["custom"].reason
        assert "extra_param" in result.metrics["custom"].reason
