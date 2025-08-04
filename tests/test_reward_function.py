import sys  # Import sys
from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock, patch

import pytest

# Ensure the module is loaded (though RewardFunction import likely does this)
import eval_protocol
from eval_protocol import RewardFunction, reward_function

# Get a direct reference to the module object
reward_function_module_obj = sys.modules["eval_protocol.reward_function"]
from eval_protocol.models import EvaluateResult, MetricResult  # Changed


def simple_reward_func(
    messages: List[Dict[str, str]],
    ground_truth: Optional[Union[str, List[Dict[str, str]]]] = None,
    **kwargs,
) -> EvaluateResult:  # Changed
    """Example reward function for testing."""
    metrics = {"length": MetricResult(score=0.5, reason="Length-based score", success=True)}  # Changed
    return EvaluateResult(score=0.5, reason="Simple reward", metrics=metrics)  # Changed


@reward_function
def decorated_reward_func(
    messages: List[Dict[str, str]],
    ground_truth: Optional[Union[str, List[Dict[str, str]]]] = None,
    **kwargs,
) -> EvaluateResult:  # Changed
    """Example decorated reward function."""
    metrics = {"test": MetricResult(score=0.7, reason="Test score", success=True)}  # Changed
    return EvaluateResult(score=0.7, reason="Decorated reward", metrics=metrics)  # Changed


class TestRewardFunction:
    """Tests for the RewardFunction class."""

    def test_local_mode_function_path(self):
        """Test RewardFunction in local mode with function path."""
        with patch.object(reward_function_module_obj, "importlib") as mock_importlib_module:
            mock_module = MagicMock()
            mock_module.simple_reward_func = simple_reward_func
            mock_importlib_module.import_module.return_value = mock_module

            reward_fn = RewardFunction(func_path="test_module.simple_reward_func", mode="local")

            test_msgs = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]
            orig_msgs = [test_msgs[0]]

            result = reward_fn(messages=test_msgs, ground_truth=orig_msgs)
            assert result.score == 0.5
            assert "length" in result.metrics
            assert result.metrics["length"].score == 0.5

    def test_local_mode_function(self):
        """Test RewardFunction in local mode with direct function."""
        reward_fn = RewardFunction(func=simple_reward_func, mode="local")

        test_msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        orig_msgs = [test_msgs[0]]

        result = reward_fn(messages=test_msgs, ground_truth=orig_msgs)
        assert result.score == 0.5
        assert "length" in result.metrics
        assert result.metrics["length"].score == 0.5

    def test_remote_mode(self):
        """Test RewardFunction in remote mode."""
        with patch.object(reward_function_module_obj, "requests") as mock_requests_module:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "score": 0.8,
                "is_score_valid": True,
                "metrics": {
                    "remote": {
                        "score": 0.8,
                        "reason": "Remote score",
                        "is_score_valid": True,
                    }
                },
            }
            mock_requests_module.post.return_value = mock_response

            reward_fn = RewardFunction(endpoint="https://example.com/reward", mode="remote")

            test_msgs = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]
            orig_msgs = [test_msgs[0]]

            result = reward_fn(messages=test_msgs, ground_truth=orig_msgs)
            assert result.score == 0.8
            assert "remote" in result.metrics
            assert result.metrics["remote"].score == 0.8

    def test_fireworks_hosted_mode(self):
        """Test RewardFunction in fireworks_hosted mode."""
        with patch.object(reward_function_module_obj, "requests") as mock_requests_module:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "score": 0.9,
                "metrics": {"hosted": {"score": 0.9, "reason": "Hosted score"}},
            }
            mock_requests_module.post.return_value = mock_response

            reward_fn = RewardFunction(model_id="fireworks/test-model", mode="fireworks_hosted")

            test_msgs = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]
            orig_msgs = [test_msgs[0]]

            result = reward_fn(messages=test_msgs, ground_truth=orig_msgs)
            assert result.score == 0.9
            assert "hosted" in result.metrics
            assert result.metrics["hosted"].score == 0.9

    def test_get_trl_adapter(self):
        """Test getting a TRL adapter from a RewardFunction."""
        reward_fn = RewardFunction(func=simple_reward_func, mode="local")

        trl_adapter = reward_fn.get_trl_adapter()
        assert callable(trl_adapter)

        # Test the adapter with a batch input
        full_conversations_batch = [
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
            [  # Add another sample for more robust testing
                {"role": "user", "content": "Test query"},
                {"role": "assistant", "content": "Test response"},
            ],
        ]

        # Prepare prompts and completions correctly for the adapter
        # Prompts are lists of messages up to (but not including) the assistant's turn
        prompts_batch = [conv[:-1] for conv in full_conversations_batch]
        # Completions are strings of the assistant's responses
        completions_batch = [conv[-1]["content"] for conv in full_conversations_batch]

        result = trl_adapter(prompts=prompts_batch, completions=completions_batch)
        assert isinstance(result, list)
        assert len(result) == 2  # We have two samples now
        assert result[0] == 0.5  # simple_reward_func always returns 0.5
        assert result[1] == 0.5  # simple_reward_func always returns 0.5


class TestRewardFunctionDecorator:
    """Tests for the @reward_function decorator."""

    def test_decorator_basic_functionality(self):
        """Test basic functionality of the reward_function decorator."""
        test_msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        orig_msgs = [test_msgs[0]]

        # Call the decorated function directly
        result = decorated_reward_func(messages=test_msgs, ground_truth=orig_msgs)
        # The legacy_reward_function (imported as reward_function here)
        # should return an EvaluateResult object.
        assert isinstance(result, EvaluateResult)

        # Pydantic attribute access
        assert result.score == 0.7
        assert result.reason == "Decorated reward"
        assert result.metrics is not None
        assert "test" in result.metrics
        metric_test = result.metrics["test"]
        assert isinstance(metric_test, MetricResult)
        assert metric_test.score == 0.7
        assert metric_test.reason == "Test score"
        assert metric_test.is_score_valid is True

        # Dictionary-style access (since EvaluateResult is now hybrid)
        assert result["score"] == 0.7
        assert result["reason"] == "Decorated reward"
        assert result["metrics"] is not None
        assert "test" in result["metrics"]
        metric_test_dict_access = result["metrics"]["test"]  # This is a MetricResult object
        assert isinstance(metric_test_dict_access, MetricResult)
        assert metric_test_dict_access["score"] == 0.7
        assert metric_test_dict_access["reason"] == "Test score"
        assert metric_test_dict_access["is_score_valid"] is True

    def test_decorator_deploy_method(self):
        """Test that the new decorator does NOT add a deploy method directly."""
        # The new reward_function from typed_interface does not add .deploy directly
        assert not hasattr(decorated_reward_func, "deploy")

        # The following lines testing the .deploy() method are removed as
        # decorated_reward_func (using the new decorator) does not have this method.
        # Deployment for functions decorated with the new decorator might be handled
        # by the RewardFunction class or a separate utility.

        # # Directly patch the requests.post call for simplicity
        # with patch("eval_protocol.reward_function.requests.post") as mock_post:
        #     # Configure the response
        #     mock_response = MagicMock()
        #     mock_response.status_code = 200
        #     mock_response.json.return_value = {
        #         "name": "accounts/test-account/evaluators/test-123"
        #     }
        #     mock_post.return_value = mock_response

        #     # Test deploy method by providing account_id directly in the config
        #     # This would fail as decorated_reward_func.deploy does not exist
        #     deploy_result = decorated_reward_func.deploy(
        #         name="test-deployment",
        #         account_id="test-account",  # Provide account_id directly
        #         auth_token="fake-token",  # Provide token directly
        #     )

        #     # Check the result is the evaluation ID
        #     assert deploy_result == "test-123"

        #     # Verify the API was called
        #     mock_post.assert_called_once()
        #     args, kwargs = mock_post.call_args
        #     assert "accounts/test-account/evaluators" in args[0]
        #     assert kwargs["headers"]["Authorization"] == "Bearer fake-token"
