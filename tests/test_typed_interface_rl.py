from typing import Any, Dict, List, Union

import pytest
from pydantic import ValidationError

from eval_protocol.models import EvaluateResult, Message, StepOutput
from eval_protocol.typed_interface import EvaluationMode, reward_function

# --- Mock User Reward Functions ---


@reward_function(mode="pointwise")
def pointwise_rl_rewards_func(messages: List[Message], ground_truth: Any, **kwargs) -> EvaluateResult:
    """Returns EvaluateResult with step_outputs for RL."""
    steps = []
    for i, msg in enumerate(messages):
        if msg.role == "assistant":
            steps.append(StepOutput(step_index=i, base_reward=0.1 * i, reason=f"Step {i} reward"))
    return EvaluateResult(score=0.5, step_outputs=steps, reason="Pointwise RL test")


@reward_function(mode="pointwise")
def pointwise_scoring_func(messages: List[Message], ground_truth: Any, **kwargs) -> EvaluateResult:
    """Returns EvaluateResult with only a score."""
    return EvaluateResult(score=0.8, reason="Pointwise scoring test")


@reward_function(mode="batch")
def batch_rl_rewards_func(rollouts_messages: List[List[Message]], ground_truth: Any, **kwargs) -> List[EvaluateResult]:
    """Returns List[EvaluateResult] with step_outputs for RL."""
    results = []
    for i, messages in enumerate(rollouts_messages):
        steps = []
        for j, msg in enumerate(messages):
            if msg.role == "assistant":
                steps.append(
                    StepOutput(
                        step_index=j,
                        base_reward=0.01 * i * j,
                        reason=f"Rollout {i} Step {j}",
                    )
                )
        results.append(EvaluateResult(score=0.6 + (0.1 * i), step_outputs=steps, reason=f"Batch RL test {i}"))
    return results


@reward_function(mode="batch")
def batch_scoring_func(rollouts_messages: List[List[Message]], ground_truth: Any, **kwargs) -> List[EvaluateResult]:
    """Returns List[EvaluateResult] with only scores."""
    results = []
    for i, messages in enumerate(rollouts_messages):
        results.append(EvaluateResult(score=0.9 - (0.1 * i), reason=f"Batch scoring test {i}"))
    return results


@reward_function(mode="pointwise")
def pointwise_invalid_output_func(messages: List[Message], ground_truth: Any, **kwargs) -> Dict:
    """Returns a dict, not an EvaluateResult."""
    return {"wrong_score_key": 0.5}  # Missing 'score'


@reward_function(mode="batch")
def batch_invalid_output_func(rollouts_messages: List[List[Message]], ground_truth: Any, **kwargs) -> List[Dict]:
    """Returns a list of dicts, not EvaluateResult."""
    return [{"score_is_actually_string": "0.5"}]


# --- Test Cases ---


class TestTypedInterfaceRL:
    def test_pointwise_rl_rewards_valid_input(self):
        """Test pointwise RL reward function with valid dict messages."""
        raw_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = pointwise_rl_rewards_func(messages=raw_messages, ground_truth=None)
        assert isinstance(result, EvaluateResult)
        assert result.score == 0.5
        assert result.step_outputs is not None
        assert len(result.step_outputs) == 1
        # Type assertion to help the linter understand step_outputs is not None
        step_outputs = result.step_outputs
        assert step_outputs is not None
        step_output = step_outputs[0]
        assert step_output.step_index == 1  # assistant message index
        assert step_output.base_reward == 0.1 * 1

    def test_pointwise_rl_rewards_with_pydantic_messages(self):
        """Test pointwise RL reward function with Pydantic Message objects."""
        pydantic_messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
        ]
        result = pointwise_rl_rewards_func(messages=pydantic_messages, ground_truth=None)
        assert isinstance(result, EvaluateResult)
        assert len(result.step_outputs) == 1

    def test_pointwise_scoring_valid(self):
        raw_messages = [{"role": "user", "content": "Test"}]
        result = pointwise_scoring_func(messages=raw_messages, ground_truth="test_gt")
        assert isinstance(result, EvaluateResult)
        assert result.score == 0.8
        assert result.step_outputs is None

    def test_batch_rl_rewards_valid_input(self):
        raw_rollouts = [
            [
                {"role": "user", "content": "R1U1"},
                {"role": "assistant", "content": "R1A1"},
            ],
            [
                {"role": "user", "content": "R2U1"},
                {"role": "assistant", "content": "R2A1"},
                {"role": "assistant", "content": "R2A2"},
            ],
        ]
        ground_truth = None
        results = batch_rl_rewards_func(rollouts_messages=raw_rollouts, ground_truth=ground_truth)
        assert isinstance(results, list)
        assert len(results) == 2
        assert isinstance(results[0], EvaluateResult)
        assert results[0].score == 0.6
        assert results[0].step_outputs is not None
        assert len(results[0].step_outputs) == 1
        assert results[0].step_outputs[0].base_reward == 0.0

        assert isinstance(results[1], EvaluateResult)
        assert results[1].score == 0.7
        assert results[1].step_outputs is not None
        assert len(results[1].step_outputs) == 2
        # The 2nd StepOutput (index 1) corresponds to the assistant message at original index j=2 in rollout i=1.
        # Reward was calculated as 0.01 * i * j = 0.01 * 1 * 2.
        assert results[1].step_outputs[1].base_reward == 0.01 * 1 * 2

    def test_batch_scoring_valid(self):
        raw_rollouts = [
            [{"role": "user", "content": "R1"}],
            [{"role": "user", "content": "R2"}],
        ]
        ground_truth = "gt"
        results = batch_scoring_func(rollouts_messages=raw_rollouts, ground_truth=ground_truth)
        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0].score == 0.9
        assert results[1].score == 0.8

    def test_pointwise_invalid_message_type(self):
        """Test pointwise with invalid message item type."""
        raw_messages = [{"role": "user", "content": "Hello"}, "not_a_dict"]
        with pytest.raises(ValueError, match="Input 'messages' failed Pydantic validation"):
            pointwise_rl_rewards_func(messages=raw_messages, ground_truth=None)

    def test_batch_invalid_inner_message_type(self):
        """Test batch with invalid message item type in one of the rollouts."""
        raw_rollouts = [
            [{"role": "user", "content": "R1U1"}],
            [{"role": "user", "content": "R2U1"}, "not_a_dict_msg"],
        ]
        with pytest.raises(ValueError, match="Input 'rollouts_messages' failed Pydantic validation"):
            batch_rl_rewards_func(rollouts_messages=raw_rollouts, ground_truth=None)

    def test_pointwise_invalid_output(self):
        """Test pointwise function returning incorrect type."""
        raw_messages = [{"role": "user", "content": "Test"}]
        with pytest.raises(
            ValueError,
            match=r"Return value from function 'pointwise_invalid_output_func' failed Pydantic validation for mode 'pointwise':\n1 validation error for EvaluateResult",
        ):
            pointwise_invalid_output_func(messages=raw_messages, ground_truth=None)

    def test_batch_invalid_output_item(self):
        """Test batch function returning list with incorrect item type."""
        raw_rollouts = [[{"role": "user", "content": "R1"}]]
        with pytest.raises(
            ValueError,
            match=r"Return value from function 'batch_invalid_output_func' failed Pydantic validation for mode 'batch':\n1 validation error for list\[EvaluateResult]",
        ):
            batch_invalid_output_func(rollouts_messages=raw_rollouts, ground_truth=None)

    def test_decorator_mode_mismatch_error_handling(self):
        """Test if calling a pointwise function as batch (or vice-versa) via wrapper would error.
        This depends on how the sandbox caller uses the mode, the decorator itself is now more flexible.
        The wrapper generated by the sandbox component should select the right call path.
        This test is more about ensuring the decorator doesn't break if func signature mismatches mode,
        though the type hints F should prevent direct misuse.
        Actual runtime check is more in the sandbox invoker.
        """
        # This scenario is hard to test directly at the decorator level without mocking inspect.signature
        # or having the sandbox invoker logic. The type hints F and explicit mode should guide usage.
        # For now, we assume the caller uses the 'mode' argument correctly.
        pass

    def test_ground_truth_coercion_pointwise(self):
        """Test ground_truth coercion for pointwise List[Message]."""

        @reward_function(mode="pointwise")
        def gt_test_func(messages: List[Message], ground_truth: List[Message], **kwargs):
            assert isinstance(ground_truth, list)
            assert all(isinstance(m, Message) for m in ground_truth)
            return EvaluateResult(score=1.0)

        raw_gt = [{"role": "system", "content": "gt_message"}]
        gt_test_func(messages=[{"role": "user", "content": "hi"}], ground_truth=raw_gt)

        with pytest.raises(ValueError, match="Input 'ground_truth' failed Pydantic validation"):
            gt_test_func(
                messages=[{"role": "user", "content": "hi"}],
                ground_truth=["not_a_message_dict"],
            )
