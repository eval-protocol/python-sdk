from typing import Any, Dict, List, Union

import pytest
from pydantic import ValidationError

from eval_protocol.agent.models import StepData

# Assuming these are the correct import paths based on our plan
from eval_protocol.models import EvaluateResult, Message as RewardKitMessage, StepOutput

# Minimal Message for StepData if direct import from eval_protocol.models is problematic in tests
# For now, assume RewardKitMessage from eval_protocol.models works.


class TestRLDataStructures:
    def test_step_output_creation_valid(self):
        """Test valid creation of StepOutput."""
        so = StepOutput(step_index=0, base_reward=0.5, reason="Good step", metrics={"accuracy": 0.9})
        assert so.step_index == 0
        assert so.base_reward == 0.5
        assert so.reason == "Good step"
        assert so.metrics == {"accuracy": 0.9}

        so_str_index = StepOutput(step_index="turn_1", base_reward=-0.1)
        assert so_str_index.step_index == "turn_1"
        assert so_str_index.base_reward == -0.1
        assert so_str_index.metrics == {}
        assert so_str_index.reason is None

    def test_step_output_invalid_types(self):
        """Test StepOutput validation errors for incorrect types."""
        with pytest.raises(ValidationError):
            StepOutput(step_index="0", base_reward="not_a_float")  # base_reward should be float
        with pytest.raises(ValidationError):
            StepOutput(step_index=None, base_reward=0.5)  # step_index is required

    def test_evaluate_result_extended(self):
        """Test extended EvaluateResult with step_outputs."""
        step_out1 = StepOutput(step_index=0, base_reward=0.1)
        step_out2 = StepOutput(step_index="assistant_1", base_reward=0.2)

        er_with_steps = EvaluateResult(
            score=0.75,
            reason="Overall good",
            step_outputs=[step_out1, step_out2],
            # metrics field is now Dict[str, MetricResult], not part of this basic test
            # for simplicity, we'll test its default or assume it's handled elsewhere
        )
        assert er_with_steps.score == 0.75
        assert er_with_steps.step_outputs is not None
        assert len(er_with_steps.step_outputs) == 2
        assert er_with_steps.step_outputs[0].base_reward == 0.1
        assert er_with_steps.step_outputs[1].step_index == "assistant_1"

    def test_evaluate_result_backward_compatibility(self):
        """Test EvaluateResult creation without new RL fields."""
        # This test assumes MetricResult is defined and works as before.
        # For simplicity, we might skip deep MetricResult testing here if it's complex to mock.
        # Let's assume metrics can be an empty dict for this test if not focusing on MetricResult itself.
        er_old_style = EvaluateResult(score=0.9, reason="Simple score")
        assert er_old_style.score == 0.9
        assert er_old_style.reason == "Simple score"
        assert er_old_style.step_outputs is None
        assert er_old_style.metrics == {}  # Due to default_factory=dict

        er_with_empty_steps = EvaluateResult(score=0.6, step_outputs=[])
        assert er_with_empty_steps.score == 0.6
        assert er_with_empty_steps.step_outputs == []

    def test_evaluate_result_invalid_step_outputs(self):
        """Test EvaluateResult with invalid step_outputs type."""
        with pytest.raises(ValidationError):
            EvaluateResult(score=0.5, step_outputs="not_a_list")
        with pytest.raises(ValidationError):
            EvaluateResult(score=0.5, step_outputs=[{"step_index": 0, "base_reward": "wrong_type"}])

    def test_step_data_creation_minimal(self):
        """Test minimal valid creation of StepData."""
        msg_hist = [RewardKitMessage(role="user", content="Hello")]
        action = {"type": "text", "content": "Hi"}
        step = StepData(
            system_step_index=0,
            observation_data={"history": msg_hist},  # Example observation
            action_taken=action,
            resulting_messages_history=[
                *msg_hist,
                RewardKitMessage(role="assistant", content="Hi"),
            ],
        )
        assert step.system_step_index == 0
        assert step.action_taken == action
        assert step.base_reward is None
        assert step.advantage is None
        assert step.is_done is False
        assert step.policy_value_estimate is None

    def test_step_data_creation_full(self):
        """Test StepData creation with all optional fields."""
        msg_hist1 = [RewardKitMessage(role="user", content="Hello")]
        msg_hist2 = [*msg_hist1, RewardKitMessage(role="assistant", content="Hi there")]
        step = StepData(
            system_step_index=1,
            observation_data=msg_hist1,
            action_taken={"type": "text", "content": "Hi there"},
            raw_policy_output="Hi there",
            resulting_messages_history=msg_hist2,
            policy_logprobs={"token_logprobs": [-0.1, -0.2]},
            policy_value_estimate=0.95,
            is_done=True,
            step_info={"tool_used": "none", "latency_ms": 100},
            base_reward=0.5,
            advantage=0.1,
            return_to_go=0.6,
        )
        assert step.policy_value_estimate == 0.95
        assert step.is_done is True
        assert step.base_reward == 0.5
        assert step.advantage == 0.1
        assert step.return_to_go == 0.6
        assert step.step_info["latency_ms"] == 100

    def test_step_data_field_validation(self):
        """Test StepData field type validations."""
        with pytest.raises(ValidationError):
            StepData(  # Missing required fields
                system_step_index=0,
                observation_data=[],
            )
        with pytest.raises(ValidationError):
            StepData(
                system_step_index="not_an_int",  # system_step_index should be int
                observation_data=[],
                action_taken={},
                resulting_messages_history=[],
            )
        with pytest.raises(ValidationError):
            StepData(
                system_step_index=0,
                observation_data=[],
                action_taken={},
                resulting_messages_history=[],
                is_done="not_a_bool",  # is_done should be bool
            )

    def test_step_data_message_import(self):
        """Test that Message can be used within StepData."""
        # This test implicitly checks if the Message import in eval_protocol.agent.models
        # is working or if the fallback is used. A more direct test might involve
        # checking the type of resulting_messages_history items if possible.
        m1 = RewardKitMessage(role="user", content="Test")
        m2 = RewardKitMessage(role="assistant", content="Response")
        step = StepData(
            system_step_index=0,
            observation_data=[m1],
            action_taken={"type": "text", "content": "Response"},
            resulting_messages_history=[m1, m2],
        )
        assert isinstance(step.resulting_messages_history[0], RewardKitMessage)
