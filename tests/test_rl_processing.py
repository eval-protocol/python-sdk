from typing import (  # Optional was already here, this is fine.
    Any,
    Dict,
    List,
    Optional,
    Union,
)

import pytest

from eval_protocol.agent.models import StepData
from eval_protocol.models import EvaluateResult, Message as RewardKitMessage, StepOutput
from eval_protocol.rl_processing import RLDataAligner


class TestRLDataAligner:
    def create_mock_step_data(
        self,
        system_step_index: int,
        assistant_turn_index: Optional[Union[int, str]] = None,
        content: str = "assistant action",
    ) -> StepData:
        """Helper to create a StepData instance for testing."""
        obs_msgs = [RewardKitMessage(role="user", content="Hello")]
        action = {"type": "text", "content": content}
        res_msgs = [*obs_msgs, RewardKitMessage(role="assistant", content=content)]

        step_info = {}
        if assistant_turn_index is not None:
            step_info["assistant_turn_index"] = assistant_turn_index

        return StepData(
            system_step_index=system_step_index,
            observation_data=obs_msgs,
            action_taken=action,
            resulting_messages_history=res_msgs,
            step_info=step_info,
        )

    def test_align_single_rollout_with_step_outputs(self):
        aligner = RLDataAligner()
        rollout_id = "rollout1"

        # User's reward function output
        user_eval_result = EvaluateResult(
            score=0.8,
            step_outputs=[
                StepOutput(
                    step_index=0, base_reward=0.25, reason="First action good"
                ),  # Matches assistant_turn_index 0
                StepOutput(
                    step_index="turn_1", base_reward=0.75, reason="Second action better"
                ),  # Matches assistant_turn_index "turn_1"
            ],
        )

        # System's collected StepData
        # RLRolloutWorker should populate step_info with 'assistant_turn_index'
        step_data_list = [
            self.create_mock_step_data(system_step_index=0, assistant_turn_index=0, content="Action 1"),
            self.create_mock_step_data(
                system_step_index=1, assistant_turn_index="intermediate_tool_step"
            ),  # No user reward for this
            self.create_mock_step_data(system_step_index=2, assistant_turn_index="turn_1", content="Action 2"),
            self.create_mock_step_data(
                system_step_index=3, assistant_turn_index=2, content="Action 3"
            ),  # No user reward for this
        ]

        aligned_step_data = aligner.align_data_for_rl_processing(
            current_eval_result=user_eval_result,
            current_step_data_list=step_data_list,
            rollout_id=rollout_id,
        )

        assert len(aligned_step_data) == 4
        # Check base_rewards
        assert aligned_step_data[0].base_reward == 0.25  # Matched step_index 0
        assert aligned_step_data[1].base_reward is None  # No matching step_index "intermediate_tool_step"
        assert aligned_step_data[2].base_reward == 0.75  # Matched step_index "turn_1"
        assert aligned_step_data[3].base_reward is None  # No StepOutput for step_index 2

    def test_align_single_rollout_no_step_outputs(self):
        aligner = RLDataAligner()
        rollout_id = "rollout2"
        user_eval_result = EvaluateResult(score=0.9, reason="Overall score only")
        step_data_list = [self.create_mock_step_data(system_step_index=0, assistant_turn_index=0)]

        aligned_step_data = aligner.align_data_for_rl_processing(
            current_eval_result=user_eval_result,
            current_step_data_list=step_data_list,
            rollout_id=rollout_id,
        )
        assert aligned_step_data[0].base_reward is None

    def test_align_single_rollout_empty_step_outputs(self):
        aligner = RLDataAligner()
        rollout_id = "rollout3"
        user_eval_result = EvaluateResult(score=0.7, step_outputs=[])  # Empty list
        step_data_list = [self.create_mock_step_data(system_step_index=0, assistant_turn_index=0)]

        aligned_step_data = aligner.align_data_for_rl_processing(
            current_eval_result=user_eval_result,
            current_step_data_list=step_data_list,
            rollout_id=rollout_id,
        )
        assert aligned_step_data[0].base_reward is None

    def test_align_step_output_index_not_in_step_data_info(self):
        """Test when a StepOutput.step_index has no corresponding assistant_turn_index in StepData."""
        aligner = RLDataAligner()
        rollout_id = "rollout4"
        user_eval_result = EvaluateResult(
            score=0.5,
            step_outputs=[StepOutput(step_index="non_existent_turn", base_reward=1.0)],
        )
        step_data_list = [self.create_mock_step_data(system_step_index=0, assistant_turn_index=0)]
        aligned_step_data = aligner.align_data_for_rl_processing(
            current_eval_result=user_eval_result,
            current_step_data_list=step_data_list,
            rollout_id=rollout_id,
        )
        assert aligned_step_data[0].base_reward is None  # No match

    def test_align_step_data_missing_assistant_turn_index_in_info(self):
        """Test when StepData.step_info is missing the 'assistant_turn_index' key."""
        aligner = RLDataAligner()
        rollout_id = "rollout5"
        user_eval_result = EvaluateResult(score=0.5, step_outputs=[StepOutput(step_index=0, base_reward=1.0)])
        # Create StepData *without* 'assistant_turn_index' in step_info
        step_data_list = [
            StepData(
                system_step_index=0,
                observation_data=[],
                action_taken={},
                resulting_messages_history=[],
                step_info={},  # No assistant_turn_index
            )
        ]
        aligned_step_data = aligner.align_data_for_rl_processing(
            current_eval_result=user_eval_result,
            current_step_data_list=step_data_list,
            rollout_id=rollout_id,
        )
        assert aligned_step_data[0].base_reward is None  # Cannot map

    def test_align_preserves_other_step_data_fields(self):
        aligner = RLDataAligner()
        rollout_id = "rollout6"
        user_eval_result = EvaluateResult(score=0.8, step_outputs=[StepOutput(step_index=0, base_reward=0.99)])

        original_step_data = self.create_mock_step_data(system_step_index=0, assistant_turn_index=0)
        original_step_data.policy_logprobs = {"logp": -0.1}
        original_step_data.policy_value_estimate = 0.5
        original_step_data.advantage = -0.05  # Should remain untouched by this aligner

        step_data_list = [original_step_data]

        aligned_step_data = aligner.align_data_for_rl_processing(
            current_eval_result=user_eval_result,
            current_step_data_list=step_data_list,
            rollout_id=rollout_id,
        )
        assert aligned_step_data[0].base_reward == 0.99
        assert aligned_step_data[0].policy_logprobs == {"logp": -0.1}
        assert aligned_step_data[0].policy_value_estimate == 0.5
        assert aligned_step_data[0].advantage == -0.05  # Ensure unrelated fields are not wiped
