from typing import Any, Dict, List, Optional, Union

from eval_protocol.agent.models import StepData  # Internal StepData model

# Assuming models are structured as planned
from eval_protocol.models import EvaluateResult  # Extended EvaluateResult

# Placeholder for actual Message type if needed for type hinting complex observation_data
# from eval_protocol.models import Message


class RLDataAligner:
    """
    Component responsible for aligning outputs from user reward functions
    (EvaluateResult containing scores and/or base_rewards per step)
    with the system's internal StepData representation. This prepares
    the data for subsequent GiGPO (or other RL algorithm) advantage calculations.
    """

    def align_data_for_rl_processing(
        self,
        current_eval_result: EvaluateResult,
        current_step_data_list: List[StepData],
        rollout_id: str,  # For logging or if needed
    ) -> List[StepData]:
        """
        Aligns the EvaluateResult (from user's reward function) with the
        internal list of StepData for a single rollout.

        Populates `StepData.base_reward` from `EvaluateResult.step_outputs.base_reward`.
        Associates `EvaluateResult.score` with the rollout for GiGPO A_E calculation.
        (Association of final_score might happen by returning it alongside, or
         by the caller managing it). For now, this function focuses on base_reward.

        Args:
            current_eval_result: The EvaluateResult from the user's reward function for this rollout.
            current_step_data_list: The list of StepData objects collected by RLRolloutWorker.
            rollout_id: Identifier for the current rollout.

        Returns:
            The list of StepData objects, with `base_reward` populated.
            The `final_score` from current_eval_result should be handled by the caller
            for GiGPO A_E calculation.
        """

        # Store final_score (for GiGPO A_E) - The caller will handle this.
        # This function's primary job is to populate base_rewards in StepData.

        if current_eval_result.step_outputs:
            # Create a dictionary for quick lookup of user-defined step rewards
            user_step_rewards_map: Dict[Union[int, str], float] = {
                step_out.step_index: step_out.base_reward for step_out in current_eval_result.step_outputs
            }

            for s_data in current_step_data_list:
                # --- Critical Mapping Logic ---
                # Strategy: Use 'assistant_turn_index' stored in StepData.step_info
                # by RLRolloutWorker. User's StepOutput.step_index should match this.
                # This assumes RLRolloutWorker adds this info.
                user_defined_step_idx = s_data.step_info.get("assistant_turn_index")

                if user_defined_step_idx is not None:
                    if user_defined_step_idx in user_step_rewards_map:
                        s_data.base_reward = user_step_rewards_map[user_defined_step_idx]
                    else:
                        # No base reward provided by user for this specific system step.
                        # s_data.base_reward remains None (or could be a default).
                        pass
                else:
                    # RLRolloutWorker did not provide 'assistant_turn_index' for this StepData,
                    # or the mapping key in step_info is different.
                    # This indicates a potential issue in RLRolloutWorker or mapping strategy.
                    pass
        else:
            # No step_outputs provided by the user. Base rewards will remain None.
            pass

        return current_step_data_list

    # TODO (Future): Consider a batch version if performance becomes an issue,
    # but the core logic per rollout remains the same.
    # def align_batch_data_for_rl_processing(...)
