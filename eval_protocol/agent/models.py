from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from eval_protocol.models import Message


class StepData(BaseModel):
    """
    Internal system structure to hold comprehensive RL information for each step of a rollout.
    This is collected by RLRolloutWorker and used by the system for advantage calculations (e.g., GiGPO).
    It is not directly exposed to the user's primary reward function if that function
    is defined to only operate on List[Message].
    """

    system_step_index: int = Field(description="System-generated index for the step within the episode/rollout.")

    observation_data: Any = Field(
        description="Observation provided to the policy for this step. For flexibility; often List[Message] or a processed version."
    )

    action_taken: Dict[str, Any] = Field(
        description="Structured representation of the action chosen by the policy (e.g., {'type': 'text', 'content': '...'} or {'type': 'tool_call', 'name': '...', 'args': ...})."
    )
    raw_policy_output: Optional[str] = Field(
        default=None,
        description="Raw output from the policy model, if different from action_taken (e.g., full completion string).",
    )

    resulting_messages_history: List[Message] = Field(
        description="List of all messages after this step's action was taken and any tool responses processed. This forms the basis for the next step's observation_data."
    )

    # RL-specific data from the policy
    policy_logprobs: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Log probability of action_taken, as produced by the policy.",
    )
    policy_value_estimate: Optional[float] = Field(
        default=None,
        description="Value estimate V(s_t) from the policy's critic component. Optional, as GiGPO can be critic-free.",
    )

    is_done: bool = Field(
        default=False,
        description="Boolean flag indicating if the episode terminated after this step.",
    )

    step_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary for diagnostic information, tool call success/failure, errors, or other metadata related to this step's execution.",
    )

    # --- Fields to be populated by the system during/after reward processing ---
    base_reward: Optional[float] = Field(
        default=None,
        description="Base reward for this step, aligned from the user's EvaluateResult.step_outputs.",
    )
    advantage: Optional[float] = Field(
        default=None,
        description="Final advantage for this step (e.g., from GiGPO calculation). This is the primary learning signal for policy gradient updates.",
    )
    return_to_go: Optional[float] = Field(
        default=None,
        description="Estimated sum of future discounted rewards from this step. Can be a target for a value function or derived from advantages.",
    )

    class Config:
        extra = "allow"  # Allow extra fields if needed for specific use cases
