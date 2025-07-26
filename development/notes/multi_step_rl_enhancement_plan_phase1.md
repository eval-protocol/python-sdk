# Multi-Step RL Enhancement Plan - Phase 1 Details (GiGPO Focused)

This document provides a detailed breakdown of the tasks involved in **Phase 1: Unified Reward Function Interface & System-Level GiGPO Setup**. This phase prioritizes the reward authoring experience, assuming rollouts are available, and then details how the system processes these rewards for GiGPO.

For the overall plan, introduction, core concepts, and links to other phases, please see:
*   [Multi-Step RL Enhancement Plan Overview (GiGPO Focused)](./multi_step_rl_enhancement_plan_overview.md)

*(Original Phase 1 content related to GAE, direct per-step rewards by RLRolloutWorker, and some aspects of Orchestrator/RLRolloutWorker's initial design are superseded by this GiGPO-focused plan).*

## Phase 1 Tasks: Unified Reward Function Interface & System-Level GiGPO Setup

### **Task 1.1: Define Core Data Structures for Reward Authoring and RL**

*   **Objective:**
    *   Solidify the `Message` model as the primary input for user reward functions.
    *   Extend the existing `EvaluateResult` Pydantic model to optionally include per-step base reward information.
    *   Define a new `StepOutput` Pydantic model to structure this per-step information within `EvaluateResult`.
    *   Define the internal `StepData` Pydantic model that the system uses to hold comprehensive RL information for each step of a rollout.

*   **Why it's important:**
    *   Clear data structures are fundamental for a clean API and robust system.
    *   `Message` provides a familiar interface for reward authors.
    *   Extending `EvaluateResult` maintains backward compatibility while adding RL capabilities.
    *   `StepOutput` standardizes how per-step base rewards are conveyed from the user's function.
    *   `StepData` is crucial for the system to perform GiGPO calculations, as it holds more context than just messages (e.g., `policy_logprobs`, `action_taken` details, etc., which are needed for PPO updates and potentially for GiGPO's state hashing or outcome analysis).

*   **How to approach (Detailed Steps for an Engineer):**

    1.  **Review `Message` Model:**
        *   Ensure `eval_protocol.models.Message` is suitable. It typically includes `role` and `content`, and may have `tool_calls`, `tool_call_id`. This should be sufficient for user reward functions.
        *   *Action Item:* No changes likely needed to `Message` itself for this task.

    2.  **Define `StepOutput` Model:**
        *   This model will be part of the `EvaluateResult`.
        *   *Action Item:* Create/define this Pydantic model, likely in `eval_protocol/typed_interface.py` or a new `eval_protocol/rl_models.py`.
            ```python
            # In eval_protocol/typed_interface.py or eval_protocol/rl_models.py
            from pydantic import BaseModel
            from typing import Dict, Any, Optional, Union

            class StepOutput(BaseModel):
                # User-defined index. This must be interpretable by the user's reward function
                # and allow the system to later map this StepOutput back to a specific
                # point in the full StepData list for the rollout.
                # e.g., index of the assistant message, a turn number, or a unique step ID.
                step_index: Union[int, str]
                base_reward: float  # Base reward calculated by the user's function for this step
                metrics: Dict[str, Any] = {}
                reason: Optional[str] = None
            ```

    3.  **Extend `EvaluateResult` Model:**
        *   Modify `eval_protocol.typed_interface.EvaluateResult`.
        *   *Action Item:* Add the `step_outputs` field.
            ```python
            # In eval_protocol/typed_interface.py
            # from .models import Message # If Message is used directly, else not needed here
            from pydantic import BaseModel # Assuming OriginalEvaluateResult is a BaseModel
            from typing import List, Optional, Dict, Any # Add List, Optional

            # Assume StepOutput is defined in this file or imported
            # from .rl_models import StepOutput # If in a separate file

            class EvaluateResult(BaseModel): # Or if inheriting: class EvaluateResult(OriginalEvaluateResult):
                score: float # Existing field
                is_score_valid: bool = True # Existing field
                reason: Optional[str] = None # Existing field

                # New field for RL per-step base rewards
                step_outputs: Optional[List[StepOutput]] = None

                # Optional: Consider a generic metrics dictionary if not already present
                # custom_metrics: Dict[str, Any] = {}
            ```
        *   Ensure backward compatibility: old code not providing `step_outputs` should still work.

    4.  **Define Internal `StepData` Model:**
        *   This is a system-internal structure, not directly manipulated by the user's message-based reward function. It's collected by `RLRolloutWorker` (Phase 2) and used by the system for GiGPO.
        *   *Action Item:* Define this Pydantic model, likely in `eval_protocol/agent/models.py` or `eval_protocol/rl_models.py`.
            ```python
            # In eval_protocol/agent/models.py or eval_protocol/rl_models.py
            from pydantic import BaseModel
            from typing import List, Dict, Any, Optional

            # Assuming Message model is importable, e.g., from eval_protocol.models import Message

            class StepData(BaseModel):
                # System-generated index for the step within the episode/rollout.
                system_step_index: int

                # Observation provided to the policy for this step.
                # This could be the List[Message] up to this point, or a more processed observation.
                observation_data: Any # For flexibility; often List[Message]

                # Structured representation of the action chosen by the policy.
                # e.g., {"type": "text", "content": "..."} or {"type": "tool_call", "name": "...", "args": ...}
                action_taken: Dict[str, Any]

                # Raw output from the policy model, if different from action_taken (e.g., full completion string)
                raw_policy_output: Optional[str] = None

                # List of all messages after this step's action was taken and any tool responses processed.
                # This forms the basis for the next step's observation_data.
                resulting_messages_history: List[Dict[str, Any]] # List[Message] effectively

                # RL-specific data from the policy
                policy_logprobs: Optional[Dict[str, Any]] = None # Log probability of action_taken
                # Value estimate V(s_t) from policy's critic. GiGPO can be critic-free,
                # but collecting it allows for future flexibility or comparison.
                policy_value_estimate: Optional[float] = None

                is_done: bool = False # Did the episode terminate after this step?

                # Info dictionary for diagnostics, tool call success/failure, errors, etc.
                step_info: Dict[str, Any] = {}

                # --- Fields to be populated by the system during/after reward processing ---
                # Base reward for this step, aligned from user's EvaluateResult.step_outputs
                base_reward: Optional[float] = None
                # Final advantage for this step (e.g., from GiGPO)
                advantage: Optional[float] = None
                # Target for value function update (e.g., G_t or GAE return)
                return_to_go: Optional[float] = None
            ```

*   **Files Involved:**
    *   `eval_protocol/typed_interface.py` (for `EvaluateResult`, `StepOutput`)
    *   `eval_protocol/agent/models.py` or `eval_protocol/rl_models.py` (for `StepData`, and potentially `StepOutput` if moved)
    *   `eval_protocol/models.py` (for `Message`)

*   **Key Learning for Engineer:** Pydantic model definition, understanding data flow implications, designing for extensibility and backward compatibility.

---

### **Task 1.2: Implement Unified User Reward Function Definition**

*   **Objective:**
    *   Enable users to define reward functions using a single `@reward_function` decorator.
    *   Support two primary modes for these functions based on their signature:
        1.  **Pointwise:** Takes `messages: List[Message]` for a single rollout.
        2.  **Batch-wise:** Takes `rollouts_messages: List[List[Message]]` for multiple rollouts.
    *   The decorated functions will return the (extended) `EvaluateResult` (or `List[EvaluateResult]` for batch mode).

*   **Why it's important:**
    *   Provides a consistent and simple entry point for users to define how rollouts are scored or how base rewards for RL are determined.
    *   Focuses the user on interpreting `List[Message]` content, abstracting away more complex RL data structures at this stage.

*   **How to approach (Detailed Steps for an Engineer):**

    1.  **Adapt `@reward_function` Decorator (in `eval_protocol/typed_interface.py`):**
        *   The decorator itself might not need much change. Its main role is to mark a function for discovery by the sandbox execution component.
        *   The "mode" (pointwise vs. batch) will likely be determined by the sandbox execution component when it prepares to call the user's function, possibly based on configuration passed to it or by inspecting the function signature if feasible (though explicit configuration is more robust).
        *   *Action Item:* Review `@reward_function`. Ensure it doesn't impose constraints that prevent these two modes.

    2.  **Define Expected User Function Signatures (for documentation and examples):**
        *   **Pointwise Mode:**
            *   `def my_pointwise_eval_or_reward_func(messages: List[Message], ground_truth: Any, **kwargs) -> EvaluateResult:`
            *   `**kwargs` can receive additional parameters specified in the task definition's evaluation criteria.
            *   The function is responsible for populating the `EvaluateResult` object, including `score` and/or `step_outputs: List[StepOutput]`.
            *   The `StepOutput.step_index` should be defined by the user to map to a conceptual step in the `messages` list (e.g., index of an 'assistant' message, or a turn counter).
        *   **Batch-wise Mode:**
            *   `def my_batch_eval_or_reward_func(rollouts_messages: List[List[Message]], ground_truths: List[Any], **kwargs) -> List[EvaluateResult]:`
            *   `rollouts_messages`: A list of message lists (one per rollout).
            *   `ground_truths`: A list of corresponding ground truths, aligned by index.
            *   `**kwargs`: Batch-level parameters (e.g., for an LLM judge configuration).
            *   The function returns a `List[EvaluateResult]`, one for each input rollout, aligned by index.

    3.  **Guidance on `StepOutput.step_index`:**
        *   Users need clear documentation on how to set `step_index` in their `StepOutput` objects. This index is critical for the system to later align the `base_reward` with the correct internal `StepData` object from the full rollout.
        *   Example strategies for `step_index`:
            *   Index of the 'assistant' message that represents the agent's action for that step.
            *   A sequential turn number if the interaction is strictly turn-based.
        *   The system, when processing `StepData` (collected by `RLRolloutWorker`), will also need to record a corresponding `system_step_index` or a way to map to the user's `step_index`.

*   **Files Involved:**
    *   `eval_protocol/typed_interface.py` (for `@reward_function` and `EvaluateResult`).
    *   Documentation and example files.

*   **Key Learning for Engineer:** API design for user-facing functions, importance of clear contracts (signatures, return types), and how data flows from user code back to the system.

---

### **Task 1.3: Adapt Sandbox Execution Component for Reward Functions**

*   **Objective:** Modify the sandbox execution component (currently in `eval_protocol/evaluation.py`, e.g., `EvalSandbox` and its related classes like `MultiMetricsSandbox`) to correctly invoke the user's pointwise or batch-wise reward functions and handle their `EvaluateResult` outputs.

*   **Why it's important:**
    *   This component is responsible for safely running user-provided code.
    *   It needs to adapt to the two modes of reward functions and correctly pass data in and retrieve results.

*   **How to approach (Detailed Steps for an Engineer):**

    1.  **Review Existing Sandbox Logic (`eval_protocol/evaluation.py`):**
        *   Understand how `EvalSandbox`, `MultiMetricsSandbox`, `MultiCriteriaSandbox` currently work, especially methods like `setup()`, `run_code()`, `evaluate()`, and the internal `evaluate_with_line()` wrapper.
        *   *(Reference: The `eval_protocol/evaluation.py` code provided in the initial prompt.)*

    2.  **Generalize the Sandbox's Main Execution Method:**
        *   The method that orchestrates running user code (e.g., `EvalSandbox.evaluate()`) needs to:
            *   Accept a configuration indicating whether to run in "pointwise" or "batch-wise" mode. This might be a new parameter or inferred from the structure of the input data.
            *   Accept the actual data to be processed:
                *   Pointwise: `(messages: List[Message], ground_truth: Any, user_kwargs: Dict)`.
                *   Batch-wise: `(rollouts_messages: List[List[Message]], ground_truths: List[Any], user_kwargs: Dict)`.
        *   *Action Item:* Refactor or add new methods to `EvalSandbox` (or a relevant class) to handle these distinct invocation patterns.

    3.  **Modify/Create Sandbox Wrapper Code:**
        *   The Python code that `EvalSandbox` writes into the sandbox to call the user's function (e.g., the current `evaluate_with_line` logic in `MultiMetricsSandbox.wrap_evaluate` or `MultiCriteriaSandbox.wrap_evaluate`) needs to be adapted:
            *   **Pointwise Wrapper:**
                *   Receives serialized `(messages, ground_truth, user_kwargs)`.
                *   Deserializes them.
                *   Locates and calls the user's decorated pointwise function: `user_module.user_eval_func(messages, ground_truth, **user_kwargs)`.
                *   Serializes the returned `EvaluateResult` object.
            *   **Batch-wise Wrapper:**
                *   Receives serialized `(rollouts_messages, ground_truths, user_kwargs)`.
                *   Deserializes them.
                *   Locates and calls the user's decorated batch-wise function: `user_module.user_batch_eval_func(rollouts_messages, ground_truths, **user_kwargs)`.
                *   Serializes the returned `List[EvaluateResult]`.
        *   The sandbox component needs to select or generate the correct wrapper based on the mode.
        *   *Action Item:* Design these two types of wrapper scripts/logic.

    4.  **Data Serialization/Deserialization:**
        *   Ensure robust serialization (e.g., JSON) of `List[Message]`, `List[List[Message]]`, `EvaluateResult`, and `List[EvaluateResult]` for communication with the sandbox. Pydantic's `.model_dump_json()` and `model_validate_json()` are useful here.

    5.  **Return Value Handling:**
        *   The calling code (outside the sandbox) needs to correctly interpret the returned data based on the mode (a single `EvaluateResult` or a `List[EvaluateResult]`).

*   **Files Involved:**
    *   `eval_protocol/evaluation.py` (significant changes to `EvalSandbox` and/or related classes).
    *   Potentially new files for sandbox wrapper templates if they become complex.

*   **Key Learning for Engineer:** Modifying existing complex classes, inter-process communication (serialization), dynamic code generation/selection for sandbox execution, API design for internal system components.

---

### **Task 1.4: Design System-Level Preprocessing for GiGPO**

*   **Objective:** Define the system component and logic, referred to as the "Aligner," that runs *after* the user's reward function (executed in the sandbox) and prepares the data for the main GiGPO advantage calculation. This involves aligning the user-provided base rewards (from `EvaluateResult.step_outputs`) or final scores (from `EvaluateResult.score`) with the system's internal, richer `StepData`.

*   **Why it's important:**
    *   The user's reward function, working with `List[Message]`, provides high-level reward signals (`base_reward` per step or a `final_score` for the rollout).
    *   The GiGPO algorithm (and the PPO updates it informs) needs these signals precisely mapped to each step in the detailed `StepData` (which includes `policy_logprobs`, `action_taken`, `observation_data`, etc., collected by `RLRolloutWorker`).
    *   This Aligner component bridges the gap between the user's simplified view (message-based rewards) and the RL algorithm's detailed data requirements.

*   **How to approach (Detailed Steps for an Engineer):**

    1.  **Define the "Aligner" Component/Logic (e.g., `RLDataAligner` class or module):**
        *   **Location:** Likely a new module, e.g., `eval_protocol/rl_processing.py`.
        *   **Primary Method (Conceptual):** `align_data_for_rl_processing`
        *   **Inputs to this method:**
            *   `eval_results_from_sandbox: Union[EvaluateResult, List[EvaluateResult]]`: The direct output from the user's reward function after sandbox execution.
            *   `raw_rollout_step_data_map: Dict[str, List[StepData]]`: A dictionary mapping a unique `rollout_id` to the `List[StepData]` that was collected by `RLRolloutWorker` for that rollout. This `StepData` contains `system_step_index`, `policy_logprobs`, etc., but `base_reward` and `advantage` are not yet populated.
            *   `rollout_ids_in_order: List[str]`: If `eval_results_from_sandbox` is a list (from batch mode), this provides the order and mapping to `raw_rollout_step_data_map`. If pointwise, this would be a single `rollout_id`.

    2.  **Core Alignment Logic within `align_data_for_rl_processing`:**
        *   The method will iterate through each rollout identified by `rollout_ids_in_order`.
        *   For each `rollout_id`:
            a.  Retrieve the corresponding `current_eval_result: EvaluateResult` (from `eval_results_from_sandbox`).
            b.  Retrieve the `current_step_data_list: List[StepData]` (from `raw_rollout_step_data_map`).
            c.  **Store `final_score` (for GiGPO `A_E`):**
                *   If `current_eval_result.score` is not `None`, this score needs to be associated with the `current_step_data_list`. This can be achieved by adding a field to a wrapper object containing the `current_step_data_list` or by ensuring the `system_apply_gigpo_to_batch_steps` function (Task 2.4) can receive these scores alongside the `StepData`. For simplicity, let's assume the `system_apply_gigpo_to_batch_steps` function will take `final_scores_for_rollouts` as a separate argument, aligned with the batch of `StepData` lists. The Aligner's role here is to ensure these scores are correctly extracted and available.
            d.  **Align `step_outputs` with `StepData` (to populate `StepData.base_reward`):**
                *   If `current_eval_result.step_outputs` is not `None` and is a list of `StepOutput`:
                    i.  **Mapping Strategy:** A robust strategy is needed to map `StepOutput.step_index` (user-defined) to `StepData.system_step_index` (system-defined).
                        *   **Recommendation:** The `RLRolloutWorker` (Phase 2, Task 2.1), when generating `StepData`, should also generate a simple, sequential `turn_index` or `assistant_action_index` for each conceptual "step" where the agent acts (e.g., produces an assistant message). This `assistant_action_index` can be stored in `StepData.step_info` (e.g., `step_info={'assistant_turn_index': k}`).
                        *   The user, when creating `StepOutput` in their reward function, would use this same `assistant_turn_index` as their `step_index`.
                        *   The Aligner can then iterate through `current_eval_result.step_outputs` and, for each `user_step_out` in it, find the `s_data` in `current_step_data_list` where `s_data.step_info.get('assistant_turn_index') == user_step_out.step_index`.
                        *   If a match is found, set `s_data.base_reward = user_step_out.base_reward`.
                    ii. **Error Handling:** Log warnings or errors if a `user_step_out.step_index` cannot be matched to any `StepData`, or if multiple `StepOutput` objects claim the same `step_index`.
                    iii.If no `base_reward` is provided by the user for a system step (no matching `StepOutput`), `s_data.base_reward` remains `None`. The GiGPO calculation (Task 2.4) will handle this, potentially using a `default_step_reward`.

    3.  **Output of the Aligner Component:**
        *   The function returns the `raw_rollout_step_data_map` (or a transformed version like `List[List[StepData]]` if preferred for batch processing by GiGPO). The key is that the `StepData` objects within are now augmented:
            *   `base_reward` field is populated if the user provided per-step rewards.
            *   The `final_score` for each rollout is available for the GiGPO `A_E` calculation.
        *   This output is the direct input to the `system_apply_gigpo_to_batch_steps` function (Phase 2, Task 2.4).

    4.  **Documentation for Reward Authors:**
        *   Clearly document how users should define `StepOutput.step_index` (e.g., "This should correspond to the Nth time the assistant generated a message in the rollout, starting from 0.").
        *   Explain that if `step_outputs` are not provided, the system might rely solely on `EvaluateResult.score` for GiGPO (leading to sparser step-level signals if `default_step_reward` is 0).

*   **Files Involved:**
    *   A new module, e.g., `eval_protocol/rl_processing.py`, to house the `RLDataAligner` logic.
    *   This module will use types from `eval_protocol/typed_interface.py` (for `EvaluateResult`) and `eval_protocol/agent/models.py` (for `StepData`).

*   **Key Learning for Engineer:** Designing data transformation pipelines, handling data alignment between different sources (user output vs. system data), defining clear contracts for indices and identifiers, considering edge cases and error handling in data mapping.

This detailed Phase 1 plan focuses on establishing the reward authoring interface and the initial system steps to prepare data for GiGPO, keeping the user's interaction with `List[Message]` central.
