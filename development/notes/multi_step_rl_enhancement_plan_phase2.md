# Multi-Step RL Enhancement Plan - Phase 2 Details (GiGPO Focused)

This document details **Phase 2: Rollout Generation & GiGPO Advantage Calculation**. Building upon the interfaces and data structures defined in Phase 1, this phase focuses on how the rich `StepData` (the internal representation of RL trajectories) is generated and then processed using GiGPO to calculate advantages for training.

For the overall plan and links to other phases, please see:
*   [Multi-Step RL Enhancement Plan Overview (GiGPO Focused)](./multi_step_rl_enhancement_plan_overview.md)
For Phase 1 details (Reward Function Interface & Preprocessing):
*   [Phase 1 Details (GiGPO Focused)](./multi_step_rl_enhancement_plan_phase1.md)

*(This phase adapts and re-prioritizes tasks from the original plan, focusing on GiGPO and the data flow established in the revised Phase 1.)*

## Phase 2 Tasks: Rollout Generation & GiGPO Advantage Calculation

### **Task 2.1: Implement `RLRolloutWorker` for Trajectory Generation**

*   **Objective:**
    *   Develop the `RLRolloutWorker` class, which is responsible for managing an agent's interaction with an environment over a complete episode.
    *   Ensure `RLRolloutWorker` collects a detailed `List[StepData]` for each episode. Each `StepData` object will capture comprehensive information about a single step (observation, action, policy info, etc.) as defined in Phase 1 (Task 1.1).
    *   The `RLRolloutWorker` will also be responsible for extracting the `messages: List[Message]` and any `ground_truth` from the episode, which are then passed to the user's reward function (via the Sandbox Execution Component as per Phase 1, Task 1.3).

*   **Why it's important:**
    *   This is the core engine for generating the experience data that RL algorithms learn from.
    *   Accurate and complete `StepData` collection is critical for effective GiGPO calculation and subsequent policy training.
    *   It decouples the environment interaction logic from the reward calculation and policy update logic.

*   **How to approach (Detailed Steps for an Engineer):**
    *(This task draws heavily from the original plan's Task 1.1, 1.2, and 1.5 in `multi_step_rl_enhancement_plan_phase1.md` (the older version), but now focuses on producing `StepData` and `List[Message]` for the new GiGPO-centric flow).*

    1.  **Design the `RLRolloutWorker` Class:**
        *   *Action Item:* Create `eval_protocol/agent/rl_rollout_worker.py`.
        *   **Constructor (`__init__`):**
            *   Takes `task_definition: TaskDefinitionModel` (for initial setup, environment details, tool paths, ground truth).
            *   Takes an `agent_policy: AgentPolicy` (interface defined in Task 2.2 of this phase).
            *   Takes configurations like `max_episode_steps`.
            *   Initializes internal state for an episode (e.g., current message history, step counter).
            *   Needs access to an "environment" abstraction. This could be simple (e.g., managing tool calls based on `task_definition`) or more complex if integrating with external environments.
        *   **Core Method (`async run_episode(self) -> Tuple[List[StepData], List[Message], Any]`):**
            *   Manages the main loop of an episode.
            *   Returns the collected `List[StepData]`, the `List[Message]` for the user's reward function, and the `ground_truth`.

    2.  **Implement the Main Rollout Loop within `run_episode`:**
        *   *Action Item:* Develop the step-by-step interaction logic.
            ```python
            # Inside RLRolloutWorker.run_episode()

            # A. Initialize Episode:
            #    self.current_messages_history: List[Message] = [] # Start with initial messages from task_definition
            #    self.collected_step_data: List[StepData] = []
            #    system_step_counter = 0
            #    done = False

            # for _ in range(self.max_episode_steps):
                # B. Prepare Observation for Policy:
                #    current_observation_for_policy = self._prepare_policy_observation(self.current_messages_history)
                #    # This might be the raw messages or a more structured version.

                # C. Get Action from Policy (using AgentPolicy interface - Task 2.2):
                #    policy_action_structured, raw_policy_output, log_probs, value_estimate = \
                #        await self.agent_policy.get_action(current_observation_for_policy)
                #    # policy_action_structured is e.g., {"type": "text", "content": "..."} or {"type": "tool_call", ...}

                # D. Execute Action in "Environment" & Get Outcome:
                #    # This involves interpreting policy_action_structured, calling tools if needed,
                #    # and determining the immediate results (e.g., tool responses, errors).
                #    action_execution_info, new_assistant_message_content, tool_response_messages = \
                #        await self._execute_action_in_environment(policy_action_structured)
                #    # new_assistant_message_content is the text part of the agent's action.
                #    # tool_response_messages are any messages from tools.

                # E. Update Message History:
                #    # Add assistant's message (derived from policy_action_structured or raw_policy_output)
                #    self.current_messages_history.append(Message(role="assistant", content=new_assistant_message_content, ...))
                #    # Add any tool call messages and tool response messages
                #    self.current_messages_history.extend(tool_response_messages) # If tools were used

                # F. Check for Episode Termination (Task 2.3):
                #    done = self._check_termination(self.current_messages_history, action_execution_info)

                # G. Collect `StepData` for this step:
                #    step_data_entry = StepData(
                #        system_step_index=system_step_counter,
                #        observation_data=current_observation_for_policy, # Or the messages before this action
                #        action_taken=policy_action_structured,
                #        raw_policy_output=raw_policy_output,
                #        resulting_messages_history=list(self.current_messages_history), # Copy
                #        policy_logprobs=log_probs,
                #        policy_value_estimate=value_estimate, # Important for GAE, optional for critic-free GiGPO
                #        is_done=done,
                #        step_info=action_execution_info # e.g., tool success/failure
                #    )
                #    self.collected_step_data.append(step_data_entry)

                # H. Increment step counter:
                #    system_step_counter += 1
                #    if done:
                #        break
            # I. Return collected data:
            # return self.collected_step_data, self.current_messages_history, self.task_definition.ground_truth
            ```

    3.  **Implement Helper Methods:**
        *   `_prepare_policy_observation()`: Converts current message history to the format expected by `AgentPolicy`.
        *   `_execute_action_in_environment()`: Handles logic for text generation actions vs. tool call actions. If a tool is called, it invokes the appropriate tool adapter (similar to current `Orchestrator` logic but simplified). It should return structured information about the execution.
        *   `_check_termination()`: Implements termination conditions (see Task 2.3).

*   **Files Involved:**
    *   `eval_protocol/agent/rl_rollout_worker.py` (New)
    *   Uses `StepData` from `eval_protocol/agent/models.py` (or `rl_models.py`) and `Message` from `eval_protocol/models.py`.

*   **Key Learning for Engineer:** Python class design, managing state within an episode, asynchronous programming (`async/await`) for policy calls and tool executions, detailed data logging.

---

### **Task 2.2: Define `AgentPolicy` Interface**

*   **Objective:** Create a well-defined `AgentPolicy` interface (Abstract Base Class or Protocol) that `RLRolloutWorker` uses to interact with the agent model. This decouples the rollout logic from specific model implementations (e.g., OpenAI API, local Hugging Face model).

*   **Why it's important:**
    *   **Modularity:** Allows different agent models to be plugged into the RL system.
    *   **Testability:** Enables mock policies for testing `RLRolloutWorker`.

*   **How to approach (Detailed Steps for an Engineer):**
    *(This task is based on the original plan's Task 1.4 in `multi_step_rl_enhancement_plan_phase1.md` (older version)).*

    1.  **Define the Interface:**
        *   *Action Item:* Create `eval_protocol/agent/policy_interface.py`.
        *   **Core Method: `async get_action(self, observation: Any, available_tools_specs: Optional[List[Dict[str,Any]]] = None) -> Tuple[Dict[str, Any], Optional[str], Optional[Dict[str, Any]], Optional[float]]`**
            *   `observation`: Data provided by `RLRolloutWorker` (e.g., `List[Message]`).
            *   `available_tools_specs`: OpenAPI-like schema for tools the agent can use this turn.
            *   **Returns a tuple:**
                1.  `action_structured` (Dict): Parsed action (e.g., `{"type": "text", "content": "..."}` or `{"type": "tool_call", "name": "tool_name", "arguments": {...}}`).
                2.  `raw_policy_output` (Optional[str]): The raw string from the LLM, if applicable.
                3.  `log_probs` (Optional[Dict]): Log probabilities of the chosen action (needed for PPO).
                4.  `value_estimate` (Optional[float]): `V(s_t)` from a critic, if the policy has one (optional for critic-free GiGPO, but good to include for future flexibility).

    2.  **Implement an Initial Policy (e.g., `OpenAIAgentPolicy`):**
        *   *Action Item:* Create `eval_protocol/agent/policies/openai_policy.py`.
        *   Implements `AgentPolicy`.
        *   Handles formatting requests to OpenAI, parsing responses into the structured action, and extracting log_probs (if `logprobs=True` requested and available). Value estimates are typically not available from standard OpenAI chat completion endpoints unless using a custom model or specific APIs.

*   **Files Involved:**
    *   `eval_protocol/agent/policy_interface.py` (New)
    *   `eval_protocol/agent/policies/openai_policy.py` (New)

*   **Key Learning for Engineer:** API abstraction, interface design (ABCs/Protocols), working with external model APIs.

---

### **Task 2.3: Implement Episode Termination Logic**

*   **Objective:** Integrate flexible episode termination conditions into `RLRolloutWorker`.

*   **Why it's important:**
    *   Prevents infinite loops.
    *   Allows agents to decide when a task is complete.
    *   Enables environment-driven or rule-based termination.

*   **How to approach (Detailed Steps for an Engineer):**
    *(This task is based on the original plan's Task 1.1 options in `multi_step_rl_enhancement_plan_phase1.md` (older version)).*

    1.  **Integrate into `RLRolloutWorker._check_termination()`:**
        *   This method is called at the end of each step in the `run_episode` loop.
        *   It checks multiple conditions:
            *   **Max Steps:** If `current_step_num >= self.max_episode_steps`.
            *   **Agent Signal (Special Token/Action):** If `policy_action_structured` from `AgentPolicy` indicates a "terminate" action (e.g., `{"type": "terminate"}`).
            *   **Agent Signal (Regex on Output):** If the `raw_policy_output` (or generated text content) matches a user-configurable regex pattern provided in `TaskDefinitionModel`.
            *   **Error Condition:** If `_execute_action_in_environment` reported a critical error.
        *   *Action Item:* Implement this logic in `RLRolloutWorker`. `TaskDefinitionModel` will need a field for the optional termination regex.

*   **Files Involved:**
    *   `eval_protocol/agent/rl_rollout_worker.py`
    *   `eval_protocol/models.py` (to add regex field to `TaskDefinitionModel`).

*   **Key Learning for Engineer:** Implementing control flow logic, string manipulation (regex), configuring behavior through data models.

---

### **Task 2.4: Implement System-Level GiGPO Advantage Calculation**

*   **Objective:** Create the system component and helper functions (in `eval_protocol.rl_helpers`) that take the processed `StepData` (which includes `base_reward` and/or `final_score_for_rollout` aligned from Phase 1's output) and compute final GiGPO advantages.

*   **Why it's important:**
    *   This is the core of the GiGPO algorithm implementation.
    *   It transforms the user-defined reward signals into effective learning signals (advantages) for the PPO trainer.

*   **How to approach (Detailed Steps for an Engineer):**
    *(This task implements the GiGPO logic detailed in `development/notes/gigpo_breakdown.md` and uses the outputs from Phase 1, Task 1.4).*

    1.  **Create `eval_protocol/rl_helpers.py`:**
        *   This module will house the GiGPO calculation logic.
        *   *Action Item:* Create the file.

    2.  **Implement `system_apply_gigpo_to_batch_steps()`:**
        *   **Function Signature (Conceptual):**
            ```python
            # In eval_protocol/rl_helpers.py
            # from eval_protocol.agent.models import StepData # Internal StepData
            # from typing import List, Dict, Any, Callable

            # def system_apply_gigpo_to_batch_steps(
            #     batch_of_step_data_lists: List[List[StepData]], # Each StepData should have base_reward (optional)
            #                                                     # and an associated final_score_for_rollout
            #     final_scores_for_rollouts: List[float], # Aligned with batch_of_step_data_lists
            #     omega: float, # GiGPO hyperparameter for balancing A_E and A_S
            #     gamma: float, # Discount factor for future rewards in A_S
            #     # Function to hash observation_data from StepData for anchor states
            #     state_hashing_function: Callable[[Any], Any],
            #     # Optional: default step reward for calculating G_k if base_rewards are sparse/missing
            #     default_step_reward: float = 0.0
            # ) -> List[List[StepData]]: # Returns StepData lists with 'advantage' populated
            ```
        *   **Logic:**
            a.  **Group Trajectories:** Ensure `batch_of_step_data_lists` are appropriately grouped if GiGPO requires rollouts from identical initial states (this grouping strategy needs to be handled by the calling infrastructure, potentially by `RLRolloutWorker` running multiple instances from the same seed/start).
            b.  **Calculate Episode-Level Advantage (`A_E`):**
                *   Use `final_scores_for_rollouts`.
                *   For each group, normalize these scores (e.g., `(score - mean_score) / (std_score + epsilon)`). This gives `A_E` for each trajectory.
                *   *(Reference: `development/notes/gigpo_breakdown.md#episode-level-relative-advantage-macro-advantage`)*
            c.  **Calculate Step-Level Advantage (`A_S`):**
                *   For each group:
                    i.  Iterate through all `StepData` in all trajectories in the group.
                    ii. Use `state_hashing_function` on `StepData.observation_data` to identify anchor states.
                    iii.For each anchor state, collect all actions taken (`StepData.action_taken`) and their discounted future returns (`G_k`).
                        *   `G_k` is calculated by summing `StepData.base_reward` from the current step to the end of the episode, discounted by `gamma`. If `StepData.base_reward` is often `None` or sparse, use `default_step_reward` (e.g., a small negative value for time penalty) plus the `final_score_for_rollout` at the terminal step.
                    iv. For each anchor state, normalize the `G_k` values to get `A_S` for each action taken from that state.
                *   *(Reference: `development/notes/gigpo_breakdown.md#step-level-relative-advantage-micro-advantage`)*
            d.  **Combine Advantages (`A_GiG`):**
                *   For each `StepData` object: `advantage = omega * A_E_of_its_trajectory + (1 - omega) * A_S_of_its_state_action`.
                *   Populate `StepData.advantage` with `A_GiG`.
                *   Optionally, populate `StepData.return_to_go` (e.g., `A_GiG + StepData.policy_value_estimate` if a critic is used, or just `G_k` if critic-free).
                *   *(Reference: `development/notes/gigpo_breakdown.md#combining-macro-and-micro-advantages`)*
        *   *Action Item:* Implement this function and its sub-components (for `A_E`, `A_S`).

    3.  **State Hashing Function:**
        *   The `state_hashing_function` is critical. It needs to take `StepData.observation_data` (which might be `List[Message]`) and produce a hashable, comparable representation. This might involve:
            *   Concatenating message content.
            *   Hashing the last N messages.
            *   Considering other parts of `observation_data` if it's more structured.
        *   This function will likely be passed in by the system configuration.

*   **Files Involved:**
    *   `eval_protocol/rl_helpers.py` (New, for GiGPO logic).
    *   The system component that calls this helper (defined in Phase 1, Task 1.4, e.g., in `eval_protocol/rl_processing.py`).

*   **Key Learning for Engineer:** Implementing complex RL algorithms from papers/specifications, advanced data manipulation (grouping, aggregation, normalization), designing flexible helper functions, understanding the nuances of state representation for LLM agents.

This completes the detailed plan for Phase 2, focusing on generating the necessary data and then applying GiGPO to it.
