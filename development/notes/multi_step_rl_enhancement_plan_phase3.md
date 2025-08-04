# Multi-Step RL Enhancement Plan - Phase 3 Details (GiGPO Focused)

This document outlines **Phase 3: Training Integration, User Experience, and Scalability**. With the reward function interfaces, GiGPO advantage calculation, and rollout generation mechanisms established in Phases 1 and 2, this phase focuses on utilizing these components in an RL training loop, ensuring a good developer experience, and considering future scalability.

For the overall plan and links to other phases, please see:
*   [Multi-Step RL Enhancement Plan Overview (GiGPO Focused)](./multi_step_rl_enhancement_plan_overview.md)
For Phase 1 details (Reward Function Interface & GiGPO Preprocessing):
*   [Phase 1 Details (GiGPO Focused)](./multi_step_rl_enhancement_plan_phase1.md)
For Phase 2 details (Rollout Generation & GiGPO Advantage Calculation):
*   [Phase 2 Details (GiGPO Focused)](./multi_step_rl_enhancement_plan_phase2.md)

*(This phase adapts tasks from the original plan's Phase 2 and 3, focusing on the GiGPO-centric workflow).*

## Phase 3 Tasks: Training Integration, User Experience, and Scalability

### **Task 3.1: Example RL Training Loop with PPO and GiGPO Advantages**

*   **Objective:**
    *   Create a runnable example script demonstrating the end-to-end workflow:
        1.  `RLRolloutWorker` (from Phase 2) generates `List[StepData]` (internal RL data) and `(messages, ground_truth)` (for user reward function).
        2.  The Sandbox Execution Component (adapted in Phase 1) processes `(messages, ground_truth)` using the user's reward function, returning an `EvaluateResult` (containing `score` and/or `step_outputs` with `base_rewards`).
        3.  The System-Level Preprocessing for GiGPO (from Phase 1) aligns `EvaluateResult` data with `List[StepData]`.
        4.  The System-Level GiGPO Advantage Calculation (from Phase 2) computes final GiGPO advantages, augmenting `List[StepData]`.
        5.  The augmented `List[StepData]` (containing `policy_logprobs` and `advantage`) is used to perform a conceptual PPO policy update.
    *   This example serves as a proof-of-concept and a template for users.

*   **Why it's important:**
    *   Validates the entire data pipeline from rollout to training signal.
    *   Provides a concrete example for users integrating `eval-protocol` into their RL training setups.
    *   Helps identify any remaining gaps or friction points in the workflow.

*   **How to approach (Detailed Steps for an Engineer):**
    *(This task is based on the original plan's Task 2.4 in `multi_step_rl_enhancement_plan_phase2.md` (older version), but now specifically uses GiGPO advantages).*

    1.  **Create Example Script (`examples/gippo_ppo_training_example.py`):**
        *   *Action Item:* Develop this new example script.
        *   **Workflow within the script:**
            a.  **Setup:**
                *   Initialize `TaskDefinitionModel`, `AgentPolicy` (e.g., a mock or simple `OpenAIAgentPolicy`), `RLRolloutWorker`.
                *   Define path to user's reward function Python file.
                *   Configure GiGPO parameters (`omega`, `gamma`, `state_hashing_function`, `default_step_reward`).
                *   Initialize the Sandbox Execution Component and the System-Level GiGPO Preprocessor/Aligner.
            b.  **Simulated Training Loop:**
                ```python
                # NUM_TRAINING_ITERATIONS = ...
                # EPISODES_PER_BATCH_GROUP = ... # For GiGPO, rollouts in a group need same start
                # MAX_EPISODE_STEPS = ...

                # for training_iter in range(NUM_TRAINING_ITERATIONS):
                #     batch_group_step_data_raw: List[List[StepData]] = []
                #     batch_group_messages_for_reward: List[List[Message]] = []
                #     batch_group_ground_truths: List[Any] = []

                #     # 1. Collect a batch/group of trajectories
                #     for _ in range(EPISODES_PER_BATCH_GROUP):
                #         # For GiGPO, ensure RLRolloutWorker can start from identical initial conditions if needed
                #         list_step_data, messages_for_reward_fn, ground_truth = \
                #             await rl_rollout_worker.run_episode()
                #         batch_group_step_data_raw.append(list_step_data)
                #         batch_group_messages_for_reward.append(messages_for_reward_fn)
                #         batch_group_ground_truths.append(ground_truth)

                #     # 2. Get base rewards/scores from user's reward function via Sandbox
                #     # Assume sandbox_executor is an instance of the adapted component
                #     # This call would be configured for "batch-wise" execution
                #     list_evaluate_result = await sandbox_executor.execute_user_reward_code(
                #         rollouts_messages=batch_group_messages_for_reward,
                #         ground_truths=batch_group_ground_truths,
                #         # ... other necessary configs for user function ...
                #     ) # Returns List[EvaluateResult]

                #     # 3. Align EvaluateResults with StepData (System-Level Preprocessing for GiGPO)
                #     # This populates StepData.base_reward or attaches final_score_for_rollout
                #     processed_batch_step_data = system_aligner.align_rewards_with_step_data(
                #         eval_results=list_evaluate_result,
                #         rollout_step_data=batch_group_step_data_raw,
                #         mode="batch"
                #     )
                #     final_scores = [er.score for er in list_evaluate_result] # Assuming score is present

                #     # 4. Calculate GiGPO Advantages (System-Level GiGPO Calculation)
                #     # This uses eval_protocol.rl_helpers.system_apply_gigpo_to_batch_steps
                #     batch_step_data_with_advantages = system_gippo_calculator.calculate_advantages(
                #         batch_of_step_data_lists=processed_batch_step_data,
                #         final_scores_for_rollouts=final_scores, # Or uses base_rewards within StepData
                #         omega=0.5, gamma=0.99, state_hashing_function=my_hasher, ...
                #     ) # Returns List[List[StepData]] with 'advantage' field populated

                #     # 5. (Simulated) PPO Policy Update
                #     # Flatten the batch for PPO update
                #     all_steps_for_update: List[StepData] = [step for traj in batch_step_data_with_advantages for step in traj]
                #
                #     # In a real PPO trainer, you'd use StepData.policy_logprobs and StepData.advantage
                #     # agent_policy.update(all_steps_for_update) # Conceptual update
                #     print(f"Iter {training_iter}: Simulated PPO update with {len(all_steps_for_update)} steps using GiGPO advantages.")
                ```
    2.  **Focus on Data Flow:** The example should clearly show the data transformation at each stage, from raw `StepData` and `Message` lists to `EvaluateResult`, then to `StepData` with aligned base rewards/scores, and finally to `StepData` with GiGPO advantages. Actual model weight updates can be simulated or commented out.

*   **Files Involved:**
    *   `examples/gippo_ppo_training_example.py` (New)
    *   Supporting mock/simple implementations for `AgentPolicy`, user reward function.

*   **Key Learning for Engineer:** Understanding the complete data lifecycle in an RL system, integrating multiple components, writing illustrative example code.

---

### **Task 3.2: Documentation and Examples for Reward Authors**

*   **Objective:** Provide comprehensive documentation and clear examples to guide users in writing effective reward functions for both standard evaluation and for providing the necessary signals (base rewards or final scores) for the system's GiGPO calculations.

*   **Why it's important:**
    *   Ease of use is a primary goal. Good documentation is crucial for users to leverage the new capabilities.
    *   Clear examples will significantly lower the barrier to entry.

*   **How to approach (Detailed Steps for an Engineer):**

    1.  **Update `eval_protocol/typed_interface.py` Docstrings:**
        *   Clearly document the extended `EvaluateResult` model, including the new `step_outputs: Optional[List[StepOutput]]` field and its purpose.
        *   Document the `StepOutput` model: `step_index`, `base_reward`, `metrics`, `reason`.
        *   Explain the expected signatures for `@reward_function` in pointwise and batch-wise modes.
        *   *Action Item:* Update these docstrings.

    2.  **Create a New Documentation Page (e.g., `docs/developer_guide/authoring_rl_rewards.mdx`):**
        *   **Explain the Two-Stage Process:**
            *   Stage 1: User's reward function (sandboxed) processes `List[Message]` (or `List[List[Message]]`) and returns `EvaluateResult` containing `score` and/or `step_outputs` (with `base_reward` and `step_index`).
            *   Stage 2: System uses this output, along with internal `StepData`, to calculate GiGPO advantages.
        *   **Guidance on `StepOutput.step_index`:** Provide detailed instructions and examples on how users should determine `step_index` so it can be reliably mapped by the system to `StepData`. (e.g., "use the index of the assistant's message in the input `messages` list that corresponds to this conceptual step").
        *   **Pointwise Reward Function Examples for RL:**
            *   Show how to iterate through `messages` to identify agent steps and assign `base_reward` to each via `StepOutput`.
            *   Example: Rewarding successful tool use based on parsing subsequent tool response messages.
            *   Example: Assigning a `base_reward` based on the content of an assistant's message.
        *   **Batch-wise Reward Function Examples:**
            *   Show how to process `List[List[Message]]`.
            *   Example: An LLM-as-a-judge comparing multiple rollouts and returning a `final_score` in each `EvaluateResult` (this score is then used by system-GiGPO for `A_E`).
            *   Example: A batch function that calculates `base_rewards` for each step in each rollout, potentially using some shared context from the batch.
        *   **How `EvaluateResult.score` vs. `EvaluateResult.step_outputs` are used by GiGPO:**
            *   Explain that `score` is typically used for GiGPO's Episode-Level Advantage (`A_E`).
            *   Explain that `step_outputs` (providing `base_reward`) are used for calculating future discounted returns for GiGPO's Step-Level Advantage (`A_S`). If `base_rewards` are sparse, the system might use `default_step_reward` and the `score` as the terminal reward for `A_S` calculations.
        *   *Action Item:* Write this documentation page with clear code examples.

*   **Files Involved:**
    *   `eval_protocol/typed_interface.py` (for docstrings).
    *   `docs/developer_guide/authoring_rl_rewards.mdx` (New documentation file).
    *   Update other relevant documentation pages to link to this new guide.

*   **Key Learning for Engineer:** Technical writing, creating clear and concise examples, understanding the user's perspective.

---

### **Task 3.3: Enhance Observability and Debugging for RL**

*   **Objective:** Implement logging and simple tools to help developers and users understand agent behavior during rollouts, inspect the data collected (`StepData`), and debug the GiGPO advantage calculation process.

*   **Why it's important:**
    *   RL systems can be complex "black boxes." Good observability is key to diagnosing issues and iterating effectively.

*   **How to approach (Detailed Steps for an Engineer):**
    *(This task is based on the original plan's Task 2.3 in `multi_step_rl_enhancement_plan_phase2.md` (older version)).*

    1.  **Logging in `RLRolloutWorker`:**
        *   Ensure comprehensive DEBUG-level logging for each step: observation, chosen action (structured and raw), `policy_logprobs`, `policy_value_estimate`, tool execution details, resulting messages.
        *   Log episode summaries: total steps, termination reason, any collected `final_score` from user reward function.
        *   *Action Item:* Implement/enhance logging in `RLRolloutWorker`.

    2.  **Logging in System-Level GiGPO Components:**
        *   Log inputs and outputs of the "Aligner" component (Phase 1, Task 1.4).
        *   Log key intermediate calculations within `eval_protocol.rl_helpers.system_apply_gigpo_to_batch_steps()`:
            *   Calculated `A_E` for trajectories.
            *   Identified anchor states and their hashed representations.
            *   Calculated `G_k` (future discounted returns) for actions from anchor states.
            *   Calculated `A_S` values.
            *   Final combined `A_GiG` (advantages) for each step.
        *   This logging should be configurable (e.g., via log levels) to avoid excessive output.
        *   *Action Item:* Add detailed logging to GiGPO helper functions and the system components that call them.

    3.  **Trajectory/`StepData` Saving Utility:**
        *   Provide an option (e.g., in `RLRolloutWorker` or the training script) to save the full `List[StepData]` for selected episodes (e.g., every Nth episode, or episodes meeting certain criteria) to disk (e.g., as JSONL files).
        *   This allows for offline inspection and debugging.
        *   *Action Item:* Implement this utility.

    4.  **Simple `StepData` Viewer/Pretty-Printer Script (Optional but helpful):**
        *   A script (`scripts/view_rl_trajectory.py`) that takes a saved `StepData` JSONL file and prints it in a human-readable format, showing the sequence of observations (key messages), actions, rewards, advantages, etc.
        *   *Action Item:* Develop this basic script.

*   **Files Involved:**
    *   `eval_protocol/agent/rl_rollout_worker.py`
    *   `eval_protocol/rl_processing.py` (or wherever Aligner and GiGPO orchestration happens)
    *   `eval_protocol/rl_helpers.py`
    *   `scripts/view_rl_trajectory.py` (New)

*   **Key Learning for Engineer:** Effective use of Python's `logging` module, designing for debuggability, creating helpful developer tools.

---

### **Task 3.4: (Future Consideration) Parallel Rollout Execution & Scalability**

*   **Objective:** Explore and outline strategies for parallelizing `RLRolloutWorker` execution to speed up data collection, especially for GiGPO which benefits from larger batches of trajectories (and groups from identical starts).

*   **Why it's important:**
    *   Data collection is often the bottleneck in RL.
    *   Scalability is crucial for tackling more complex problems and training larger models.

*   **How to approach (High-Level Exploration for now):**
    *(This task is based on the original plan's Task 3.1 in `multi_step_rl_enhancement_plan_phase3.md` (older version)).*

    1.  **Identify Parallelization Points:**
        *   Running multiple `RLRolloutWorker` instances concurrently, each handling a separate episode.
        *   If `AgentPolicy.get_action()` involves remote calls (e.g., to OpenAI), these can be made asynchronously across multiple workers.
        *   Tool executions within `RLRolloutWorker` can also be asynchronous.

    2.  **Consider Frameworks/Libraries:**
        *   Python's `asyncio` for managing concurrent tasks within a single process.
        *   Libraries like Ray (as used by VeRL, SkyRL - see `agent_rl_survey.md`) for distributed rollouts across multiple processes or machines. This is a larger architectural step.

    3.  **Challenges for GiGPO:**
        *   Ensuring that rollouts intended for the same GiGPO "group" (starting from identical initial conditions) are correctly managed and collected together, even if generated in parallel.
        *   Synchronizing data from parallel workers to the central components responsible for reward alignment and GiGPO calculation.

    *   *Action Item (for this phase):* Document these considerations and potential approaches. Detailed implementation would be a significant follow-on effort beyond this initial GiGPO integration.

*   **Files Involved:**
    *   This task is primarily for design documentation at this stage, perhaps in a new `development/notes/rl_scalability_plan.md`.

This detailed Phase 3 plan aims to make the GiGPO-enhanced `eval-protocol` usable and understandable, providing a complete loop from data generation to training signals, along with essential developer experience features.
