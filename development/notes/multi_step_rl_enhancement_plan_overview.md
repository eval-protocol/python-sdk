# Plan: Enhancing Reward-Kit for Multi-Step Reinforcement Learning (GiGPO Focused)

## 1. Introduction and Goals

**Overall Objective:** To significantly enhance the `eval-protocol` framework, enabling it to robustly support the development, training, and evaluation of agents that perform complex, multi-step interactions. This plan focuses on integrating advanced reinforcement learning (RL) techniques, specifically **Group-in-Group Policy Optimization (GiGPO)**, to handle long-horizon tasks and sparse rewards effectively. The primary user interface for defining rewards will remain simple, centered around `List[Message]` objects.

**Target Audience:** This plan is designed to be understandable and actionable by entry-level AI infrastructure engineers. It will break down complex tasks into smaller, more manageable pieces, explaining the "what," "why," and "how" for each.

**Key Challenges in Multi-Step RL (with GiGPO) we aim to address:**
*   **Simplified Reward Authoring:** Allowing users to define rewards based on familiar `List[Message]` structures, while the system handles complex advantage calculations.
*   **Effective Credit Assignment for Delayed Rewards:** Utilizing GiGPO to assign credit or blame to specific actions within long sequences, even with sparse terminal rewards.
*   **Flexible Episode and Batch Processing:** Supporting both pointwise (per-rollout) and batch-wise (cross-rollout) reward definitions.
*   **Robust Rollout Generation:** Establishing a clear process for generating the detailed trajectory data needed for RL.
*   **Clear Algorithmic Integration:** Ensuring the framework produces data usable by common RL algorithms like PPO, with advantages derived via GiGPO.

*(References: For general multi-step RL challenges and approaches, see `development/notes/agent_rl_survey.md`.)*

## 2. Core Concepts for Multi-Step RL & GiGPO (A Quick Primer)

Before diving into the plan, let's define some key terms:

*   **Episode:** A complete interaction sequence of an agent with its environment, from an initial state to a terminal state.
*   **Step (or Timestep):** A single point in an episode where the agent observes the environment, takes an action, and (conceptually) receives a reward and a new observation.
*   **`Message`:** The primary data unit (e.g., `{'role': 'user', 'content': '...'}`) that reward authors will interact with. A sequence of messages (`List[Message]`) represents a rollout's conversational history.
*   **`StepData` (Internal System Structure):** A detailed internal representation of a single step in an episode, containing `observation_data` (which includes `List[Message]`), the `action_taken` by the policy, `policy_logprobs`, `policy_value_estimate` (though GiGPO can be critic-free), `is_done` flags, and other RL-specific information. This is collected by the `RLRolloutWorker`.
*   **`EvaluateResult` (Extended):** The Pydantic model (from `eval_protocol.typed_interface.py`) that user reward functions will return. It will be extended to carry not just a `score` but also optional per-step `base_reward` information derived from messages.
*   **Base Rewards:** Scalar values assigned by the user's reward function to conceptual steps within a rollout, based on `List[Message]`. These are inputs to the GiGPO calculation.
*   **GiGPO (Group-in-Group Policy Optimization):** An RL algorithm that improves credit assignment by comparing trajectories (episode-level advantage `A_E`) and actions taken in similar states (step-level advantage `A_S`) within groups of rollouts. It's particularly useful for sparse rewards. *(Reference: `development/notes/gigpo_breakdown.md`)*
*   **Advantage (`A_t`):** A measure of how much better an action is compared to a baseline or average. In this plan, advantages will be primarily computed using GiGPO.
*   **Sandbox Execution Component:** The part of the system responsible for running user-provided Python code (their reward functions) in an isolated environment. (Implemented in `eval_protocol/evaluation.py`).

## 3. Proposed Enhancements (Phased Approach - GiGPO Focus)

This plan prioritizes the reward authoring experience first, assuming rollouts are available, and then details rollout generation and integration.

### Phase 1: Unified Reward Function Interface & System-Level GiGPO Setup

This phase focuses on defining how users write reward functions based on `List[Message]` and how the system prepares their output for GiGPO.

*   **Task 1.1: Define Core Data Structures for Reward Authoring and RL**
    *   **Objective:** Solidify `Message`, extend `EvaluateResult`, and define `StepOutput` (for `EvaluateResult`) and the internal `StepData` model.
    *   **Details:** See `development/notes/multi_step_rl_enhancement_plan_phase1.md#task-11-define-core-data-structures`.
*   **Task 1.2: Implement Unified User Reward Function Definition**
    *   **Objective:** Enable users to define reward functions (decorated with `@reward_function`) that take `List[Message]` (pointwise) or `List[List[Message]]` (batch-wise) and return an (extended) `EvaluateResult`.
    *   **Details:** See `development/notes/multi_step_rl_enhancement_plan_phase1.md#task-12-implement-unified-user-reward-function-definition`.
*   **Task 1.3: Adapt Sandbox Execution Component for Reward Functions**
    *   **Objective:** Modify the sandbox execution component (from `eval_protocol/evaluation.py`) to correctly invoke these user reward functions (pointwise or batch-wise) and handle the (extended) `EvaluateResult` output.
    *   **Details:** See `development/notes/multi_step_rl_enhancement_plan_phase1.md#task-13-adapt-sandbox-execution-component`.
*   **Task 1.4: Design System-Level Preprocessing for GiGPO**
    *   **Objective:** Define how the system takes the `EvaluateResult` (containing `score` and/or `step_outputs` with `base_rewards`) from the user's reward function and aligns it with the internal `StepData` (collected during rollouts). This prepares the necessary inputs for the main GiGPO calculation.
    *   **Details:** See `development/notes/multi_step_rl_enhancement_plan_phase1.md#task-14-design-system-level-preprocessing-for-gigpo`.

### Phase 2: Rollout Generation & GiGPO Advantage Calculation

This phase focuses on generating the `StepData` and then applying the GiGPO algorithm.

*   **Task 2.1: Implement `RLRolloutWorker` for Trajectory Generation**
    *   **Objective:** Develop the `RLRolloutWorker` class responsible for agent-environment interactions and collecting detailed `List[StepData]` per episode.
    *   **Details:** See `development/notes/multi_step_rl_enhancement_plan_phase2.md#task-21-implement-rlrolloutworker`. *(Initial concepts in old Phase 1, Task 1.1, 1.5 of `multi_step_rl_enhancement_plan_phase1.md`)*
*   **Task 2.2: Define `AgentPolicy` Interface**
    *   **Objective:** Create the interface for agent policies, specifying how `RLRolloutWorker` gets actions and other RL-relevant data (like `log_probs`).
    *   **Details:** See `development/notes/multi_step_rl_enhancement_plan_phase2.md#task-22-define-agentpolicy-interface`. *(Initial concepts in old Phase 1, Task 1.4)*
*   **Task 2.3: Implement Episode Termination Logic**
    *   **Objective:** Add mechanisms to `RLRolloutWorker` for ending episodes (max steps, agent signals, regex).
    *   **Details:** See `development/notes/multi_step_rl_enhancement_plan_phase2.md#task-23-implement-episode-termination-logic`. *(Initial concepts in old Phase 1, Task 1.1)*
*   **Task 2.4: Implement System-Level GiGPO Advantage Calculation**
    *   **Objective:** Create the system component and helper functions (`eval_protocol.rl_helpers`) that take the processed `StepData` (with aligned `base_rewards` or `final_scores` from Phase 1) and compute GiGPO advantages (`A_E`, `A_S`, `A_GiG`).
    *   **Details:** See `development/notes/multi_step_rl_enhancement_plan_phase2.md#task-24-implement-system-level-gigpo-advantage-calculation`. *(Based on `gigpo_breakdown.md` and concepts from old Phase 2, Task 2.2)*

### Phase 3: Training Integration, User Experience, and Scalability

This phase focuses on using the GiGPO advantages for training and ensuring the system is usable and robust.

*   **Task 3.1: Example RL Training Loop with PPO and GiGPO Advantages**
    *   **Objective:** Demonstrate end-to-end data flow into a PPO trainer using GiGPO advantages.
    *   **Details:** See `development/notes/multi_step_rl_enhancement_plan_phase3.md#task-31-example-rl-training-loop`. *(Old Phase 2, Task 2.4)*
*   **Task 3.2: Documentation and Examples for Reward Authors**
    *   **Objective:** Provide comprehensive guides on writing reward functions for scoring and for providing signals for GiGPO.
    *   **Details:** See `development/notes/multi_step_rl_enhancement_plan_phase3.md#task-32-documentation-for-reward-authors`.
*   **Task 3.3: Enhance Observability and Debugging for RL**
    *   **Objective:** Implement logging and tools for inspecting rollouts and GiGPO calculations.
    *   **Details:** See `development/notes/multi_step_rl_enhancement_plan_phase3.md#task-33-enhance-observability`. *(Old Phase 2, Task 2.3)*
*   **Task 3.4: (Future) Parallel Rollout Execution & Scalability**
    *   **Objective:** Explore and implement parallel execution of `RLRolloutWorker` for faster data collection.
    *   **Details:** See `development/notes/multi_step_rl_enhancement_plan_phase3.md#task-34-parallel-rollout-execution`. *(Old Phase 3, Task 3.1)*

This revised overview should guide the subsequent detailed phase documents.
