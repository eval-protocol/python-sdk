# Frozen Lake Implementation Context

This document provides context for the implementation of the Frozen Lake example within the `eval-protocol` framework.

## High-Level Goal

The primary objective is to create a robust and reproducible reinforcement learning environment for the Frozen Lake game. This involves allowing an LLM-based agent to interact with the game, and critically, enabling a data-driven approach to evaluations where initial conditions (like random seeds) are controlled by a dataset.

## Core Components

The implementation is distributed across several key files:

-   **`examples/frozen_lake/client/dataset.jsonl`**: The source of truth for evaluation runs. Each line defines a scenario, specifying the `seed` for the environment's initial state.
-   **`examples/frozen_lake/client/task_def.yaml`**: The main configuration file for the task. It points to the dataset and defines how many rollouts to perform for each sample in the dataset.
-   **`examples/frozen_lake/server/http_rollout_server.py`**: A FastAPI server that wraps the Frozen Lake game logic, exposing it via an HTTP API that the `eval-protocol` agent can interact with.
-   **`examples/frozen_lake/gymnasium_frozen_lake_server.py`**: The core game logic, which wraps the official `gymnasium` Frozen Lake environment. It is responsible for accepting a `seed` to create a deterministic starting state.
-   **`examples/frozen_lake/client/reward.py`**: A reward function that evaluates the agent's performance based on the outcome of the game (e.g., reaching the goal).
-   **`eval_protocol/agent/`**: The core agent framework, including the `TaskManager` and `Orchestrator`, which together manage the data-driven execution of rollouts based on the task definition and dataset.

## Data-Driven Rollout Flow

The evaluation process follows a clear, data-driven flow:

1.  The **TaskManager** reads the `task_def.yaml`.
2.  It loads the scenarios from the specified `dataset.jsonl` file.
3.  For each scenario (i.e., each `seed` in the dataset), it schedules `num_rollouts_per_sample` rollouts.
4.  For each individual rollout, the **Orchestrator** is invoked with the specific `seed`.
5.  The **Orchestrator** passes the `seed` to the **HttpRolloutResource**.
6.  The **HttpRolloutResource** sends the `seed` in a request to the `/start_episode` endpoint of the **http_rollout_server**.
7.  The server uses the `seed` to initialize the **GymnasiumFrozenLakeGame** in a deterministic state.
8.  The agent then plays the game, and the final outcome is evaluated by the reward function.

This architecture ensures that evaluations are reproducible and that the agent's performance can be measured across a controlled set of initial conditions.
