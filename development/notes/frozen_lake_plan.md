# Frozen Lake Example Plan: Data-Driven Rollouts

This document outlines the plan for refactoring the Frozen Lake example to use a data-driven evaluation workflow. The goal is to make the system more robust, extensible, and aligned with standard practices in reinforcement learning research.

The core principle is to treat the initial conditions of an environment (like a random seed) as data. Each row in a dataset will define a specific scenario, and the framework will run a configurable number of rollouts for each scenario.

### 1. The Dataset (`dataset.jsonl`)

The foundation of this new approach is a dataset file that defines the experimental conditions.

-   **Action:** Create a new dataset file at `examples/frozen_lake/client/dataset.jsonl`.
-   **Format:** Each line in the file will be a JSON object representing a single experimental sample. Initially, this will just contain a unique `id` and a `seed`.
-   **Example Content:**
    ```json
    {"id": "run_001", "seed": 42}
    {"id": "run_002", "seed": 123}
    {"id": "run_003", "seed": 555}
    {"id": "run_004", "seed": 678}
    ```

### 2. The Task Definition (`task_def.yaml`)

The task definition will be updated to reference the dataset and specify how many rollouts (`N`) to perform for each sample.

-   **File to Modify:** `examples/frozen_lake/client/task_def.yaml`
-   **Changes:**
    -   Remove the old `num_rollouts` field.
    -   Add `dataset_path` to point to our new `dataset.jsonl` file.
    -   Add `num_rollouts_per_sample` to define `N`.
-   **Example:**
    ```yaml
    name: "frozen_lake_http_rollout"
    description: "Evaluate an agent's ability to navigate a Frozen Lake environment via HTTP rollout"

    # Data-driven configuration
    dataset_path: "examples/frozen_lake/client/dataset.jsonl"
    num_rollouts_per_sample: 5 # This is 'N', the number of rollouts per seed

    # Resource configuration remains the same
    resource_type: "http_rollout"
    # ... (rest of the file)
    ```

### 3. Core Framework Modifications

The following changes will plumb the `seed` from the dataset through the framework to the game environment.

1.  **Data Model (`eval_protocol/models.py`):**
    -   Update `TaskDefinitionModel` to include `dataset_path: Optional[str]` and `num_rollouts_per_sample: int`.

2.  **TaskManager (`eval_protocol/agent/task_manager.py`):**
    -   Modify the `execute_tasks` method to load samples from the `dataset_path`.
    -   For each sample, generate `num_rollouts_per_sample` rollout jobs.
    -   Pass the sample data (containing the `seed`) for each job down to the `Orchestrator`.

3.  **Orchestrator (`eval_protocol/agent/orchestrator.py`):**
    -   Modify `execute_task_poc` to accept `sample_data` as a parameter.
    -   Pass this data to the resource's `initialize` method: `await episode_resource.initialize(**sample_data)`.

4.  **HTTP Rollout Resource (`eval_protocol/agent/resources/http_rollout_resource.py`):**
    -   The `initialize` method will accept `**kwargs`.
    -   These `kwargs` (the `sample_data`) will be sent as the JSON body of the POST request to the `/start_episode` endpoint.

5.  **HTTP Rollout Server & Protocol:**
    -   The `/start_episode` endpoint in `examples/frozen_lake/server/http_rollout_server.py` will be updated to accept a JSON request body.
    -   It will pass the entire request body as keyword arguments to the `GymnasiumFrozenLakeGame` constructor: `game = FrozenLakeGame(**request_data)`.
    -   The `StartEpisodeRequest` model in `eval_protocol/agent/resources/http_rollout_protocol.py` will be updated to allow arbitrary extra fields.

6.  **Gymnasium Game (`examples/frozen_lake/gymnasium_frozen_lake_server.py`):**
    -   The `__init__` method of `GymnasiumFrozenLakeGame` will be changed to accept `**kwargs`.
    -   The `reset` method will use the `seed` from these arguments to initialize the environment deterministically: `self.env.reset(seed=self.seed)`.

### 4. Visualization of the Flow

```mermaid
sequenceDiagram
    participant TaskManager
    participant Orchestrator
    participant Resource as HttpRolloutResource
    participant Server as http_rollout_server
    participant Game as GymnasiumFrozenLakeGame

    TaskManager->>TaskManager: Reads dataset.jsonl
    TaskManager->>Orchestrator: execute_task_poc(sample_data={"seed": 42})
    Orchestrator->>Resource: fork()
    Orchestrator->>Resource: initialize(**sample_data)
    Resource->>Server: POST /start_episode (body={"seed": 42})
    Server->>Game: __init__(**{"seed": 42})
    Game->>Game: self.env.reset(seed=42)
    Game-->>Server: observation
    Server-->>Resource: {episode_id, observation}
    Resource-->>Orchestrator: (initialization complete)
    Orchestrator->>Orchestrator: (proceeds with agent interaction)
