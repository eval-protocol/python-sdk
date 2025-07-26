# Plan: Forkable Filesystem Scenario for RL Rollouts

**Status: Phase 1 (Single Instance Test) - Completed**

---

## 1. Objective

Enable Reinforcement Learning (RL) rollouts with the `mcp/filesystem` backend, where each RL episode (or instance within a batch) operates on an isolated, independently initialized copy of a filesystem state. This allows for reproducible experiments and avoids interference between concurrent rollouts.

## 2. Chosen Forking Strategy for Filesystem

For the `mcp/filesystem` backend (and similar backends where state is primarily managed on a mounted Docker volume), we will implement a "host directory copy" strategy:

1.  **Template Directory**: A template directory structure will be defined on the host machine. This directory will contain the desired initial set of files and folders for an RL scenario.
2.  **Configuration**: The `template_data_path_host` field in `BackendServerConfig` (within `mcp_agent_config.yaml`) will point to this host template directory.
3.  **Copy-on-Provision**: When the `RewardKitIntermediaryServer` (via `LocalDockerOrchestrationClient`) provisions a new `mcp/filesystem` instance:
    *   It will create a new, unique directory on the host machine (e.g., under a base path like `/tmp/rk_mcp_instance_data/<session_id>/<instance_uuid>/`).
    *   It will copy the entire contents of the specified `template_data_path_host` into this unique instance directory.
    *   It will then start a standard `mcp/filesystem` Docker container, mounting this unique instance directory to the container's `/data` path (or the configured data path for the filesystem server).
4.  **Isolation**: Each instance thus operates on its own copy of the initial file state, ensuring isolation.
5.  **Cleanup**: Upon deprovisioning, the unique host directory created for the instance will be deleted.

This strategy is preferred over `docker commit` for `mcp/filesystem` because the base Docker image is generic, and only the initial volume content needs to be forked.

## 3. Required Code Changes

### 3.1. `eval_protocol/mcp_agent/config.py`

*   **`BackendServerConfig`**:
    *   Verify that the `template_data_path_host: Optional[str]` field exists. Its description should be updated or understood to include its use as a source directory for the copy-on-provision strategy for applicable backend types like "filesystem".

### 3.2. `eval_protocol/mcp_agent/orchestration/local_docker_client.py`

*   **Imports**: Add `import shutil` and ensure `from pathlib import Path` is present.
*   **Constants**: Define `DEFAULT_INSTANCE_DATA_BASE_PATH = Path("/tmp/rk_mcp_instance_data")` (or similar configurable path).
*   **`__init__`**: Initialize `self.instance_data_base_path`.
*   **`startup`**: Ensure `self.instance_data_base_path.mkdir(parents=True, exist_ok=True)` is called.
*   **`provision_instances` Method**:
    *   Before the instance creation loop, check if `backend_config.backend_type == "filesystem"` and `backend_config.template_data_path_host` is set.
    *   If true:
        *   Inside the loop for each instance:
            *   Construct `instance_host_data_path = self.instance_data_base_path / session_id / backend_config.backend_name_ref / instance_uuid`.
            *   Create the directory: `instance_host_data_path.mkdir(parents=True, exist_ok=True)`.
            *   Copy template: `shutil.copytree(backend_config.template_data_path_host, instance_host_data_path, dirs_exist_ok=True)`.
            *   Store `str(instance_host_data_path)` in `instance_internal_details["instance_host_data_path"]`.
            *   Dynamically create `instance_specific_volumes` for the Docker run command: ` {str(instance_host_data_path): {"bind": "/data", "mode": "rw"}}` (assuming `/data` is the standard target path for `mcp/filesystem`). This will override any static `container_volumes` from the config for this instance.
        *   Pass `instance_specific_volumes` to the `self.docker_client.containers.run(...)` call.
    *   If not using the template copy method, the existing logic for `container_volumes` (from `backend_config.container_volumes`) should apply.
*   **`deprovision_instances` Method**:
    *   After stopping and removing the container:
        *   Retrieve `instance_host_data_path_str = details.get("instance_host_data_path")`.
        *   If `instance_host_data_path_str`:
            *   `shutil.rmtree(instance_host_data_path_str, ignore_errors=True)` to clean up the copied directory.

## 4. Test Plan

### 4.1. Prepare Host Template Directory

*   Create a directory structure on the host, e.g.:
    ```
    ./mcp_agent_test_templates/fs_move_scenario/
    ├── source_dir/
    │   └── file_to_move.txt  (contains "Hello from source")
    └── target_dir/           (empty)
    ```

### 4.2. Update `mcp_agent_config.yaml`

*   Modify the `filesystem_test` backend entry to use this template:
    ```yaml
    - backend_name_ref: "filesystem_test"
      backend_type: "filesystem"
      orchestration_mode: "local_docker"
      instance_scoping: "session"
      mcp_transport: "stdio"
      docker_image: "mcp/filesystem"
      container_command: ["/data"] # Served directory inside container
      template_data_path_host: "./mcp_agent_test_templates/fs_move_scenario/" # Path to host template
      # container_volumes can be omitted or will be overridden if template_data_path_host is used for filesystem type
    ```

### 4.3. Develop New Test Script (e.g., `tests/mcp_agent/test_rl_filesystem_scenario.py`)

*   This script will simulate an RL episode:
    1.  **Initialize Session**: Call `initialize_session` on the intermediary server to get one `filesystem_test` instance.
    2.  **Verify Initial State (Optional but Recommended)**:
        *   Use `call_backend_tool` with `tool_name="list_directory"` and args `{"path": "/data/source_dir"}`. Verify `file_to_move.txt` is present.
        *   Use `call_backend_tool` with `tool_name="read_file"` and args `{"path": "/data/source_dir/file_to_move.txt"}`. Verify content.
        *   Use `call_backend_tool` with `tool_name="list_directory"` and args `{"path": "/data/target_dir"}`. Verify it's empty.
    3.  **Simulate Agent Action**:
        *   Use `call_backend_tool` with `tool_name="move_file"` (or the equivalent tool name reported by `list_tools` for `mcp/filesystem`, e.g., `move_file`) with arguments to move `/data/source_dir/file_to_move.txt` to `/data/target_dir/file_to_move.txt`.
    4.  **Verify Final State & Determine Reward (Simulated)**:
        *   Use `list_directory` on `/data/target_dir/`. Verify `file_to_move.txt` is present.
        *   Use `list_directory` on `/data/source_dir/`. Verify `file_to_move.txt` is absent.
        *   Assert these conditions. A successful outcome implies a positive reward in an RL context.
    5.  **Cleanup Session**: Call `cleanup_session`.

## 5. Success Criteria

*   The `LocalDockerOrchestrationClient` correctly provisions and deprovisions `mcp/filesystem` instances with isolated, copied state from the host template.
*   The new test script (`test_rl_filesystem_scenario.py`) passes, demonstrating end-to-end functionality of initializing with a template, performing a state-changing tool call, and verifying the outcome.
*   No orphaned Docker containers or temporary host directories remain after tests.

## 6. Future Considerations

*   Generalize this "host directory copy" forking strategy for other volume-stateful backends (e.g., `mcp-server-motherduck` by copying database files).
*   Error handling for `shutil` operations (e.g., template path not found, copy errors, cleanup errors).

---

## Phase 2: Multi-Instance Testing and Full RL Example Integration

**Objective**: To validate the system's capability to handle multiple concurrent RL rollouts with isolated, stateful MCP environments and to create a complete end-to-end example using the `eval-protocol` CLI.

### 2.1. Verify Multi-Instance Functionality

*   **Objective**: Confirm that the `LocalDockerOrchestrationClient` and `RewardKitIntermediaryServer` can correctly provision, manage, and isolate multiple `filesystem_test` instances concurrently within a single session when requested by `initialize_session`.
*   **Test Plan**:
    *   Modify `tests/mcp_agent/test_rl_filesystem_scenario.py` (or create a new dedicated test script, e.g., `tests/mcp_agent/test_multi_instance_filesystem.py`).
    *   In the `initialize_session` call, request `num_instances: N` (e.g., N=2 or 3) for the `filesystem_test` backend.
    *   The test script should iterate through each provisioned instance ID.
    *   For each instance:
        *   Perform the "move file" scenario (verify initial state from template, move file, verify final state) independently.
        *   Ensure that actions on one instance do not affect the state of other instances. This can be implicitly verified if each instance correctly reflects its own templated start and subsequent modification.
    *   Verify that all temporary host directories for all instances are cleaned up during `cleanup_session`.
*   **Success Criteria**: The multi-instance test passes, demonstrating correct state isolation and concurrent operation for N instances.

### 2.2. Develop Full End-to-End RL Example

*   **Objective**: Create a complete, runnable example showcasing an LLM agent interacting with forkable filesystem environments, with evaluation orchestrated by the `eval-protocol` CLI using the `mcp-agent` component.
*   **Example Location**: A new directory, e.g., `examples/mcp_agent_filesystem_rl/`.
*   **Components**:
    1.  **Dataset (`dataset.jsonl`)**:
        *   Create a simple JSONL file. For instance, a single line:
            ```json
            {"prompt": "You have access to a filesystem. Please move the file named 'important_document.txt' from the '/data/source_files/' directory to the '/data/archive/' directory."}
            ```
        *   The template directory for `filesystem_test` (from `mcp_agent_config.yaml`) should be updated or a new one created to reflect this initial state (e.g., `source_files/important_document.txt` exists, `archive/` is an empty directory).
    2.  **Reward Function (`reward_function.py`)**:
        *   A Python function decorated with `@reward_function`.
        *   **Input**: It will receive the LLM's generated output (which should ideally be a tool call to `move_file` or a sequence of actions leading to that). It will also need access to the `rk_session_id` and the specific `instance_id` for the current rollout to interact with the correct MCP environment. (The mechanism for passing this context to the reward function needs to be confirmed based on `eval-protocol`'s capabilities when using an agent).
        *   **Logic**:
            *   Construct an MCP client (similar to `test_rl_filesystem_scenario.py`) to connect to the `RewardKitIntermediaryServer`.
            *   Use `call_backend_tool` to interact with the specific `filesystem_test` instance for the current rollout.
            *   Tools to use: `list_directory` (or `get_file_info`) to check:
                *   If `important_document.txt` is now in `/data/archive/`.
                *   If `important_document.txt` is no longer in `/data/source_files/`.
        *   **Output**: Return a reward score (e.g., `1.0` for a successful move, `0.0` otherwise).
    3.  **Configuration (`config.yaml` or `rewardkit.yaml` within the example directory)**:
        *   `model`: `accounts/fireworks/models/qwen3-235b-a22b` (or as specified).
        *   `rollout_settings`:
            *   `count`: 4 (for N concurrent rollouts).
            *   `generation_config`: `{"temperature": 1.0}`.
        *   `dataset`: `{"path": "dataset.jsonl"}`.
        *   `reward_function`: `{"path": "reward_function.py"}`.
        *   `agent_config`:
            *   `type`: `mcp_agent` (or the appropriate identifier for the MCP agent).
            *   `mcp_agent_config_path`: Path to the main `mcp_agent_config.yaml` (which defines the `filesystem_test` backend with its template).
            *   `intermediary_server_url`: `http://localhost:8001/mcp` (if the CLI doesn't manage the server automatically).
    4.  **README.md**:
        *   Detailed instructions on:
            *   Setting up the template directory for this example.
            *   Ensuring the `mcp_agent_config.yaml` points to this template.
            *   The `eval-protocol` CLI command to run the evaluation (e.g., `eval-protocol run --config ./config.yaml` or `eval-protocol mcp-agent run ...`).
*   **Execution & Verification**:
    *   The `eval-protocol` CLI command should successfully launch 4 concurrent rollouts.
    *   Each rollout should use an independent, templated filesystem environment.
    *   The LLM's actions should be evaluated by the reward function against the state of its specific environment.
    *   The CLI should produce an evaluation report.
*   **Success Criteria**: The example runs end-to-end, demonstrating concurrent rollouts with isolated, stateful environments managed by the `mcp-agent`, and rewards are correctly calculated based on interactions with these environments.
