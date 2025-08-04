# Recorded Trajectories - FrozenLake MCP-Gym

This directory contains recorded trajectory files from the FrozenLake MCP-Gym tests. These files are preserved for review and debugging purposes.

## Files

### `production_trajectory.jsonl`
Recorded trajectories from the production server test (`test_production_server_record_and_replay`). Contains the complete interaction log between the policy and the MCP server during live recording.

### `conda_isolation_trajectory.jsonl`
Recorded trajectories from the conda isolation test (`test_frozen_lake_step_by_step`). Demonstrates that the MCP-Gym framework works correctly with `CondaServerProcessManager` for isolated environments.

### `playback_only_test.jsonl`
Simple test recording used by `test_production_only_recorded_policy` to verify that the playback mechanism works correctly.

## File Format

Each `.jsonl` file contains one JSON object per line, representing a single step in the trajectory:

```json
{
  "env_index": 0,
  "step": 1,
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": null, "tool_calls": [...]}
  ]
}
```

## Usage

### Review Trajectories
These files can be opened and reviewed to understand:
- How the policy interacts with the FrozenLake environment
- What tool calls are made and their responses
- Whether the environment behaves deterministically
- Performance characteristics of different configurations

### Debugging
If tests fail, these recordings provide detailed logs of what happened during the interaction, making it easier to debug issues.

### CI/CD
In CI environments, existing recordings are used for fast playback-only tests, avoiding the need for live LLM API calls.

## Regenerating Recordings

To force regeneration of recordings (e.g., after changes to the environment):

```bash
# Remove existing recordings
rm examples/frozen_lake_mcp/tests/recordings/*.jsonl

# Run tests to generate new recordings (auto-records when not in CI)
pytest examples/frozen_lake_mcp/tests/test_record_and_replay_e2e.py -v

# CI mode - only runs playback with existing recordings
CI=true pytest examples/frozen_lake_mcp/tests/test_record_and_replay_e2e.py -v
```

## North Star Compliance

These recordings demonstrate key aspects of the north star MCP-Gym vision:
- **Tool-based interaction**: All environment interactions use `lake_move` tool calls
- **Proper MCP context**: Uses real FastMCP `Context` objects
- **Data/Control plane separation**: Observations in tool responses, rewards in metadata
- **CondaServerProcessManager compatibility**: Isolated execution works correctly
- **Deterministic behavior**: Same seeds produce same trajectories
