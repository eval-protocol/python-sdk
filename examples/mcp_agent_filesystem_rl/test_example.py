#!/usr/bin/env python3
"""
Test script for MCP Agent Filesystem RL Example

This script verifies that the example setup is working correctly by:
1. Testing the dataset format
2. Testing the reward function with mock data
3. Testing MCP server connectivity (if running)
"""

import asyncio
import json  # Ensure json is imported
import os
import sys
from pathlib import Path

# Add the eval-protocol package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eval_protocol.models import (
    EvaluateResult,
    Message,
)


def test_dataset_format():
    """Test that the dataset file is correctly formatted."""
    print("Testing dataset format...")

    dataset_file_path = Path(__file__).parent / "dataset.jsonl"
    assert dataset_file_path.exists(), f"Dataset file not found: {dataset_file_path}"
    with open(dataset_file_path, "r") as f:
        line = f.readline().strip()
        data = json.loads(line)

    assert "user_query" in data, "Dataset missing 'user_query' field"
    assert "ground_truth_for_eval" in data, "Dataset missing 'ground_truth_for_eval' field"
    # Specific checks for this example's dataset content
    assert "file_to_move.txt" in data["user_query"], "User query doesn't mention target file"
    assert "/data/source_dir/" in data["user_query"], "User query doesn't mention source directory"
    assert "/data/target_dir/" in data["user_query"], "User query doesn't mention target directory"
    assert (
        "move /data/source_dir/file_to_move.txt to /data/target_dir/file_to_move.txt" in data["ground_truth_for_eval"]
    )

    print("✓ Dataset format is correct")


def test_reward_function_import():
    """Test that the reward function can be imported and has correct signature."""
    print("Testing reward function import...")

    # Import EvaluateResult specifically within this function's scope
    from eval_protocol.models import EvaluateResult
    from examples.mcp_agent_filesystem_rl import main as filesystem_rl_main

    assert hasattr(filesystem_rl_main, "evaluate"), "Reward function 'evaluate' not found in main.py"

    # Test with mock data
    # The 'evaluate' function expects: messages, ground_truth, final_filesystem_state, task_description
    mock_messages = [
        Message(
            role="user",
            content="Move the file /data/source_dir/file_to_move.txt to /data/target_dir/file_to_move.txt",
        ),
        Message(role="assistant", content="Okay, I will move the file."),
    ]
    mock_ground_truth = "move file_to_move.txt to target_dir"  # Matched to main.py's parse_ground_truth

    # Mock a final_filesystem_state that indicates success for the move operation
    # This structure should match what the 'directory_tree' tool (or similar) would output,
    # and what the 'evaluate' function in main.py expects after parsing.
    mock_final_fs_state_success = {
        "content": [
            {
                "type": "text",
                "text": json.dumps(
                    [  # Simulating output of directory_tree tool for /data
                        {
                            "name": "source_dir",
                            "type": "directory",
                            "children": [],
                        },  # file_to_move.txt is gone
                        {
                            "name": "target_dir",
                            "type": "directory",
                            "children": [
                                {
                                    "name": "file_to_move.txt",
                                    "type": "file",
                                }  # file_to_move.txt is here
                            ],
                        },
                    ]
                ),
            }
        ]
    }

    result_success = filesystem_rl_main.evaluate(
        messages=mock_messages,
        ground_truth=mock_ground_truth,
        final_filesystem_state=mock_final_fs_state_success,
        task_description="Test move success",
    )

    assert isinstance(result_success, EvaluateResult), "evaluate function did not return an EvaluateResult"
    # Based on main.py logic, a perfect move should result in score 1.0
    assert (
        result_success.score == 1.0
    ), f"Expected score 1.0 for mock success, got {result_success.score}. Reason: {result_success.reason}"
    assert result_success.is_score_valid

    print("✓ Reward function 'evaluate' import and basic validation works")


async def test_mcp_server_connectivity():
    """Test connectivity to MCP intermediary server if it's running."""
    print("Testing MCP server connectivity...")

    try:
        import httpx

        async with httpx.AsyncClient() as client:
            # Try to connect to the MCP server
            response = await client.get("http://localhost:8001/health", timeout=5.0)
            if response.status_code == 200:
                print("✓ MCP server is running and accessible")
                return True
            else:
                print(f"⚠ MCP server responded with status {response.status_code}")
                return False

    except httpx.ConnectError:
        print("⚠ MCP server not running (this is okay for testing)")
        return False
    except Exception as e:
        print(f"⚠ Error connecting to MCP server: {e}")
        return False


def test_config_file():
    """Test that the config file is valid."""
    print("Testing config file...")

    import yaml

    config_file_path = Path(__file__).parent / "config.yaml"
    assert config_file_path.exists(), f"Config file not found: {config_file_path}"
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)

    assert "generation" in config, "Config missing 'generation' field"
    assert "model_name" in config["generation"], "Config missing 'generation.model_name' field"
    assert "dataset" in config, "Config missing 'dataset' field"
    assert "reward" in config, "Config missing 'reward' field"  # Changed from reward_function
    assert "agent" in config, "Config missing 'agent' field"  # Changed from agent_config

    # Check agent config specifics
    agent_config = config["agent"]
    assert agent_config["type"] == "mcp_agent", "Agent type should be 'mcp_agent'"
    assert "mcp_backend_ref" in agent_config, "Agent config missing 'mcp_backend_ref'"  # Key name changed

    print("✓ Config file is valid")


async def main():
    """Run all tests."""
    print("Running MCP Agent Filesystem RL Example Tests")
    print("=" * 50)

    try:
        test_template_structure()
        test_dataset_format()
        test_reward_function_import()
        test_config_file()
        await test_mcp_server_connectivity()

        print("\n" + "=" * 50)
        print("✓ All tests passed! Example setup is ready.")
        print("\nTo run the full example:")
        print("1. Start the MCP intermediary server:")
        print("   python -m eval_protocol.mcp_agent.main")
        print("2. Run the evaluation:")
        print("   eval-protocol run --config config.yaml")

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
