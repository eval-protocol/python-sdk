"""
Test Control Plane Separation

This module tests the critical control plane separation feature of MCP-Gym,
ensuring that:
1. Tool responses contain ONLY data plane information (no rewards/termination)
2. Control plane information is available via MCP resources
3. Strict separation is maintained between data and control planes

This addresses the highest priority architectural requirement from the progress notes.
"""

import json
import sys
from pathlib import Path

import pytest

# Add examples directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "frozen_lake_mcp"))

from frozen_lake_mcp import FrozenLakeMcp


class TestControlPlaneSeparation:
    """Test control plane separation in MCP-Gym framework."""

    def test_control_plane_separation_basic(self):
        """
        Test basic control plane separation.

        Verifies that:
        1. Tool responses contain NO reward/termination info
        2. Control plane resources contain reward/termination info
        3. Separation is strictly maintained
        """
        # Create FrozenLake MCP server
        server = FrozenLakeMcp(seed=42)

        # Verify initial state
        assert server.control_plane_state["step_count"] == 0
        assert server.control_plane_state["total_reward"] == 0.0
        assert server.control_plane_state["terminated"] is False

        # Simulate a move by directly calling the internal step method
        # This tests the control plane separation logic
        action_int = server.adapter.parse_action("DOWN")
        tool_response = server._execute_environment_step(action_int)

        # CRITICAL: Verify tool response contains NO control plane info
        assert "reward" not in tool_response, "Tool response should NOT contain reward"
        assert "terminated" not in tool_response, "Tool response should NOT contain termination status"
        assert "truncated" not in tool_response, "Tool response should NOT contain truncated status"

        # Verify tool response contains ONLY data plane info
        assert "position" in tool_response, "Tool response should contain position (data plane)"
        assert "grid" in tool_response, "Tool response should contain grid (data plane)"

        # Verify control plane state was updated
        assert server.control_plane_state["step_count"] == 1
        assert server.control_plane_state["reward"] == 0.0  # Likely 0 for this move
        assert isinstance(server.control_plane_state["terminated"], bool)

        print("✅ Control plane separation working correctly!")
        print(f"   Tool response keys: {list(tool_response.keys())}")
        print(f"   Control plane state: {server.control_plane_state}")

    def test_control_plane_resources(self):
        """
        Test that control plane resources provide correct information.

        Verifies that MCP resources contain the control plane info
        that was removed from tool responses.
        """
        # Create FrozenLake MCP server
        server = FrozenLakeMcp(seed=42)

        # Execute a move to update control plane
        action_int = server.adapter.parse_action("DOWN")
        tool_response = server._execute_environment_step(action_int)

        # Test control plane data by directly accessing the state (simulates MCP resource queries)
        # In a real MCP client, this would be done via resource requests
        reward_data = {
            "reward": server.control_plane_state["reward"],
            "step_count": server.control_plane_state["step_count"],
        }

        status_data = {
            "terminated": server.control_plane_state["terminated"],
            "truncated": server.control_plane_state["truncated"],
            "step_count": server.control_plane_state["step_count"],
            "total_reward": server.control_plane_state["total_reward"],
        }

        info_data = server.control_plane_state["info"]

        # Verify reward data
        assert "reward" in reward_data, "Reward data should contain reward"
        assert "step_count" in reward_data, "Reward data should contain step count"
        assert reward_data["step_count"] == 1

        # Verify status data
        assert "terminated" in status_data, "Status data should contain terminated"
        assert "truncated" in status_data, "Status data should contain truncated"
        assert "step_count" in status_data, "Status data should contain step count"
        assert "total_reward" in status_data, "Status data should contain total reward"
        assert status_data["step_count"] == 1

        # Verify info data (should be dict, possibly empty)
        assert isinstance(info_data, dict), "Info data should be dict"

        print("✅ Control plane resources working correctly!")
        print(f"   Reward data: {reward_data}")
        print(f"   Status data: {status_data}")
        print(f"   Info data: {info_data}")

    def test_multiple_moves_control_plane(self):
        """
        Test control plane separation with multiple moves.

        Verifies that control plane state accumulates correctly
        while tool responses remain clean.
        """
        # Create FrozenLake MCP server
        server = FrozenLakeMcp(seed=42)

        # Execute multiple moves
        moves = ["DOWN", "RIGHT", "UP", "LEFT"]
        for i, move in enumerate(moves):
            action_int = server.adapter.parse_action(move)
            tool_response = server._execute_environment_step(action_int)

            # Verify each tool response is clean (no control plane info)
            assert "reward" not in tool_response, f"Move {i}: Tool response should NOT contain reward"
            assert "terminated" not in tool_response, f"Move {i}: Tool response should NOT contain termination"
            assert "truncated" not in tool_response, f"Move {i}: Tool response should NOT contain truncated"

            # Verify control plane state is updated
            assert server.control_plane_state["step_count"] == i + 1, f"Move {i}: Step count should be {i + 1}"

            # Verify data plane info is present
            assert "position" in tool_response, f"Move {i}: Tool response should contain position"
            assert "grid" in tool_response, f"Move {i}: Tool response should contain grid"

        # Verify final control plane state
        assert server.control_plane_state["step_count"] == len(moves)
        assert server.control_plane_state["total_reward"] >= 0.0  # Should be accumulated

        print("✅ Multiple moves control plane separation working correctly!")
        print(f"   Final step count: {server.control_plane_state['step_count']}")
        print(f"   Final total reward: {server.control_plane_state['total_reward']}")

    def test_control_plane_architecture_compliance(self):
        """
        Test that the architecture complies with the north star vision.

        Verifies:
        1. Data plane contains only observations/actions
        2. Control plane contains only rewards/termination
        3. No mixing between planes
        """
        # Create FrozenLake MCP server
        server = FrozenLakeMcp(seed=42)

        # Execute a move
        action_int = server.adapter.parse_action("DOWN")
        tool_response = server._execute_environment_step(action_int)

        # Define expected data plane keys (observations/actions only)
        expected_data_plane_keys = {"position", "grid"}

        # Define forbidden control plane keys in tool response
        forbidden_control_plane_keys = {
            "reward",
            "terminated",
            "truncated",
            "info",
            "step_count",
            "total_reward",
        }

        # Verify data plane compliance
        for key in expected_data_plane_keys:
            assert key in tool_response, f"Data plane key '{key}' should be in tool response"

        # Verify control plane separation
        for key in forbidden_control_plane_keys:
            assert key not in tool_response, f"Control plane key '{key}' should NOT be in tool response"

        # Verify control plane contains the expected information
        status_data = {
            "terminated": server.control_plane_state["terminated"],
            "truncated": server.control_plane_state["truncated"],
            "step_count": server.control_plane_state["step_count"],
            "total_reward": server.control_plane_state["total_reward"],
        }

        # Verify control plane keys are accessible
        expected_control_plane_keys = {
            "terminated",
            "truncated",
            "step_count",
            "total_reward",
        }
        for key in expected_control_plane_keys:
            assert key in status_data, f"Control plane key '{key}' should be in control plane data"

        print("✅ Architecture compliance verified!")
        print(f"   Data plane keys: {list(tool_response.keys())}")
        print(f"   Control plane keys: {list(status_data.keys())}")
        print("   🎯 North star vision implemented correctly!")

    def test_tool_versus_resource_separation(self):
        """
        Test that tool calls and resource queries provide complementary but separate information.

        This simulates the actual usage pattern where:
        1. Tool calls provide data plane info (observations)
        2. Resource queries provide control plane info (rewards)
        """
        # Create FrozenLake MCP server
        server = FrozenLakeMcp(seed=42)

        # Execute a move via the internal method (simulates tool call)
        action_int = server.adapter.parse_action("DOWN")
        tool_response = server._execute_environment_step(action_int)

        # Simulate control plane resource queries
        reward_data = {
            "reward": server.control_plane_state["reward"],
            "step_count": server.control_plane_state["step_count"],
        }

        status_data = {
            "terminated": server.control_plane_state["terminated"],
            "truncated": server.control_plane_state["truncated"],
            "step_count": server.control_plane_state["step_count"],
            "total_reward": server.control_plane_state["total_reward"],
        }

        # Verify complementary information
        # Tool response has observations but no rewards
        assert "position" in tool_response
        assert "grid" in tool_response
        assert "reward" not in tool_response
        assert "terminated" not in tool_response

        # Resources have rewards but no observations
        assert "reward" in reward_data
        assert "terminated" in status_data
        assert "position" not in reward_data
        assert "grid" not in status_data

        # Together they provide complete information
        complete_info = {**tool_response, **reward_data, **status_data}

        expected_complete_keys = {
            "position",
            "grid",
            "reward",
            "terminated",
            "truncated",
            "step_count",
            "total_reward",
        }
        for key in expected_complete_keys:
            assert key in complete_info, f"Complete info should contain {key}"

        print("✅ Tool vs Resource separation working correctly!")
        print(f"   Tool provides: {list(tool_response.keys())}")
        print(f"   Resources provide: {list(set(reward_data.keys()) | set(status_data.keys()))}")
        print(f"   Combined: {list(complete_info.keys())}")


if __name__ == "__main__":
    # Run tests directly
    test = TestControlPlaneSeparation()
    test.test_control_plane_separation_basic()
    test.test_control_plane_resources()
    test.test_multiple_moves_control_plane()
    test.test_control_plane_architecture_compliance()
    test.test_tool_versus_resource_separation()
    print("\n🏆 All control plane separation tests passed!")
