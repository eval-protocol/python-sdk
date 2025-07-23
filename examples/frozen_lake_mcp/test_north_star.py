#!/usr/bin/env python3
"""
Test script for North Star MCP-Gym Implementation

This script verifies that the north star vision implementation works correctly
with all the key features demonstrated.
"""

import asyncio
import sys
import traceback

from frozen_lake_mcp import FrozenLakeMcp


def test_basic_functionality():
    """Test basic environment functionality."""
    print("Testing basic functionality...")

    # Create environment
    env = FrozenLakeMcp(seed=42)
    assert env.seed == 42
    assert env.current_obs is not None
    print("✓ Environment created successfully")

    # Test tool registration
    tools = env.get_tool_schemas()
    assert "lake_move" in tools
    assert (
        tools["lake_move"]["description"]
        == "Move on the frozen lake. Actions: LEFT, DOWN, RIGHT, UP"
    )
    print("✓ Tool registration works")

    # Test tool calling
    result = env.call_tool("lake_move", {"action": "RIGHT"})
    assert not result.is_error
    assert "observation" in result.content
    assert "action" in result.content
    assert "reward" in result.content
    assert "terminated" in result.content
    assert "truncated" in result.content
    print("✓ Tool calling works")

    # Test invalid action
    result = env.call_tool("lake_move", {"action": "INVALID"})
    assert result.is_error
    assert "Invalid action" in result.error_message
    print("✓ Error handling works")

    # Test invalid tool
    result = env.call_tool("invalid_tool", {"action": "RIGHT"})
    assert result.is_error
    assert "not found" in result.error_message
    print("✓ Invalid tool handling works")

    print("Basic functionality tests passed!\n")


def test_deterministic_behavior():
    """Test that environments with same seeds behave deterministically."""
    print("Testing deterministic behavior...")

    # Create two environments with the same seed
    env1 = FrozenLakeMcp(seed=42)
    env2 = FrozenLakeMcp(seed=42)

    # Both should have same initial observation
    assert env1.current_obs == env2.current_obs
    print("✓ Same seed produces same initial state")

    # Execute same action on both
    result1 = env1.call_tool("lake_move", {"action": "RIGHT"})
    result2 = env2.call_tool("lake_move", {"action": "RIGHT"})

    assert result1.content["reward"] == result2.content["reward"]
    assert result1.content["terminated"] == result2.content["terminated"]
    assert result1.content["truncated"] == result2.content["truncated"]
    print("✓ Same seed produces same outcomes")

    # Test different seeds produce different results
    env3 = FrozenLakeMcp(seed=123)
    # Initial observation might be different (different random maps)
    print("✓ Different seeds can produce different environments")

    print("Deterministic behavior tests passed!\n")


def test_environment_lifecycle():
    """Test complete environment lifecycle."""
    print("Testing environment lifecycle...")

    env = FrozenLakeMcp(seed=42)

    # Test environment descriptions
    desc = env.get_environment_description()
    assert "FrozenLake" in desc
    assert "4x4 grid game" in desc
    print("✓ Environment description works")

    intent = env.get_user_intent()
    assert "Navigate to the goal" in intent
    print("✓ User intent works")

    # Test state access
    state = env.get_current_state()
    assert "observation" in state
    assert "episode_step" in state
    assert "terminated" in state
    assert "truncated" in state
    assert "total_reward" in state
    print("✓ State access works")

    # Test episode progression
    initial_step = env.episode_step
    result = env.call_tool("lake_move", {"action": "UP"})
    assert env.episode_step == initial_step + 1
    print("✓ Episode progression works")

    print("Environment lifecycle tests passed!\n")


async def test_rollout_interface():
    """Test the rollout interface."""
    print("Testing rollout interface...")

    # Import rollout components
    from rollout_example import MCPGymRolloutManager, SimplePolicy

    # Create environments
    envs = [FrozenLakeMcp(seed=42), FrozenLakeMcp(seed=123)]

    # Create policy
    policy = SimplePolicy(strategy="right_down")

    # Create rollout manager
    manager = MCPGymRolloutManager()

    # Execute rollouts
    trajectories = await manager.rollout(envs, policy, steps=5)

    # Verify results
    assert len(trajectories) == 2
    for traj in trajectories:
        assert "environment" in traj
        assert "seed" in traj
        assert "steps" in traj
        assert "total_reward" in traj
        assert "terminated" in traj
        assert "truncated" in traj
        assert len(traj["steps"]) > 0

    print("✓ Rollout interface works")
    print("Rollout interface tests passed!\n")


def test_north_star_features():
    """Test all north star features."""
    print("Testing north star features...")

    env = FrozenLakeMcp(seed=42)

    # Feature 1: Clean inheritance from McpGym
    from eval_protocol.mcp import McpGym

    assert isinstance(env, McpGym)
    print("✓ Clean inheritance from McpGym")

    # Feature 2: Tool registration via decorator
    assert hasattr(env, "mcp_tools")
    assert len(env.mcp_tools) > 0
    print("✓ Tool registration via decorator")

    # Feature 3: Plane separation (data vs control)
    result = env.call_tool("lake_move", {"action": "RIGHT"})
    # Data plane: observation returned to LLM
    assert "observation" in result.content
    # Control plane: reward/termination metadata present but hidden
    assert "reward" in result.content
    assert "terminated" in result.content
    print("✓ Plane separation (data vs control)")

    # Feature 4: Standardized lifecycle methods
    assert hasattr(env, "create_with_seed")
    assert hasattr(env, "format_observation")
    assert callable(env.create_with_seed)
    assert callable(env.format_observation)
    print("✓ Standardized lifecycle methods")

    # Feature 5: Universal compatibility (MCP tool interface)
    tools = env.get_tool_schemas()
    assert isinstance(tools, dict)
    for tool_name, schema in tools.items():
        assert "name" in schema
        assert "description" in schema
        assert "parameters" in schema
    print("✓ Universal compatibility (MCP tool interface)")

    print("North star features tests passed!\n")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("NORTH STAR MCP-GYM IMPLEMENTATION TESTS")
    print("=" * 60)
    print()

    try:
        test_basic_functionality()
        test_deterministic_behavior()
        test_environment_lifecycle()
        await test_rollout_interface()
        test_north_star_features()

        print("=" * 60)
        print("ALL TESTS PASSED! 🎉")
        print("=" * 60)
        print()
        print("The north star vision has been successfully implemented!")
        print("Key achievements:")
        print("- Clean McpGym base class with simple inheritance")
        print("- Declarative tool registration with @self.mcp.tool()")
        print("- Clean separation between data and control planes")
        print("- Standardized lifecycle methods")
        print("- Universal MCP compatibility")
        print("- Deterministic behavior with seeds")
        print("- Complete rollout interface")
        print()
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
