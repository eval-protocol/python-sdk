#!/usr/bin/env python3
"""
Quick test script to validate multi-session functionality of FrozenLake MCP server.

This script tests:
1. Multi-session environment creation
2. Session-based control plane state management
3. Static policy integration
4. Basic functionality verification

Usage:
    python test_multi_session.py
"""

import asyncio
import sys
import time
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from static_policy import StaticPolicy

import reward_kit as rk


async def test_multi_session():
    """Test multi-session functionality with static policy."""

    print("🧪 === MULTI-SESSION FUNCTIONALITY TEST ===")

    # Create test dataset with different seeds
    test_dataset = [
        {
            "id": "test_001",
            "system_prompt": "You are playing FrozenLake, a 4x4 grid game. Use lake_move tool with LEFT, DOWN, RIGHT, UP actions.",
            "user_prompt_template": "Current state: {observation}. Choose your move wisely.",
            "environment_context": {
                "game": "FrozenLake",
                "map_name": "4x4",
                "seed": 42,
            },
        },
        {
            "id": "test_002",
            "system_prompt": "You are playing FrozenLake, a 4x4 grid game. Use lake_move tool with LEFT, DOWN, RIGHT, UP actions.",
            "user_prompt_template": "Current state: {observation}. Navigate carefully to avoid holes.",
            "environment_context": {
                "game": "FrozenLake",
                "map_name": "4x4",
                "seed": 123,
            },
        },
    ]

    print(f"📊 Created {len(test_dataset)} test environments")

    # Create static policy for fast testing
    policy = StaticPolicy(action_sequence=["RIGHT", "RIGHT", "DOWN", "DOWN"])

    try:
        # Create environments (assumes server is running on localhost:8000)
        envs = rk.make(
            "http://localhost:8000/mcp/",
            dataset=test_dataset,
            model_id=policy.model_id,
        )

        print(f"✅ Connected to {len(envs.sessions)} environment sessions")

        # Run rollout
        start_time = time.time()
        trajectories = await rk.rollout(envs, policy=policy, steps=6)
        duration = time.time() - start_time

        # Validate results
        assert len(trajectories) == len(
            test_dataset
        ), "Should have trajectory for each environment"

        print(
            f"✅ Multi-session test completed with {len(trajectories)} trajectories in {duration:.2f}s"
        )

        # Print trajectory summaries
        print("📊 Multi-Session Trajectory Summary:")
        for i, traj in enumerate(trajectories):
            dataset_entry = test_dataset[i]
            seed = dataset_entry.get("environment_context", {}).get("seed", "N/A")
            print(
                f"  Trajectory {i} (seed: {seed}): {traj.steps} steps, reward: {traj.total_reward:.2f}, terminated: {traj.terminated}"
            )

        # Validate that different seeds produce different results (if they do)
        unique_rewards = set(traj.total_reward for traj in trajectories)
        print(f"📈 Unique rewards across environments: {unique_rewards}")

        # Clean up
        await envs.close()

        print("✅ Multi-session test completed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("💡 Make sure the FrozenLake MCP server is running on localhost:8000")
        print("   Run: python server.py --port 8000")
        return False

    return True


async def test_static_policy_only():
    """Test static policy functionality independently."""

    print("\n🧪 === STATIC POLICY UNIT TEST ===")

    # Create policy
    policy = StaticPolicy(action_sequence=["RIGHT", "DOWN", "LEFT", "UP"])

    # Initialize
    policy.initialize_conversations(
        n_envs=2,
        system_prompts=["Test system prompt 1", "Test system prompt 2"],
        user_prompts=["Test user prompt 1", "Test user prompt 2"],
    )

    # Test action generation
    for step in range(6):
        actions = await policy.act(
            observations=[None, None], tools_list=[[], []], env_indices=[0, 1]
        )

        assert len(actions) == 2, "Should generate action for each environment"

        for i, action in enumerate(actions):
            assert action["type"] == "function", "Should be function call"
            assert action["function"]["name"] == "lake_move", "Should call lake_move"

            import json

            args = json.loads(action["function"]["arguments"])
            assert "action" in args, "Should have action argument"
            assert args["action"] in [
                "RIGHT",
                "DOWN",
                "LEFT",
                "UP",
            ], "Should be valid action"

            print(f"  Step {step}, Env {i}: {args['action']}")

    print("✅ Static policy unit test completed successfully!")
    return True


async def main():
    """Run all tests."""
    print("🚀 Starting multi-session functionality validation...")

    # Test static policy first (no server required)
    success_policy = await test_static_policy_only()

    if success_policy:
        print("\n" + "=" * 50)
        # Test multi-session functionality (requires server)
        success_multi = await test_multi_session()

        if success_multi:
            print(
                "\n🎉 All tests passed! Multi-session functionality is working correctly."
            )
        else:
            print("\n⚠️ Multi-session test failed, but static policy works.")
    else:
        print("\n❌ Static policy test failed.")


if __name__ == "__main__":
    asyncio.run(main())
