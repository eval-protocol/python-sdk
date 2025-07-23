#!/usr/bin/env python3
"""
FrozenLake MCP-Gym Rollout Example

This script demonstrates the north star vision for MCP-Gym rollouts,
showing how to use the clean rk.rollout() interface with MCP-Gym environments.

Usage:
    python rollout_example.py
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from frozen_lake_mcp import FrozenLakeMcp
from eval_protocol.mcp import McpGym


class SimplePolicy:
    """
    Simple policy for demonstration purposes.

    In the north star vision, this would be a Fireworks policy that
    receives tool schemas and makes tool calls via LLM inference.
    """

    def __init__(self, strategy: str = "right_down"):
        """
        Initialize simple policy.

        Args:
            strategy: Strategy to use ("right_down", "random", "shortest_path")
        """
        self.strategy = strategy
        self.actions = ["LEFT", "DOWN", "RIGHT", "UP"]

    async def __call__(self, env: McpGym, step: int) -> str:
        """
        Generate action for current step.

        Args:
            env: MCP-Gym environment
            step: Current step number

        Returns:
            Action string
        """
        if self.strategy == "right_down":
            # Simple strategy: alternate between RIGHT and DOWN
            return "RIGHT" if step % 2 == 0 else "DOWN"
        elif self.strategy == "random":
            import random

            return random.choice(self.actions)
        else:
            # Default to RIGHT
            return "RIGHT"


class MCPGymRolloutManager:
    """
    Rollout manager implementing the north star vision.

    This demonstrates how rk.rollout() would work in the full implementation,
    handling multiple environments and policies with clean separation between
    data plane (tool calls) and control plane (rewards).
    """

    def __init__(self):
        """Initialize rollout manager."""
        self.logger = logging.getLogger(__name__)

    async def rollout(
        self, envs: List[McpGym], policy: SimplePolicy, steps: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Execute rollouts using the north star interface.

        Args:
            envs: List of MCP-Gym environments
            policy: Policy for action selection
            steps: Maximum number of steps per rollout

        Returns:
            List of trajectory dictionaries
        """
        self.logger.info(
            f"Starting rollouts with {len(envs)} environments for {steps} steps"
        )

        trajectories = []

        for i, env in enumerate(envs):
            self.logger.info(f"Running rollout {i+1}/{len(envs)}")

            trajectory = {
                "environment": env.__class__.__name__,
                "seed": env.seed,
                "steps": [],
                "total_reward": 0.0,
                "terminated": False,
                "truncated": False,
            }

            for step in range(steps):
                # Get action from policy
                action = await policy(env, step)

                # Execute action via MCP tool call
                result = env.call_tool("lake_move", {"action": action})

                if result.is_error:
                    self.logger.error(f"Tool call failed: {result.error_message}")
                    break

                step_data = result.content
                trajectory["steps"].append(
                    {
                        "step": step,
                        "action": action,
                        "observation": step_data.get("observation", ""),
                        "reward": step_data.get("reward", 0.0),
                        "terminated": step_data.get("terminated", False),
                        "truncated": step_data.get("truncated", False),
                    }
                )

                trajectory["total_reward"] += step_data.get("reward", 0.0)

                self.logger.info(
                    f"Step {step}: {action} -> reward={step_data.get('reward', 0.0)}"
                )

                # Check for episode termination
                if step_data.get("terminated") or step_data.get("truncated"):
                    trajectory["terminated"] = step_data.get("terminated", False)
                    trajectory["truncated"] = step_data.get("truncated", False)
                    self.logger.info(f"Episode finished at step {step}")
                    break

            trajectories.append(trajectory)
            self.logger.info(
                f"Rollout {i+1} completed: {trajectory['total_reward']} total reward"
            )

        return trajectories

    def print_trajectory_summary(self, trajectories: List[Dict[str, Any]]):
        """Print summary of rollout results."""
        print("\n" + "=" * 60)
        print("ROLLOUT SUMMARY")
        print("=" * 60)

        for i, traj in enumerate(trajectories):
            print(f"\nTrajectory {i+1}:")
            print(f"  Environment: {traj['environment']}")
            print(f"  Seed: {traj['seed']}")
            print(f"  Steps: {len(traj['steps'])}")
            print(f"  Total Reward: {traj['total_reward']}")
            print(f"  Terminated: {traj['terminated']}")
            print(f"  Truncated: {traj['truncated']}")

            if traj["steps"]:
                print(f"  First Action: {traj['steps'][0]['action']}")
                print(f"  Last Action: {traj['steps'][-1]['action']}")
                print(f"  Final Reward: {traj['steps'][-1]['reward']}")

        # Overall statistics
        total_reward = sum(traj["total_reward"] for traj in trajectories)
        avg_reward = total_reward / len(trajectories) if trajectories else 0
        success_rate = (
            sum(1 for traj in trajectories if traj["total_reward"] > 0)
            / len(trajectories)
            if trajectories
            else 0
        )

        print(f"\nOverall Statistics:")
        print(f"  Total Environments: {len(trajectories)}")
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Success Rate: {success_rate:.2%}")


# North Star Interface Implementation
class RewardKit:
    """
    Simplified RewardKit interface implementing north star vision.

    This demonstrates how the rk.rollout() interface would work
    in the full implementation.
    """

    @staticmethod
    async def rollout(
        envs: List[McpGym], policy: SimplePolicy, steps: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Execute rollouts using north star interface.

        This is the clean interface from the north star document:
        rollouts = await rk.rollout(envs, policy, steps=20)
        """
        manager = MCPGymRolloutManager()
        return await manager.rollout(envs, policy, steps)


# Create a module-level instance for the north star interface
rk = RewardKit()


async def main():
    """Main demonstration of the north star vision."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("FrozenLake MCP-Gym Rollout Example")
    print("=" * 40)
    print("Demonstrating the north star vision for MCP-Gym rollouts")
    print()

    # Create multiple environments with different seeds
    envs = [FrozenLakeMcp(seed=42), FrozenLakeMcp(seed=123), FrozenLakeMcp(seed=456)]

    print(f"Created {len(envs)} environments with different seeds")
    print()

    # Create policy
    policy = SimplePolicy(strategy="right_down")
    print(f"Using policy: {policy.strategy}")
    print()

    # Execute rollouts using the north star interface
    print("Executing rollouts...")
    trajectories = await rk.rollout(envs, policy, steps=20)

    # Print results
    manager = MCPGymRolloutManager()
    manager.print_trajectory_summary(trajectories)

    print("\n" + "=" * 60)
    print("NORTH STAR VISION DEMONSTRATION COMPLETE")
    print("=" * 60)
    print()
    print("Key Features Demonstrated:")
    print("1. Clean McpGym base class inheritance")
    print("2. Simple tool registration with @self.mcp.tool() decorator")
    print("3. Clean rk.rollout() interface")
    print("4. Separation of data plane (tool calls) and control plane (rewards)")
    print("5. Multiple environments with different seeds")
    print("6. Consistent API across different environment types")


if __name__ == "__main__":
    asyncio.run(main())
