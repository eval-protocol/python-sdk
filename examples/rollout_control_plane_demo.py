#!/usr/bin/env python3
"""
Rollout Control Plane Demonstration

This script demonstrates the complete rollout system with control plane separation,
showing how:
1. Data plane (tool responses) contains only observations
2. Control plane (MCP resources) provides rewards/termination
3. Trajectories capture both planes correctly
4. Termination uses control plane signals

Run this to see the control plane separation in action and inspect
trajectory structure.
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add the mcp directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval_protocol.mcp.execution.manager import ExecutionManager
from eval_protocol.mcp.session.manager import GeneralMCPVectorEnv
from eval_protocol.types import DatasetRow, MCPSession, MCPToolCall, Trajectory


class DemoPolicy:
    """Demo policy that follows a predetermined path to reach the goal."""

    def __init__(self):
        # Actions that will reach the goal in FrozenLake 4x4
        self.actions = ["right", "down", "right", "down"]
        self.step_count = 0
        self.conversation_histories = {}

    async def __call__(self, tool_schemas, observations, system_prompts, user_prompts):
        """Return predetermined actions as tool calls."""
        tool_calls = []
        for i, _ in enumerate(observations):
            if self.step_count < len(self.actions):
                action = self.actions[self.step_count]
            else:
                action = "right"  # Default action

            tool_calls.append(MCPToolCall(tool_name="lake_move", arguments={"action": action}))

        print(f"🎯 Policy step {self.step_count}: action={action}")
        self.step_count += 1
        return tool_calls

    def add_tool_response(self, env_index, tool_call, response):
        """Mock method for conversation tracking."""
        if env_index not in self.conversation_histories:
            self.conversation_histories[env_index] = []
        self.conversation_histories[env_index].append({"tool_call": tool_call, "response": response})


async def demonstrate_control_plane_rollout():
    """Demonstrate rollout system with control plane separation."""
    print("🎯 CONTROL PLANE SEPARATION DEMONSTRATION")
    print("=" * 60)
    print("This demo shows the rollout system with strict control plane separation:")
    print("• Data plane: Tool responses contain ONLY observations")
    print("• Control plane: MCP resources provide rewards/termination")
    print("• Trajectories: Capture both planes separately")
    print()

    # Create mock sessions and dataset
    sessions = [
        MCPSession(
            session_id="demo_session",
            base_url="http://localhost:8000",
            seed=42,
            model_id="demo_model",
        )
    ]

    dataset_rows = [
        DatasetRow(
            id="demo_row",
            seed=42,
            system_prompt="You are navigating FrozenLake to reach the goal",
            user_prompt_template="Current position: {observation}. Navigate carefully to avoid holes.",
            environment_context={"grid_type": "4x4"},
        )
    ]

    # Create rollout manager
    execution_manager = ExecutionManager()

    # Mock the vector environment to simulate control plane separation
    with (
        patch.object(GeneralMCPVectorEnv, "__init__", return_value=None),
        patch.object(GeneralMCPVectorEnv, "reset") as mock_reset,
        patch.object(GeneralMCPVectorEnv, "step") as mock_step,
        patch.object(GeneralMCPVectorEnv, "close") as mock_close,
    ):
        # Setup mock vector environment
        mock_env = GeneralMCPVectorEnv(sessions, dataset_rows)
        mock_env.sessions = sessions
        mock_env.dataset_rows = dataset_rows
        mock_env.n = 1
        mock_env.user_prompt_formatter = lambda template, obs, context: template.format(observation=obs)

        print("🔧 Setting up mock environment with control plane separation...")

        # Mock reset to return initial state
        mock_reset.return_value = (
            [
                {
                    "position": 0,
                    "grid_layout": "S...\n.H.H\n...H\n..HG",
                    "description": "Start at position 0",
                }
            ],
            [
                [
                    {
                        "name": "lake_move",
                        "description": "Move in FrozenLake",
                        "input_schema": {
                            "type": "object",
                            "properties": {"action": {"type": "string"}},
                        },
                    }
                ]
            ],
            ["You are navigating FrozenLake to reach the goal"],
        )

        # Mock step responses with strict control plane separation
        step_responses = [
            # Step 1: Move right (position 0 -> 1)
            (
                [
                    {
                        "position": 1,
                        "grid_layout": ".P..\n.H.H\n...H\n..HG",
                        "description": "Moved right to position 1",
                    }
                ],
                [0.0],  # No reward yet (from control plane)
                [False],  # Not terminated (from control plane)
                [
                    {
                        "control_plane": {
                            "reward_source": "control_plane",
                            "status_source": "control_plane",
                        }
                    }
                ],
            ),
            # Step 2: Move down (position 1 -> 5)
            (
                [
                    {
                        "position": 5,
                        "grid_layout": "S...\n.P.H\n...H\n..HG",
                        "description": "Moved down to position 5",
                    }
                ],
                [0.0],  # No reward yet (from control plane)
                [False],  # Not terminated (from control plane)
                [
                    {
                        "control_plane": {
                            "reward_source": "control_plane",
                            "status_source": "control_plane",
                        }
                    }
                ],
            ),
            # Step 3: Move right (position 5 -> 6)
            (
                [
                    {
                        "position": 6,
                        "grid_layout": "S...\n.H.H\n..PH\n..HG",
                        "description": "Moved right to position 6",
                    }
                ],
                [0.0],  # No reward yet (from control plane)
                [False],  # Not terminated (from control plane)
                [
                    {
                        "control_plane": {
                            "reward_source": "control_plane",
                            "status_source": "control_plane",
                        }
                    }
                ],
            ),
            # Step 4: Move down (position 6 -> 10) - Fall into hole
            (
                [
                    {
                        "position": 10,
                        "grid_layout": "S...\n.H.H\n...H\n..PG",
                        "description": "Moved down to position 10",
                    }
                ],
                [0.0],  # No reward (fell in hole)
                [True],  # Terminated (from control plane)
                [
                    {
                        "control_plane": {
                            "reward_source": "control_plane",
                            "status_source": "control_plane",
                            "termination_reason": "fell_in_hole",
                        }
                    }
                ],
            ),
        ]

        step_call_count = 0

        def mock_step_side_effect(tool_calls):
            nonlocal step_call_count
            print(f"📡 MCP Tool Call {step_call_count + 1}: {tool_calls[0].tool_name}({tool_calls[0].arguments})")

            if step_call_count < len(step_responses):
                obs, reward, done, info = step_responses[step_call_count]
                step_call_count += 1

                print(f"   📊 Data Plane: {list(obs[0].keys())}")
                print(f"   🎛️  Control Plane: reward={reward[0]}, terminated={done[0]}")

                return obs, reward, done, info
            else:
                # Default to terminated if we run out of responses
                return (
                    [
                        {
                            "position": 15,
                            "grid_layout": "S...\n.H.H\n...H\n..HW",
                            "description": "Reached goal!",
                        }
                    ],
                    [1.0],  # Success reward from control plane
                    [True],  # Terminated from control plane
                    [
                        {
                            "control_plane": {
                                "reward_source": "control_plane",
                                "status_source": "control_plane",
                                "termination_reason": "goal_reached",
                            }
                        }
                    ],
                )

        mock_step.side_effect = mock_step_side_effect
        mock_close.return_value = None

        # Create demo policy
        policy = DemoPolicy()

        print("🚀 Executing rollout with control plane separation...")
        print()

        # Execute rollout
        trajectories = await execution_manager.execute_rollout(mock_env, policy, steps=10)

        print()
        print("📋 ROLLOUT RESULTS")
        print("=" * 60)

        # Analyze the trajectory
        trajectory = trajectories[0]

        print("Basic Trajectory Info:")
        print(f"  • Total Steps: {trajectory.steps}")
        print(f"  • Total Reward: {trajectory.total_reward}")
        print(f"  • Terminated: {trajectory.terminated}")
        print(f"  • Duration: {trajectory.duration:.3f}s")
        print()

        print("Data Plane Analysis (Observations):")
        print(f"  • Observation Count: {len(trajectory.observations)}")
        for i, obs in enumerate(trajectory.observations):
            if i == 0:
                print(f"    Initial: {obs}")
            elif i <= 3:
                print(f"    Step {i}: {obs}")
        print()

        print("Control Plane Analysis (Rewards/Termination):")
        print(f"  • Reward Count: {len(trajectory.rewards)}")
        print(f"  • Rewards: {trajectory.rewards}")
        print(f"  • Actions: {trajectory.actions}")
        print()

        # Validate control plane separation
        print("Control Plane Separation Validation:")

        # Check data plane contains no rewards
        data_plane_clean = True
        for obs in trajectory.observations:
            if "reward" in obs or "terminated" in obs:
                data_plane_clean = False
                break

        print(f"  ✅ Data plane clean (no rewards/termination): {data_plane_clean}")

        # Check control plane information exists
        has_control_plane_steps = hasattr(trajectory, "control_plane_steps")
        has_control_plane_summary = hasattr(trajectory, "control_plane_summary")

        print(f"  ✅ Control plane steps recorded: {has_control_plane_steps}")
        print(f"  ✅ Control plane summary available: {has_control_plane_summary}")

        if has_control_plane_steps:
            print(f"  • Control plane step count: {len(trajectory.control_plane_steps)}")
            print(f"  • Sample control plane step: {trajectory.control_plane_steps[0]}")

        if has_control_plane_summary:
            print(f"  • Control plane summary: {trajectory.control_plane_summary}")

        print()
        print("🎛️  TRAJECTORY STRUCTURE DEMONSTRATION")
        print("=" * 60)

        # Show complete trajectory structure
        trajectory_dict = {
            "session_id": trajectory.session.session_id,
            "basic_info": {
                "steps": trajectory.steps,
                "total_reward": trajectory.total_reward,
                "terminated": trajectory.terminated,
                "duration": trajectory.duration,
            },
            "data_plane": {
                "observations": trajectory.observations[:2],  # Show first 2 only
                "actions": trajectory.actions,
            },
            "control_plane": {
                "rewards": trajectory.rewards,
                "control_plane_steps": (trajectory.control_plane_steps if has_control_plane_steps else []),
                "control_plane_summary": (trajectory.control_plane_summary if has_control_plane_summary else {}),
            },
        }

        print("Complete trajectory structure (first 2 observations only):")
        print(json.dumps(trajectory_dict, indent=2))

        print()
        print("✅ DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("Key achievements:")
        print("• Strict data/control plane separation maintained")
        print("• Tool responses contain ONLY observations")
        print("• Control plane provides rewards and termination")
        print("• Trajectories capture both planes for analysis")
        print("• Rollout system uses control plane for termination decisions")


if __name__ == "__main__":
    asyncio.run(demonstrate_control_plane_rollout())
