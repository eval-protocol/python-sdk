"""
Test Rollout System with Control Plane Integration

This module tests the complete rollout system with control plane separation,
ensuring that:
1. Data plane (tool responses) contain only observations
2. Control plane (MCP resources) contain rewards/termination info
3. Trajectories capture both planes correctly
4. Termination decisions use control plane signals
5. Rollout system works end-to-end with separated architecture

This validates the complete implementation of the control plane separation
feature in the rollout execution pipeline.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add examples directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "frozen_lake_mcp"))


from eval_protocol.mcp.execution.manager import ExecutionManager
from eval_protocol.mcp.session.manager import GeneralMCPVectorEnv
from eval_protocol.mcp.types import DatasetRow, MCPSession, MCPToolCall, Trajectory


class MockPolicy:
    """Mock policy for testing that returns predetermined actions."""

    def __init__(self, actions=None):
        self.actions = actions or ["right", "down", "right", "down"]
        self.step_count = 0

    async def __call__(self, tool_schema, env_index, conversation_history):
        """Return predetermined actions as tool calls."""
        if self.step_count < len(self.actions):
            action = self.actions[self.step_count]
        else:
            action = "right"  # Default action

        tool_calls = []
        tool_call = MCPToolCall(tool_name="lake_move", arguments={"action": action})
        tool_calls.append(tool_call)

        self.step_count += 1
        return tool_calls, None

    def add_tool_response(
        self,
        env_index,
        tool_call,
        response,
        conversation_history,
        reward=0.0,
        done=False,
        info=None,
    ):
        """Mock method for conversation tracking."""
        conversation_history.append(
            {
                "tool_call": tool_call,
                "response": response,
                "reward": reward,
                "done": done,
                "info": info,
            }
        )


class TestRolloutControlPlaneIntegration:
    """Test rollout system with control plane integration."""

    def setup_method(self):
        """Setup test environment."""
        self.execution_manager = ExecutionManager()

    @pytest.mark.asyncio
    async def test_rollout_with_control_plane_separation(self):
        """
        Test that rollout system properly handles control plane separation.

        This test validates:
        1. Tool responses contain only data plane info
        2. Control plane resources provide rewards/termination
        3. Trajectories capture both planes correctly
        4. Termination uses control plane signals
        """
        # Create mock sessions
        sessions = [
            MCPSession(
                session_id="test_session_1",
                base_url="http://localhost:8000",
                seed=42,
                model_id="test_model",
            )
        ]

        # Create dataset rows
        dataset_rows = [
            DatasetRow(
                id="test_row_1",
                seed=42,
                system_prompt="You are playing FrozenLake",
                user_prompt_template="Navigate to the goal",
                environment_context={"grid_type": "4x4"},
            )
        ]

        # Mock the vector environment to simulate control plane separation
        with (
            patch.object(GeneralMCPVectorEnv, "__init__", return_value=None),
            patch.object(GeneralMCPVectorEnv, "reset") as mock_reset,
            patch.object(GeneralMCPVectorEnv, "step") as mock_step,
            patch.object(GeneralMCPVectorEnv, "close") as mock_close,
            patch.object(GeneralMCPVectorEnv, "format_user_prompt") as mock_format_user_prompt,
        ):

            # Setup mock vector environment
            mock_env = GeneralMCPVectorEnv(sessions, dataset_rows)
            mock_env.sessions = sessions
            mock_env.dataset_rows = dataset_rows
            mock_env.n = 1
            mock_env.user_prompt_formatter = lambda template, obs, context: template

            # Mock reset to return initial state
            mock_reset.return_value = (
                {"position": 0, "grid": "4x4 FrozenLake"},  # single observation
                [{"name": "lake_move", "description": "Move in FrozenLake"}],  # single tool schema
            )

            # Mock format_user_prompt to return template
            mock_format_user_prompt.return_value = "Navigate to the goal"

            # Mock step to simulate control plane separation
            step_responses = [
                # Step 1: Move right, no reward, not terminated
                (
                    {
                        "position": 1,
                        "grid": "4x4 FrozenLake",
                    },  # single observation (data plane only)
                    0.0,  # single reward (from control plane)
                    False,  # single done (from control plane)
                    {
                        "control_plane": {
                            "reward_source": "control_plane",
                            "status_source": "control_plane",
                        }
                    },  # single info
                ),
                # Step 2: Move down, no reward, not terminated
                (
                    {"position": 5, "grid": "4x4 FrozenLake"},
                    0.0,
                    False,
                    {
                        "control_plane": {
                            "reward_source": "control_plane",
                            "status_source": "control_plane",
                        }
                    },
                ),
                # Step 3: Move right, reach goal, reward, terminated
                (
                    {"position": 6, "grid": "4x4 FrozenLake"},
                    1.0,  # Success reward from control plane
                    True,  # Terminated from control plane
                    {
                        "control_plane": {
                            "reward_source": "control_plane",
                            "status_source": "control_plane",
                        }
                    },
                ),
            ]

            step_call_count = 0

            def mock_step_side_effect(env_index, tool_call):
                nonlocal step_call_count
                if step_call_count < len(step_responses):
                    result = step_responses[step_call_count]
                    step_call_count += 1
                    return result
                else:
                    # Default to terminated if we run out of responses
                    return (
                        {"position": 6, "grid": "4x4 FrozenLake"},
                        0.0,
                        True,
                        {
                            "control_plane": {
                                "reward_source": "control_plane",
                                "status_source": "control_plane",
                            }
                        },
                    )

            mock_step.side_effect = mock_step_side_effect
            mock_close.return_value = None

            # Create mock policy
            policy = MockPolicy(["right", "down", "right"])

            # Execute rollout
            trajectories = await self.execution_manager.execute_rollouts(mock_env, policy, steps=10)

            # Validate results
            assert len(trajectories) == 1, "Should have one trajectory"
            trajectory = trajectories[0]

            # Validate basic trajectory structure
            assert trajectory.steps == 3, f"Expected 3 steps, got {trajectory.steps}"
            assert trajectory.total_reward == 1.0, f"Expected reward 1.0, got {trajectory.total_reward}"
            assert trajectory.terminated == True, "Trajectory should be terminated"

            # Validate data plane information (observations)
            assert len(trajectory.observations) == 4, "Should have 4 observations (initial + 3 steps)"
            for obs in trajectory.observations:
                # Data plane should only contain observations
                assert "position" in obs, "Observation should contain position"
                assert "grid" in obs, "Observation should contain grid"
                # Data plane should NOT contain rewards or termination
                assert "reward" not in obs, "Data plane should not contain reward"
                assert "terminated" not in obs, "Data plane should not contain termination"

            # Validate control plane information
            assert len(trajectory.rewards) == 3, "Should have 3 reward values"
            assert trajectory.rewards == [
                0.0,
                0.0,
                1.0,
            ], "Rewards should match control plane"

            # Validate enhanced control plane tracking
            assert hasattr(trajectory, "control_plane_steps"), "Should have control plane steps"
            assert len(trajectory.control_plane_steps) == 3, "Should have 3 control plane steps"

            for i, cp_step in enumerate(trajectory.control_plane_steps):
                assert "step" in cp_step, "Control plane step should have step number"
                assert "reward" in cp_step, "Control plane step should have reward"
                assert "terminated" in cp_step, "Control plane step should have terminated status"
                assert "info" in cp_step, "Control plane step should have control plane info"
                assert "tool_calls" in cp_step, "Control plane step should have tool calls"

            # Validate control plane summary
            assert hasattr(trajectory, "control_plane_summary"), "Should have control plane summary"
            summary = trajectory.control_plane_summary
            assert summary["total_reward"] == 1.0, "Summary should have correct total reward"
            assert summary["termination_reason"] == "control_plane_signal", "Should terminate via control plane"
            assert summary["final_step"] == 2, "Should record final step"

            # Validate policy interaction
            assert policy.step_count == 3, "Policy should have been called 3 times"

    @pytest.mark.asyncio
    async def test_rollout_trajectory_recording_with_control_plane(self):
        """
        Test that trajectory recording captures both data and control plane information.
        """
        # Create a simple test scenario with manual trajectory construction
        session = MCPSession(
            session_id="test_session",
            base_url="http://localhost",
            seed=42,
            model_id="test_model",
        )

        # Create a trajectory and manually populate it with control plane data
        trajectory = Trajectory(
            session=session,
            observations=[],
            actions=[],
            rewards=[],
            terminated=False,
            total_reward=0.0,
            steps=0,
            duration=0.0,
            control_plane_steps=[],
            control_plane_summary={},
            termination_reason="",
            conversation_history=[],
            llm_usage_summary={},
        )

        # Simulate steps with control plane separation
        steps = [
            {
                "observation": {"position": 1, "grid": "4x4"},
                "action": "lake_move(right)",
                "reward": 0.0,
                "terminated": False,
                "control_plane_info": {"reward_source": "control_plane"},
            },
            {
                "observation": {"position": 15, "grid": "4x4"},
                "action": "lake_move(down)",
                "reward": 1.0,
                "terminated": True,
                "control_plane_info": {
                    "reward_source": "control_plane",
                    "status_source": "control_plane",
                },
            },
        ]

        # Simulate the rollout manager's trajectory building logic
        trajectory.control_plane_steps = []

        for i, step_data in enumerate(steps):
            # Data plane recording
            trajectory.observations.append(step_data["observation"])
            trajectory.actions.append(step_data["action"])
            trajectory.rewards.append(step_data["reward"])
            trajectory.total_reward += step_data["reward"]
            trajectory.steps += 1

            # Control plane recording
            control_plane_step = {
                "step": i,
                "reward": step_data["reward"],
                "terminated": step_data["terminated"],
                "info": step_data["control_plane_info"],
                "tool_call": step_data["action"],
            }
            trajectory.control_plane_steps.append(control_plane_step)

            if step_data["terminated"]:
                trajectory.terminated = True
                trajectory.control_plane_summary = {
                    "total_reward": trajectory.total_reward,
                    "termination_reason": "control_plane_signal",
                    "final_step": i,
                    "control_plane_source": step_data["control_plane_info"],
                }

        # Validate the trajectory structure
        assert len(trajectory.observations) == 2, "Should have 2 observations"
        assert len(trajectory.actions) == 2, "Should have 2 actions"
        assert len(trajectory.rewards) == 2, "Should have 2 rewards"
        assert len(trajectory.control_plane_steps) == 2, "Should have 2 control plane steps"

        # Validate data plane contains only observations
        for obs in trajectory.observations:
            assert "position" in obs, "Observation should contain position"
            assert "reward" not in obs, "Data plane should not contain reward"
            assert "terminated" not in obs, "Data plane should not contain termination"

        # Validate control plane contains rewards and termination info
        assert trajectory.rewards == [0.0, 1.0], "Control plane should have rewards"
        assert trajectory.total_reward == 1.0, "Control plane should track total reward"
        assert trajectory.terminated == True, "Control plane should handle termination"

        # Validate control plane summary
        assert trajectory.control_plane_summary["total_reward"] == 1.0
        assert trajectory.control_plane_summary["termination_reason"] == "control_plane_signal"
        assert trajectory.control_plane_summary["final_step"] == 1

    @pytest.mark.asyncio
    async def test_rollout_handles_control_plane_failure_gracefully(self):
        """
        Test that rollout system handles control plane failures gracefully.
        """
        # Create mock sessions
        sessions = [
            MCPSession(
                session_id="test_session",
                base_url="http://localhost",
                seed=42,
                model_id="test_model",
            )
        ]
        dataset_rows = [
            DatasetRow(
                id="test_row",
                seed=42,
                system_prompt="Test",
                user_prompt_template="Test",
                environment_context={},
            )
        ]

        with (
            patch.object(GeneralMCPVectorEnv, "__init__", return_value=None),
            patch.object(GeneralMCPVectorEnv, "reset") as mock_reset,
            patch.object(GeneralMCPVectorEnv, "step") as mock_step,
            patch.object(GeneralMCPVectorEnv, "close") as mock_close,
            patch.object(GeneralMCPVectorEnv, "format_user_prompt") as mock_format_user_prompt,
        ):

            mock_env = GeneralMCPVectorEnv(sessions, dataset_rows)
            mock_env.sessions = sessions
            mock_env.dataset_rows = dataset_rows
            mock_env.n = 1
            mock_env.user_prompt_formatter = lambda template, obs, context: template

            # Mock reset
            mock_reset.return_value = (
                {"position": 0},  # single observation
                [{"name": "move", "description": "Move"}],  # single tool schema
            )

            # Mock step to simulate control plane failure (no control plane info)
            mock_step.return_value = (
                {"position": 1},  # single observation
                0.0,  # single reward (fallback)
                False,  # single done (fallback)
                {},  # single info (no control plane)
            )

            mock_close.return_value = None

            # Execute rollout with control plane failure
            policy = MockPolicy(["right"])
            trajectories = await self.execution_manager.execute_rollouts(mock_env, policy, steps=1)

            # Should still work, but without control plane info
            assert len(trajectories) == 1
            trajectory = trajectories[0]
            assert trajectory.steps == 1
            assert trajectory.total_reward == 0.0

            # Control plane steps should still be recorded (even if empty)
            assert hasattr(trajectory, "control_plane_steps")
            assert len(trajectory.control_plane_steps) == 1
            assert trajectory.control_plane_steps[0]["info"] == {}

    def test_control_plane_trajectory_serialization(self):
        """
        Test that trajectories with control plane information can be serialized.
        """
        # Create a trajectory with control plane data
        session = MCPSession(
            session_id="test",
            base_url="http://localhost",
            seed=42,
            model_id="test_model",
        )
        trajectory = Trajectory(
            session=session,
            observations=[{"position": 0}, {"position": 1}],
            actions=["move(right)"],
            rewards=[0.0],
            terminated=False,
            total_reward=0.0,
            steps=1,
            duration=1.0,
            control_plane_steps=[],
            control_plane_summary={},
            termination_reason="",
            conversation_history=[],
            llm_usage_summary={},
        )

        # Add control plane data
        trajectory.control_plane_steps = [
            {
                "step": 0,
                "reward": 0.0,
                "terminated": False,
                "info": {"reward_source": "control_plane"},
                "tool_call": "move(right)",
            }
        ]

        trajectory.control_plane_summary = {
            "total_reward": 0.0,
            "termination_reason": "control_plane_signal",
            "final_step": 0,
            "control_plane_source": {"reward_source": "control_plane"},
        }

        # Test serialization
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            trajectory_dict = {
                "session_id": trajectory.session.session_id,
                "observations": trajectory.observations,
                "actions": trajectory.actions,
                "rewards": trajectory.rewards,
                "terminated": trajectory.terminated,
                "total_reward": trajectory.total_reward,
                "steps": trajectory.steps,
                "duration": trajectory.duration,
                "control_plane_steps": trajectory.control_plane_steps,
                "control_plane_summary": trajectory.control_plane_summary,
            }

            json.dump(trajectory_dict, f)
            f.flush()

            # Test deserialization
            with open(f.name, "r") as read_f:
                loaded_data = json.load(read_f)

                assert loaded_data["session_id"] == "test"
                assert len(loaded_data["control_plane_steps"]) == 1
                assert loaded_data["control_plane_summary"]["termination_reason"] == "control_plane_signal"

        # Clean up
        Path(f.name).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
