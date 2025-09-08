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

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import eval_protocol as ep

# Add examples directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "frozen_lake_mcp"))


from eval_protocol.mcp.execution.manager import ExecutionManager
from eval_protocol.mcp.session.manager import GeneralMCPVectorEnv
from eval_protocol.types import DatasetRow, MCPSession, MCPToolCall, Trajectory


class MockPolicy:
    """Mock policy for testing that returns predetermined actions."""

    def __init__(self, actions=None):
        self.actions = actions or ["right", "down", "right", "down"]
        self.step_count = 0
        self.model_id = "mock-model"

    async def __call__(self, tool_schema, env_index, conversation_history):
        """Return predetermined actions as tool calls."""
        if self.step_count < len(self.actions):
            action = self.actions[self.step_count]
        else:
            action = "right"  # Default action

        tool_calls = []
        tool_call = MCPToolCall(tool_name="lake_move", arguments={"action": action})
        tool_calls.append(tool_call)
        if self.step_count == 3:
            self.step_count += 1
            no_tool_call = MCPToolCall(tool_name="_no_tool_call", arguments={})
            return [no_tool_call], None, "stop"

        self.step_count += 1
        return tool_calls, None, None

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
        """Mock method for conversation tracking - adds proper OpenAI-format messages."""
        # Add assistant message with tool call
        conversation_history.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call.tool_call_id or f"call_{len(conversation_history)}",
                        "type": "function",
                        "function": {"name": tool_call.tool_name, "arguments": str(tool_call.arguments)},
                    }
                ],
            }
        )

        # Add tool response message
        conversation_history.append(
            {
                "role": "tool",
                "content": response,
                "tool_call_id": tool_call.tool_call_id or f"call_{len(conversation_history) - 1}",
                "control_plane_step": {
                    "step": env_index,
                    "reward": reward,
                    "terminated": done,
                    "info": info.get("control_plane", {}) if info else {},
                    "tool_calls": [f"{tool_call.tool_name}({tool_call.arguments})"],
                    "num_tool_calls": 1,
                },
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
            mock_env.tool_schemas = [{"name": "lake_move", "description": "Move in FrozenLake"}]

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
            semaphore = asyncio.Semaphore(1)  # Create semaphore for test
            tasks = self.execution_manager.execute_rollouts(mock_env, policy, semaphore, steps=10)
            evaluation_rows = []
            for task in tasks:
                row = await task
                evaluation_rows.append(row)

            # Validate results
            assert len(evaluation_rows) == 1, "Should have one evaluation row"
            evaluation_row = evaluation_rows[0]

            # Extract trajectory information from messages' control_plane_step data
            messages_with_control_plane = [
                msg for msg in evaluation_row.messages if msg.control_plane_step is not None
            ]
            steps = len(messages_with_control_plane)
            total_reward = sum(msg.control_plane_step["reward"] for msg in messages_with_control_plane)
            terminated = any(msg.control_plane_step["terminated"] for msg in messages_with_control_plane)

            # Validate basic trajectory structure
            assert steps == 3, f"Expected 3 steps, got {steps}"
            assert total_reward == 1.0, f"Expected reward 1.0, got {total_reward}"
            assert terminated == True, "Trajectory should be terminated"

            # Validate that data plane and control plane are properly separated in messages
            # Tool responses should only contain observations, rewards/termination are in control_plane_step
            for msg in evaluation_row.messages:
                if msg.role == "tool":
                    # Tool responses should only contain data plane information
                    content = msg.content or ""
                    # The content should not directly contain rewards or termination (they're in control_plane_step)
                    assert "reward" not in content.lower() or "reward_source" in content.lower(), (
                        "Tool response should not directly contain reward"
                    )

            # Validate control plane information from messages
            rewards = [msg.control_plane_step["reward"] for msg in messages_with_control_plane]
            assert rewards == [0.0, 0.0, 1.0], "Rewards should match control plane"

            # Validate enhanced control plane tracking via messages
            assert len(messages_with_control_plane) == 3, "Should have 3 messages with control plane steps"

            for i, msg in enumerate(messages_with_control_plane):
                cp_step = msg.control_plane_step
                assert "step" in cp_step, "Control plane step should have step number"
                assert "reward" in cp_step, "Control plane step should have reward"
                assert "terminated" in cp_step, "Control plane step should have terminated status"
                assert "info" in cp_step, "Control plane step should have control plane info"
                assert "tool_calls" in cp_step, "Control plane step should have tool calls"

            # Validate final step has termination
            final_msg = messages_with_control_plane[-1]
            final_cp_step = final_msg.control_plane_step
            assert final_cp_step["terminated"] == True, "Final step should be terminated"
            assert final_cp_step["reward"] == 1.0, "Final step should have correct reward"
            assert final_cp_step["termination_reason"] == "stop", "Should terminate via control plane"
            assert final_cp_step["step"] == 2, "Should record final step"

            # Validate policy interaction
            assert policy.step_count == 4, "Policy should have been called 4 times"

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
            usage={},
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
            # Add tool_schemas attribute expected by execute_rollouts
            mock_env.tool_schemas = [{"name": "move", "description": "Move"}]

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
            mock_format_user_prompt.return_value = "Test"

            # Execute rollout with control plane failure
            policy = MockPolicy(["right"])
            semaphore = asyncio.Semaphore(1)  # Create semaphore for test
            tasks = self.execution_manager.execute_rollouts(mock_env, policy, semaphore, steps=1)
            evaluation_rows = []
            for task in tasks:
                row = await task
                evaluation_rows.append(row)

            # Should still work, but without control plane info
            assert len(evaluation_rows) == 1
            evaluation_row = evaluation_rows[0]

            # Extract trajectory information from messages
            messages_with_control_plane = [
                msg for msg in evaluation_row.messages if msg.control_plane_step is not None
            ]
            steps = len(messages_with_control_plane)
            total_reward = sum(msg.control_plane_step["reward"] for msg in messages_with_control_plane)

            assert steps == 1
            assert total_reward == 0.0

            # Control plane steps should still be recorded (even if empty)
            assert len(messages_with_control_plane) == 1
            assert messages_with_control_plane[0].control_plane_step["info"] == {}

    @pytest.mark.asyncio
    async def test_rollout_creates_envs_from_url(self):
        """Ensure rollout can create environments automatically when given a URL."""

        dataset = [
            {
                "id": "row1",
                "system_prompt": "sys",
                "user_prompt_template": "tmpl",
                "environment_context": {"seed": 1},
            }
        ]

        policy = MockPolicy(["right"])

        with (
            patch("eval_protocol.mcp_env.make") as mock_make,
            patch("eval_protocol.mcp_env.ExecutionManager") as MockManager,
        ):
            mock_env = MagicMock()
            mock_make.return_value = mock_env

            manager_instance = MockManager.return_value

            # Mock execute_rollouts to return tasks and track calls
            call_args = []

            async def mock_task():
                return "ok"

            def mock_execute_rollouts(*args, **kwargs):
                call_args.append((args, kwargs))

                return [asyncio.create_task(mock_task())]

            manager_instance.execute_rollouts = mock_execute_rollouts

            result = []
            tasks = await ep.rollout(
                "http://localhost:1234/mcp/",
                policy,
                dataset=dataset,
                model_id="test_model",
                steps=5,
            )
            result.extend(tasks)

            mock_make.assert_called_once_with(
                "http://localhost:1234/mcp/",
                evaluation_rows=None,
                dataset=dataset,
                model_id="test_model",
            )

            # Verify execute_rollouts was called with correct arguments
            assert len(call_args) == 1, "execute_rollouts should be called once"
            args, kwargs = call_args[0]

            assert args[0] == mock_make.return_value, "First arg should be mock env"
            assert args[1] == policy, "Second arg should be policy"
            assert isinstance(kwargs.get("semaphore"), asyncio.Semaphore), "semaphore should be in kwargs"
            assert kwargs.get("steps") == 5, "steps should be in kwargs"

            assert result == ["ok"]

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
            usage={},
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
