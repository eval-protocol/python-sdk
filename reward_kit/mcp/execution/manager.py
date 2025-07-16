"""
MCP Execution Management

Unified class that handles both session management and rollout execution.
Combines the functionality of SessionManager and RolloutManager.
"""

import asyncio
import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from ..client.connection import MCPConnectionManager
from ..types import MCPSession, MCPToolCall, Trajectory

if TYPE_CHECKING:
    from ..session.manager import GeneralMCPVectorEnv
    from .policy import LLMBasePolicy

logger = logging.getLogger(__name__)


class ExecutionManager:
    """
    Unified manager that handles both MCP session lifecycle and rollout execution.
    
    Combines the functionality of SessionManager and RolloutManager for better
    organization and reduced complexity.
    """

    def __init__(self):
        """Initialize the execution manager."""
        self.connection_manager = MCPConnectionManager()

    async def initialize_sessions(self, sessions: List[MCPSession]) -> None:
        """
        Initialize multiple MCP sessions in parallel.

        Args:
            sessions: List of MCPSessions to initialize
        """
        tasks = [
            self.connection_manager.initialize_session(session) for session in sessions
        ]
        await asyncio.gather(*tasks)

    async def close_sessions(self, sessions: List[MCPSession]) -> None:
        """
        Close multiple MCP sessions in parallel.

        Args:
            sessions: List of MCPSessions to close
        """
        tasks = [
            asyncio.create_task(self.connection_manager.close_session(session))
            for session in sessions
        ]

        if tasks:
            try:
                # Wait for all close operations to complete
                await asyncio.gather(*tasks, return_exceptions=True)
            except asyncio.CancelledError:
                # Handle cancellation gracefully (especially important for Python 3.12)
                logger.debug(
                    "Close operation was cancelled, but sessions are marked as closed"
                )

    async def execute_rollout(
        self,
        envs: "GeneralMCPVectorEnv",
        policy: Union["LLMBasePolicy", Callable],
        steps: int = 512,
        openai_format_log_file: Optional[str] = None,
    ) -> List[Trajectory]:
        """
        Execute general rollouts using tool calling interface with automatic record/playback.

        This works with ANY MCP environment because:
        1. Policy receives tool schemas and makes tool calls
        2. Environment prompts come from dataset
        3. No hardcoded environment logic

        Args:
            envs: GeneralMCPVectorEnv instance
            policy: Policy that takes tool schemas, observations, prompts and returns tool calls
            steps: Maximum steps per rollout
            openai_format_log_file: Optional file to log clean OpenAI format for terminated trajectories only

        Environment Variable Control:
            REWARD_KIT_PLAYBACK_FILE: Controls record/playback mode
            - Not set: Normal live mode
            - Set but file doesn't exist: Record mode (file will be created)
            - Set and file exists: Playback mode (uses recorded data)

        Returns:
            List of Trajectory objects with complete rollout data
        """
        start_time = time.time()

        # Check for record/playback mode
        playback_file = os.environ.get("REWARD_KIT_PLAYBACK_FILE")
        recording_mode = playback_file and not os.path.exists(playback_file)
        playback_mode = playback_file and os.path.exists(playback_file)

        if recording_mode:
            logger.info(f"📝 Recording mode: Will record to {playback_file}")
        elif playback_mode:
            logger.info(f"🎬 Playback mode: Using recorded data from {playback_file}")
        else:
            logger.info(f"🚀 Live mode: No recording/playback")

        # Initialize OpenAI format logging for terminated trajectories only
        openai_logger = None
        if openai_format_log_file:
            # Clear the file at start
            with open(openai_format_log_file, "w") as f:
                pass
            openai_logger = lambda data: self._log_openai_entry(
                openai_format_log_file, data
            )

        # Initialize trajectories
        trajectories = []
        for session in envs.sessions:
            trajectories.append(
                Trajectory(
                    session=session,
                    observations=[],
                    actions=[],
                    rewards=[],
                    terminated=False,
                    total_reward=0.0,
                    steps=0,
                    duration=0.0,
                )
            )

        # Reset environments and get initial state with tool discovery
        print(f"🔄 Resetting {envs.n} MCP environments...")
        current_observations, tool_schemas, system_prompts = await envs.reset()

        # Record initial observations
        for trajectory, obs in zip(trajectories, current_observations):
            trajectory.observations.append(obs)

        print(f"✅ Starting rollouts with {envs.n} environments for {steps} steps...")

        # Run rollout loop with tool calling
        for step in range(steps):
            step_start_time = time.time()

            # Early termination check - prevent tool call generation for already terminated environments
            if all(traj.terminated for traj in trajectories):
                print(
                    f"🏁 All environments already terminated before step {step} (control plane signals)"
                )
                break

            # Check control plane termination status for each environment BEFORE generating tool calls
            # This ensures we don't generate tool calls for environments that are already terminated
            for i, session in enumerate(envs.sessions):
                if not trajectories[i].terminated:
                    # Query control plane status directly from the session
                    control_plane_state = await self._get_control_plane_status(session)
                    if control_plane_state and control_plane_state.get(
                        "terminated", False
                    ):
                        # Environment is terminated according to control plane
                        trajectories[i].terminated = True
                        session.terminated = True
                        logger.debug(
                            f"Environment {i} terminated by control plane before step {step}"
                        )

            # Format user prompts based on current observations (callback pattern)
            user_prompts = envs.format_user_prompts(current_observations)

            # Filter out terminated environments when generating tool calls
            active_indices = [
                i for i, traj in enumerate(trajectories) if not traj.terminated
            ]
            active_tool_schemas = [tool_schemas[i] for i in active_indices]
            active_observations = [current_observations[i] for i in active_indices]
            active_system_prompts = [system_prompts[i] for i in active_indices]
            active_user_prompts = [user_prompts[i] for i in active_indices]

            # Generate tool calls only for active environments
            if active_indices:
                active_tool_calls = await policy(
                    active_tool_schemas,
                    active_observations,
                    active_system_prompts,
                    active_user_prompts,
                )
            else:
                active_tool_calls = []

            # Create full tool_calls list with no-op calls for terminated environments
            tool_calls = []
            active_call_idx = 0
            for i, traj in enumerate(trajectories):
                if traj.terminated:
                    # Use a no-op tool call for terminated environments
                    no_op_call = MCPToolCall(
                        tool_name="_no_tool_call",
                        arguments={"reason": "environment_already_terminated"},
                    )
                    tool_calls.append(no_op_call)
                else:
                    tool_calls.append(active_tool_calls[active_call_idx])
                    active_call_idx += 1

            # Execute tool calls via MCP protocol (now with control plane separation)
            observations, rewards, dones, infos = await envs.step(tool_calls)

            # Update conversation histories with tool responses (for proper OpenAI trajectories)
            if hasattr(policy, "add_tool_response"):
                for i, (tool_call, obs, reward, done, info) in enumerate(
                    zip(tool_calls, observations, rewards, dones, infos)
                ):
                    # Skip adding tool response for no-op calls (terminated environments)
                    if tool_call.tool_name == "_no_tool_call":
                        logger.debug(
                            f"Env {i}: Skipping tool response recording for no-op call"
                        )
                        continue

                    # Convert observation to tool response format
                    tool_response = (
                        json.dumps(obs) if isinstance(obs, dict) else str(obs)
                    )

                    # Check if policy supports control plane metadata
                    import inspect

                    sig = inspect.signature(policy.add_tool_response)
                    if (
                        len(sig.parameters) > 3
                    ):  # More than (env_index, tool_call, tool_response)
                        # Pass control plane information for policies that support it
                        policy.add_tool_response(
                            i, tool_call, tool_response, reward, done, info
                        )
                    else:
                        # Fallback for older policies
                        policy.add_tool_response(i, tool_call, tool_response)

                    # Log conversation state for playback if in recording mode
                    if recording_mode and hasattr(
                        policy, "log_conversation_state_for_playback"
                    ):
                        policy.log_conversation_state_for_playback(i, step)

            # Update trajectories with both data and control plane information
            for i, (trajectory, obs, reward, done, info) in enumerate(
                zip(trajectories, observations, rewards, dones, infos)
            ):
                if not trajectory.terminated:
                    # Record data plane (observation)
                    trajectory.observations.append(obs)

                    # Record action (tool call)
                    action_str = f"{tool_calls[i].tool_name}({tool_calls[i].arguments})"
                    trajectory.actions.append(action_str)

                    # Record control plane (reward/termination)
                    trajectory.rewards.append(reward)
                    trajectory.total_reward += reward
                    trajectory.steps += 1

                    # Enhanced trajectory recording with control plane info
                    if not hasattr(trajectory, "control_plane_steps"):
                        trajectory.control_plane_steps = []

                    control_plane_step = {
                        "step": step,
                        "reward": reward,
                        "terminated": done,
                        "info": info.get("control_plane", {}),
                        "tool_call": action_str,
                    }
                    trajectory.control_plane_steps.append(control_plane_step)

                    # Use control plane information for termination decision
                    if done:
                        trajectory.terminated = True
                        # Also update session termination status to ensure synchronization
                        envs.sessions[i].terminated = True

                        # Add final control plane summary
                        if not hasattr(trajectory, "control_plane_summary"):
                            trajectory.control_plane_summary = {}

                        trajectory.control_plane_summary.update(
                            {
                                "total_reward": trajectory.total_reward,
                                "termination_reason": "control_plane_signal",
                                "final_step": step,
                                "control_plane_source": info.get("control_plane", {}),
                            }
                        )

                        # Log final OpenAI conversation for terminated trajectories only
                        if openai_logger and hasattr(policy, "conversation_histories"):
                            conversation = policy.conversation_histories.get(i, [])
                            if conversation:  # Only log if we have a conversation
                                openai_logger(
                                    {
                                        "messages": conversation,
                                        "metadata": {
                                            "session_id": envs.sessions[i].session_id,
                                            "seed": envs.sessions[i].seed,
                                            "total_steps": trajectory.steps,
                                            "total_reward": trajectory.total_reward,
                                            "terminated": True,
                                            "success": reward > 0,
                                            "control_plane_summary": trajectory.control_plane_summary,
                                        },
                                    }
                                )

            # Update current observations for next step
            current_observations = observations

            # Check if all environments are done (using control plane termination)
            if all(traj.terminated for traj in trajectories):
                print(
                    f"🏁 All environments terminated at step {step + 1} (control plane signals)"
                )
                break

            # Progress logging with control plane info
            active_envs = sum(1 for traj in trajectories if not traj.terminated)
            if step % 10 == 0 and active_envs > 0:
                avg_reward = sum(traj.total_reward for traj in trajectories) / len(
                    trajectories
                )
                print(
                    f"📊 Step {step}: {active_envs}/{len(trajectories)} active, avg reward: {avg_reward:.2f}"
                )

        # Calculate durations
        total_duration = time.time() - start_time
        for trajectory in trajectories:
            trajectory.duration = total_duration

        # Clean up
        await envs.close()

        # Enhanced reporting with control plane info
        successful = sum(1 for traj in trajectories if traj.total_reward > 0)
        terminated_by_control_plane = sum(
            1
            for traj in trajectories
            if hasattr(traj, "control_plane_summary")
            and traj.control_plane_summary.get("termination_reason")
            == "control_plane_signal"
        )

        print(f"📊 Rollout complete: {successful}/{len(trajectories)} reached goal")
        print(
            f"🎛️  Control plane terminations: {terminated_by_control_plane}/{len(trajectories)}"
        )
        print(f"⏱️  Total duration: {total_duration:.2f}s")

        # Print log file locations if created
        if openai_format_log_file:
            print(f"💬 OpenAI format log: {openai_format_log_file}")
        if recording_mode:
            print(f"📝 Recorded trajectory: {playback_file}")
            # Add note about control plane separation
            print(f"🎛️  Trajectories include control plane separation")

        return trajectories

    async def _get_control_plane_status(self, session) -> Optional[Dict[str, Any]]:
        """
        Query the control plane status endpoint directly for a session.

        Args:
            session: MCP session object

        Returns:
            Control plane status dictionary or None if query fails
        """
        try:
            import httpx

            # Extract base URL and session ID
            base_url = session.base_url.rstrip("/mcp").rstrip("/")
            session_id = session.session_id

            if not session_id:
                logger.debug("Control plane query failed: No session ID")
                return None

            headers = {"mcp-session-id": session_id}

            # Query status endpoint
            async with httpx.AsyncClient(timeout=2.0) as client:
                status_response = await client.get(
                    f"{base_url}/control/status",
                    headers=headers,
                    timeout=2.0,  # Short timeout for performance
                )

                if status_response.status_code == 200:
                    status_data = status_response.json()
                    return status_data
                else:
                    logger.debug(
                        f"Control plane endpoint returned {status_response.status_code} for session {session_id[:16]}"
                    )
                    return None

        except asyncio.TimeoutError:
            logger.debug(
                f"Control plane query timed out for session {session.session_id[:16]}"
            )
            return None
        except Exception as e:
            logger.debug(
                f"Control plane query failed for session {session.session_id[:16]}: {e}"
            )
            return None

    def _log_openai_entry(self, log_file: str, data: Dict[str, Any]):
        """Helper function to log OpenAI format entries."""
        with open(log_file, "a") as f:
            f.write(json.dumps(data) + "\n") 