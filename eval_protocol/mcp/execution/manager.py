"""
MCP Execution Management

Unified class that handles both session management and rollout execution.
Combines the functionality of SessionManager and RolloutManager.
"""

import asyncio
from dataclasses import dataclass
import json
import logging
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from ..client.connection import MCPConnectionManager
from ..types import MCPSession, MCPToolCall, Trajectory, TerminationReason, LLMUsageStats

from tau2.user.user_simulator import UserSimulator
from tau2.data_model.message import AssistantMessage, UserMessage

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
        tasks = [self.connection_manager.initialize_session(session) for session in sessions]
        await asyncio.gather(*tasks)

    async def close_sessions(self, sessions: List[MCPSession]) -> None:
        """
        Close multiple MCP sessions in parallel.

        Args:
            sessions: List of MCPSessions to close
        """
        tasks = [asyncio.create_task(self.connection_manager.close_session(session)) for session in sessions]

        if tasks:
            try:
                # Wait for all close operations to complete
                await asyncio.gather(*tasks, return_exceptions=True)
            except asyncio.CancelledError:
                # Handle cancellation gracefully (especially important for Python 3.12)
                logger.debug("Close operation was cancelled, but sessions are marked as closed")

    async def execute_rollouts(
        self,
        envs: "GeneralMCPVectorEnv",
        policy: Union["LLMBasePolicy", Callable],
        steps: int = 512,
        openai_format_log_file: Optional[str] = None,
        max_concurrent_rollouts: int = 8,
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
            max_concurrent_rollouts: Maximum number of concurrent threads to run

        Environment Variable Control:
            EP_PLAYBACK_FILE: Controls record/playback mode
            - Not set: Normal live mode
            - Set but file doesn't exist: Record mode (file will be created)
            - Set and file exists: Playback mode (uses recorded data)

        Returns:
            List of Trajectory objects with complete rollout data
        """
        start_time = time.time()

        # Check for record/playback mode
        playback_file = os.environ.get("EP_PLAYBACK_FILE")
        recording_mode = bool(playback_file and not os.path.exists(playback_file))
        playback_mode = bool(playback_file and os.path.exists(playback_file))

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
            openai_logger = lambda data: self._log_openai_entry(openai_format_log_file, data)

        logger.info(f"🧵 Starting {envs.n} rollouts with max {max_concurrent_rollouts} concurrent threads...")

        results = {}

        semaphore = asyncio.Semaphore(max_concurrent_rollouts)

        async def _execute_with_semaphore(idx):
            async with semaphore:
                return await self._execute_rollout(
                    envs, policy, idx, steps, openai_logger, recording_mode, playback_mode, start_time
                )

        tasks = [_execute_with_semaphore(i) for i in range(envs.n)]
        trajectories = await asyncio.gather(*tasks)

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
            if traj.control_plane_summary.get("termination_reason") == "control_plane_signal"
        )

        logger.info(f"📊 Rollout complete: {successful}/{len(trajectories)} reached goal")
        logger.info(f"🎛️  Control plane terminations: {terminated_by_control_plane}/{len(trajectories)}")
        logger.info(f"⏱️  Total duration: {total_duration:.2f}s")
        logger.info(f"🧵 Used {max_concurrent_rollouts} concurrent threads")

        # Print log file locations if created
        if openai_format_log_file:
            logger.info(f"💬 OpenAI format log: {openai_format_log_file}")
        if recording_mode:
            logger.info(f"📝 Recorded trajectory: {playback_file}")
            # Add note about control plane separation
            logger.info(f"🎛️  Trajectories include control plane separation")

        return trajectories

    async def _execute_rollout(
        self,
        envs: "GeneralMCPVectorEnv",
        policy: Union["LLMBasePolicy", Callable],
        rollout_idx: int,
        steps: int,
        openai_logger: Optional[Callable],
        recording_mode: bool,
        playback_mode: bool,
        start_time: float,
    ) -> Trajectory:
        """
        Execute a single rollout for one environment (async version for thread execution).

        This method runs within a thread's event loop and handles all async operations.
        """
        session = envs.sessions[rollout_idx]
        dataset_row = envs.dataset_rows[rollout_idx]

        # Initialize trajectory
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
            llm_usage_summary={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        )

        current_observation, tool_schema = await envs.reset(session)
        system_prompt = dataset_row.system_prompt

        # Record initial observation
        trajectory.observations.append(current_observation)

        # Create user simulator for this rollout if configured in dataset
        user_simulator = None
        user_simulator_state = None

        # If user simulation is enabled, initial message is from the simulated user
        if dataset_row.user_simulation and dataset_row.user_simulation.get("enabled", False):
            user_simulator = UserSimulator(
                instructions=dataset_row.user_simulation.get("system_prompt"),
                llm=dataset_row.user_simulation.get("llm", "gpt-4.1"),
                llm_args=dataset_row.user_simulation.get("llm_args", {"temperature": 0.7}),
            )

            # Get initial messages in tau2-bench format for user simulator
            user_simulator_state = user_simulator.get_init_state()
            user_message, user_simulator_state = user_simulator.generate_next_message(
                AssistantMessage(role="assistant", content="Hi! How can I help you today?"),
                user_simulator_state,
            )
            current_observation = user_message.content if user_message.content else ""

        user_prompt = envs.format_user_prompt(rollout_idx, current_observation)
        conversation_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        usage_stats_list: List[LLMUsageStats] = []

        logger.info(f"🎯 Starting rollout {rollout_idx} in thread {threading.current_thread().name}")

        # Run rollout loop for this specific environment
        step = 0
        rollout_end = False

        while step < steps and not trajectory.terminated:
            turn_completed = False
            info = {}
            reward = 0.0
            observation = current_observation
            tool_calls = []

            if user_simulator and user_simulator_state:
                # Get user simulator messages and find the last assistant message
                user_simulator_messages = self._get_user_simulator_messages(conversation_history)

                # Last message was agent, simulated user response
                if user_simulator_messages and isinstance(user_simulator_messages[-1], AssistantMessage):
                    # Generate user response using the simulator
                    user_message, user_simulator_state = user_simulator.generate_next_message(
                        user_simulator_messages[-1], user_simulator_state
                    )
                    user_content = user_message.content if user_message.content else ""

                    user_prompt = envs.format_user_prompt(rollout_idx, user_content)
                    conversation_history.append({"role": "user", "content": user_prompt})

                    # Check if user simulator signaled termination
                    if UserSimulator.is_stop(user_message):
                        trajectory.terminated = True
                        trajectory.termination_reason = TerminationReason.USER_STOP

            # In each turn: keep looping until assistant is ready to provide final response
            while not turn_completed and not trajectory.terminated:
                tool_calls, usage_stats = await policy(tool_schema, rollout_idx, conversation_history)

                # If no tool call is generated, turn is finished
                if len(tool_calls) == 1:
                    # No tool calls means the policy is ready to provide final response on this turn
                    if tool_calls[0].tool_name == "_no_tool_call":
                        trajectory.terminated = True
                        break
                    elif tool_calls[0].tool_name == "_playback_terminate":
                        trajectory.terminated = True
                        break

                # Execute each tool call sequentially
                for tool_call in tool_calls:

                    # Execute tool call for this environment
                    observation, reward, rollout_end, info = await envs.step(rollout_idx, tool_call)

                    tool_response = envs.format_tool_response(observation)

                    policy.add_tool_response(
                        rollout_idx,
                        tool_call,
                        tool_response,
                        conversation_history,
                        reward,
                        rollout_end,
                        info,
                    )

                    # Update trajectory with both data and control plane information
                    trajectory.observations.append(observation)

                    # Record action (tool call)
                    action_str = f"{tool_call.tool_name}({tool_call.arguments})"
                    trajectory.actions.append(action_str)

                    # Record control plane (reward/termination)
                    trajectory.rewards.append(reward)
                    trajectory.total_reward += reward

                    # Non-user simulator step counter: each tool call is a step
                    if user_simulator is None:
                        step += 1
                        trajectory.steps = step

                        control_plane_step = {
                            "step": step - 1,
                            "reward": reward,
                            "terminated": rollout_end,
                            "info": info.get("control_plane", {}),
                            "tool_calls": [f"{tool_call.tool_name}({tool_call.arguments})"],
                            "num_tool_calls": 1,
                        }
                        trajectory.control_plane_steps.append(control_plane_step)

                        # Log conversation state for playback if in recording mode
                        if recording_mode:
                            policy.log_conversation_state_for_playback(rollout_idx, step - 1, conversation_history)

                    if rollout_end:
                        trajectory.terminated = True
                        trajectory.termination_reason = TerminationReason.CONTROL_PLANE_SIGNAL
                        break
                    elif step >= steps:
                        trajectory.terminated = True
                        trajectory.termination_reason = TerminationReason.MAX_STEPS
                        break

                # Update current observation for potential next turn
                if observation is not None:
                    current_observation = observation

                # calc llm usage stats happened in this turn if there is aany
                if usage_stats:
                    usage_stats_list.append(usage_stats)

            # With user simulator, increment step after an entire conversation step
            if user_simulator is not None:
                step += 1
                trajectory.steps = step

                # Enhanced trajectory recording with control plane info
                # Create summary of all tool calls executed in this step
                tool_calls_summary = [f"{tc.tool_name}({tc.arguments})" for tc in tool_calls]

                control_plane_step = {
                    "step": step - 1,
                    "reward": reward,
                    "terminated": rollout_end,
                    "info": info.get("control_plane", {}),
                    "tool_calls": tool_calls_summary,
                    "num_tool_calls": len(tool_calls),
                }
                trajectory.control_plane_steps.append(control_plane_step)

                # Log conversation state for playback if in recording mode
                if recording_mode:
                    policy.log_conversation_state_for_playback(rollout_idx, step - 1, conversation_history)

            # Use control plane information for termination decision
            if rollout_end:
                trajectory.terminated = True
                trajectory.termination_reason = TerminationReason.CONTROL_PLANE_SIGNAL

                # Add final control plane summary
                trajectory.control_plane_summary.update(
                    {
                        "total_reward": trajectory.total_reward,
                        "termination_reason": trajectory.termination_reason,
                        "final_step": step - 1,
                        "control_plane_source": info.get("control_plane", {}),
                    }
                )

                # Log final OpenAI conversation for terminated trajectories only
                if openai_logger:
                    if conversation_history and len(conversation_history) > 0:
                        openai_logger(
                            {
                                "messages": conversation_history,
                                "metadata": {
                                    "session_id": session.session_id,
                                    "seed": session.seed,
                                    "total_steps": trajectory.steps,
                                    "total_reward": trajectory.total_reward,
                                    "terminated": True,
                                    "success": reward > 0,
                                    "control_plane_summary": trajectory.control_plane_summary,
                                },
                            }
                        )

                logger.info(
                    f"🏁 Rollout {rollout_idx} terminated at step {step} (reward: {trajectory.total_reward}) in thread {threading.current_thread().name}"
                )
                break

            # Progress logging
            if step % 10 == 0:
                logger.debug(f"Rollout {rollout_idx} step {step}, reward: {trajectory.total_reward:.2f}")

        # Set termination reason if not already set (e.g., due to step limit)
        if not trajectory.termination_reason and step >= steps:
            trajectory.termination_reason = TerminationReason.MAX_STEPS

        trajectory.conversation_history = conversation_history

        for usage_stats in usage_stats_list:
            trajectory.llm_usage_summary["prompt_tokens"] += usage_stats.prompt_tokens
            trajectory.llm_usage_summary["completion_tokens"] += usage_stats.completion_tokens
            trajectory.llm_usage_summary["total_tokens"] += usage_stats.total_tokens

        logger.info(
            f"✅ Rollout {rollout_idx} completed: {trajectory.steps} steps, reward: {trajectory.total_reward:.2f}, termination: {trajectory.termination_reason}, in thread {threading.current_thread().name}"
        )
        return trajectory

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
            logger.debug(f"Control plane query timed out for session {session.session_id[:16]}")
            return None
        except Exception as e:
            logger.debug(f"Control plane query failed for session {session.session_id[:16]}: {e}")
            return None

    def _log_openai_entry(self, log_file: str, data: Dict[str, Any]):
        """Helper function to log OpenAI format entries."""
        with open(log_file, "a") as f:
            f.write(json.dumps(data) + "\n")

    def _get_user_simulator_messages(self, conversation_history: List[Dict[str, Any]]) -> List:
        """
        Filter conversation history for user simulator and convert to tau2-bench format.
        """
        tau2_messages = []

        for message in conversation_history:
            role = message.get("role")
            content = message.get("content", "")

            if role == "assistant":
                if "tool_calls" not in message or not message.get("tool_calls"):
                    tau2_messages.append(AssistantMessage(role="assistant", content=content))

            elif role == "user":
                tau2_messages.append(UserMessage(role="user", content=content))

        return tau2_messages
