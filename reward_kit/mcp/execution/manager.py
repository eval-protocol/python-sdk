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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
            openai_logger = lambda data: self._log_openai_entry(
                openai_format_log_file, data
            )

        logger.info(f"🧵 Starting {envs.n} rollouts with max {max_concurrent_rollouts} concurrent threads...")

        results = {}

        with ThreadPoolExecutor(max_workers=max_concurrent_rollouts, thread_name_prefix="rollout") as executor:
            futures = {
                executor.submit(
                    lambda idx=i: asyncio.run(
                        self._execute_rollout(
                            envs, policy, idx, steps, openai_logger, 
                            recording_mode, playback_mode, start_time
                        )
                    )
                ): i 
                for i in range(envs.n)
            }

            completed_count = 0
            for future in as_completed(futures):
                rollout_idx = futures[future]
                trajectory = future.result()

                results[rollout_idx] = trajectory

                completed_count += 1
        
        trajectories = [results[i] for i in range(envs.n)]
        
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

        logger.info(f"📊 Rollout complete: {successful}/{len(trajectories)} reached goal")
        logger.info(
            f"🎛️  Control plane terminations: {terminated_by_control_plane}/{len(trajectories)}"
        )
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
        start_time: float
    ) -> Trajectory:
        """
        Execute a single rollout for one environment (async version for thread execution).
        
        This method runs within a thread's event loop and handles all async operations.
        """
        session = envs.sessions[rollout_idx]
        
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
        )

        current_observation, tool_schema = await envs.reset(session)
        system_prompt = envs.dataset_rows[rollout_idx].system_prompt
        
        # Record initial observation
        trajectory.observations.append(current_observation)

        # Initialize conversation history for this thread
        user_prompt = envs.format_user_prompt(rollout_idx, current_observation)
        conversation_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        logger.info(f"🎯 Starting rollout {rollout_idx} in thread {threading.current_thread().name}")

        # Run rollout loop for this specific environment
        for step in range(steps):
            # Check if already terminated
            if trajectory.terminated:
                logger.debug(f"Rollout {rollout_idx} already terminated at step {step}")
                break

            # Check control plane termination status
            control_plane_state = await self._get_control_plane_status(session)
            if control_plane_state and control_plane_state.get("terminated", False):
                trajectory.terminated = True
                session.terminated = True
                logger.debug(f"Rollout {rollout_idx} terminated by control plane before step {step}")
                break

            # Format user prompt based on current observation
            user_prompt = envs.format_user_prompt(rollout_idx, current_observation)

            # Generate tool call for this environment using thread-local conversation history
            tool_call = await policy(tool_schema, rollout_idx, conversation_history)

            # Handle fallback case where no tool call is generated
            if tool_call is None:
                tool_call = MCPToolCall(
                    tool_name="_no_tool_call",
                    arguments={"reason": "no_tool_call_generated"},
                )

            # Execute tool call for this environment
            observation, reward, done, info = await envs.step(rollout_idx, tool_call)

            tool_response = envs.format_tool_response(observation)
            
            policy.add_tool_response(
                rollout_idx, tool_call, tool_response, conversation_history, reward, done, info
            )

            # Log conversation state for playback if in recording mode
            if recording_mode:
                policy.log_conversation_state_for_playback(rollout_idx, step, conversation_history)

            # Update trajectory with both data and control plane information
            trajectory.observations.append(observation)
            
            # Record action (tool call)
            action_str = f"{tool_call.tool_name}({tool_call.arguments})"
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
                session.terminated = True

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

                logger.info(f"🏁 Rollout {rollout_idx} terminated at step {step + 1} (reward: {trajectory.total_reward}) in thread {threading.current_thread().name}")
                break

            # Update current observation for next step
            current_observation = observation

            # Progress logging
            if step % 10 == 0:
                logger.debug(f"Rollout {rollout_idx} step {step}, reward: {trajectory.total_reward:.2f}")

        # Final completion logging with termination reason
        termination_reason = "max_steps_reached"
        if trajectory.terminated:
            termination_reason = "environment_terminated"
        
        logger.info(f"✅ Rollout {rollout_idx} completed: {trajectory.steps} steps, reward: {trajectory.total_reward:.2f}, reason: {termination_reason} in thread {threading.current_thread().name}")

        return trajectory

    def _add_tool_response(
        self,
        conversation_history: List[Dict[str, Any]], 
        tool_call: MCPToolCall,
        tool_response: Union[str, List[Dict[str, Any]]],
        reward: float = 0.0,
        terminated: bool = False,
        info: Optional[Dict[str, Any]] = None,
    ):
        """
        Add tool call and response to conversation history with control plane metadata.
        
        Extracted from policy.py to use directly in thread-local conversation management.
        """
        # Find the most recent assistant message with tool calls to get the correct call_id
        call_id = None
        for i in range(len(conversation_history) - 1, -1, -1):
            if (
                conversation_history[i]["role"] == "assistant"
                and "tool_calls" in conversation_history[i]
            ):
                # Find the tool call that matches our tool_name
                for tc in conversation_history[i]["tool_calls"]:
                    if tc["function"]["name"] == tool_call.tool_name:
                        call_id = tc["id"]
                        break
                if call_id:
                    break

        tool_message = {
            "role": "tool",
            "tool_call_id": call_id,
            "content": tool_response,
        }

        # Add control plane metadata if provided
        if reward != 0.0 or terminated or info:
            tool_message["metadata"] = {
                "reward": reward,
                "terminated": terminated,
                "info": info or {},
            }

        conversation_history.append(tool_message)

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