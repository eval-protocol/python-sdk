"""
MCP Execution Management

Unified class that handles both session management and rollout execution.
Combines the functionality of SessionManager and RolloutManager.
"""

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union, cast

import anyio
from openai.types import CompletionUsage

from ...models import EvaluationRow, InputMetadata, Message, Status
from ...types import TerminationReason, Trajectory, NonSkippableException

if TYPE_CHECKING:
    from ..session.manager import GeneralMCPVectorEnv
    from .policy import LLMBasePolicy

logger = logging.getLogger(__name__)


class ExecutionManager:
    """
    Manage rollout for MCP environments.
    """

    def execute_rollouts(
        self,
        envs: "GeneralMCPVectorEnv",
        policy: Union["LLMBasePolicy", Callable],
        semaphore: asyncio.Semaphore,
        steps: int = 512,
        openai_format_log_file: Optional[str] = None,
        evaluation_rows: Optional[List[EvaluationRow]] = None,
    ) -> List[asyncio.Task[EvaluationRow]]:
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
            semaphore: Semaphore to control concurrent rollout execution

        Environment Variable Control:
            EP_PLAYBACK_FILE: Controls record/playback mode
            - Not set: Normal live mode
            - Set but file doesn't exist: Record mode (file will be created)
            - Set and file exists: Playback mode (uses recorded data)

        Returns:
            List of asyncio.Task objects for external handling
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
            logger.info("🚀 Live mode: No recording/playback")

        # Initialize OpenAI format logging for terminated trajectories only
        openai_logger = None
        if openai_format_log_file:
            # Clear the file at start
            with open(openai_format_log_file, "w") as f:
                pass
            openai_logger = lambda data: self._log_openai_entry(openai_format_log_file, data)

        logger.info(f"🧵 Starting {envs.n} rollouts with max {semaphore._value} concurrent threads...")

        if evaluation_rows is None:
            evaluation_rows = [EvaluationRow(messages=[], input_metadata=InputMetadata()) for _ in range(envs.n)]

        shared_tool_schema = envs.tool_schemas

        async def _execute_with_semaphore(idx):
            async with semaphore:
                evaluation_row: EvaluationRow = evaluation_rows[idx]
                row_start_time = time.perf_counter()

                trajectory = await self._execute_rollout(
                    envs, policy, idx, steps, openai_logger, recording_mode, playback_mode, start_time, evaluation_row
                )

                # Handle multimodal content by extracting text from complex content structures
                messages = []
                for msg in trajectory.conversation_history:
                    # Create a copy to avoid modifying the original
                    msg_dict = dict(msg)

                    # Handle multimodal content (list of content blocks) by extracting text
                    if isinstance(msg_dict.get("content"), list):
                        text_content = None
                        for content_block in msg_dict["content"]:
                            if isinstance(content_block, dict) and content_block.get("type") == "text":
                                text_content = content_block.get("text")
                                break
                        msg_dict["content"] = text_content or ""

                    messages.append(Message.model_validate(msg_dict))

                evaluation_row.messages = messages
                evaluation_row.tools = shared_tool_schema
                evaluation_row.execution_metadata.usage = CompletionUsage(
                    prompt_tokens=trajectory.usage.get("prompt_tokens", 0),
                    completion_tokens=trajectory.usage.get("completion_tokens", 0),
                    total_tokens=trajectory.usage.get("total_tokens", 0),
                )
                evaluation_row.input_metadata.completion_params = {
                    "model": policy.model_id,
                    "temperature": getattr(policy, "temperature", None),
                    "max_tokens": getattr(policy, "max_tokens", None),
                    "max_tool_calls": getattr(policy, "max_tools_per_turn", None),
                }

                if trajectory.terminated:
                    extra_info = None
                    if trajectory.control_plane_summary.get("error_message"):
                        extra_info = {"error_message": trajectory.control_plane_summary.get("error_message")}
                    # Convert string termination reason to TerminationReason enum if needed
                    term_reason = (
                        trajectory.termination_reason
                        if isinstance(trajectory.termination_reason, TerminationReason)
                        else TerminationReason.from_str(str(trajectory.termination_reason))
                    )
                    evaluation_row.rollout_status = Status.rollout_finished(
                        termination_reason=term_reason, extra_info=extra_info
                    )
                else:
                    evaluation_row.rollout_status = Status.rollout_running()

                evaluation_row.execution_metadata.duration_seconds = time.perf_counter() - row_start_time

                return evaluation_row

        # Create all tasks
        tasks = [asyncio.create_task(_execute_with_semaphore(i)) for i in range(envs.n)]
        return tasks

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
        evaluation_row: Optional[EvaluationRow] = None,
    ) -> Trajectory:
        """
        Execute a single rollout for one environment (async version for thread execution).

        This method runs within a thread's event loop and handles all async operations.
        """
        session = envs.sessions[rollout_idx]
        dataset_row = envs.dataset_rows[rollout_idx]

        # Helper function to sync conversation history to evaluation_row.messages
        def update_evaluation_row_messages():
            if evaluation_row:

                def extract_text_content(msg_dict):
                    msg_copy = dict(msg_dict)
                    if isinstance(msg_copy.get("content"), list):
                        for content_block in msg_copy["content"]:
                            if isinstance(content_block, dict) and content_block.get("type") == "text":
                                msg_copy["content"] = content_block.get("text", "")
                                break
                        else:
                            msg_copy["content"] = ""
                    return msg_copy

                evaluation_row.messages = [
                    Message.model_validate(extract_text_content(msg)) for msg in trajectory.conversation_history
                ]

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
            usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        )
        failure_reason = None
        try:
            current_observation, tool_schema = await envs.reset(session)
            system_prompt = dataset_row.system_prompt

            # Record initial observation
            trajectory.observations.append(current_observation)

            # Create user simulator for this rollout if configured in dataset
            user_simulator = None
            user_simulator_state = None

            # If user simulation is enabled, initial message is from the simulated user
            if dataset_row.user_simulation and dataset_row.user_simulation.get("enabled", False):
                # Lazy import vendor.tau2 - only load when user simulation is actually used
                from vendor.tau2.data_model.message import AssistantMessage, UserMessage
                from vendor.tau2.user.user_simulator import UserSimulator

                user_simulator = UserSimulator(
                    instructions=dataset_row.user_simulation.get("system_prompt"),
                    llm=dataset_row.user_simulation.get("llm", "gpt-4.1"),
                    llm_args=dataset_row.user_simulation.get("llm_args", {"temperature": 0.0}),
                )

                # Get initial messages in tau2-bench format for user simulator
                user_simulator_state = user_simulator.get_init_state()
                # Generate initial user response by prompting the simulator with a user role message
                user_message, user_simulator_state = await user_simulator.generate_next_message(
                    UserMessage(role="user", content=""),
                    user_simulator_state,
                )
                current_observation = user_message.content if user_message.content else ""

            user_prompt = envs.format_user_prompt(rollout_idx, current_observation)
            trajectory.conversation_history = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            update_evaluation_row_messages()

            logger.info(f"🎯 Starting rollout {rollout_idx} in thread {threading.current_thread().name}")

            # Run rollout loop for this specific environment
            step = 0
            env_end = False  # if the env indicates the rollout reaches the goal

            while step < steps and not trajectory.terminated:
                turn_completed = False
                info = {}
                reward = 0.0
                observation = current_observation
                tool_calls = []

                if user_simulator and user_simulator_state:
                    # Get user simulator messages and find the last assistant message
                    user_simulator_messages = self._get_user_simulator_messages(trajectory.conversation_history)

                    # Last message was agent, simulated user response
                    if user_simulator_messages and isinstance(user_simulator_messages[-1], AssistantMessage):
                        # Generate user response using the simulator
                        # Pass the assistant message content to drive the simulated user's next response
                        last_assistant = user_simulator_messages[-1]
                        # Convert last assistant message into a valid user input message for simulator
                        from vendor.tau2.data_model.message import UserMessage as TauUserMessage

                        converted_user_prompt = (
                            last_assistant.content if getattr(last_assistant, "content", None) else ""
                        )
                        converted_message = TauUserMessage(role="user", content=converted_user_prompt)
                        user_message, user_simulator_state = await user_simulator.generate_next_message(
                            converted_message,
                            user_simulator_state,
                        )
                        user_content = user_message.content if user_message.content else ""

                        user_prompt = envs.format_user_prompt(rollout_idx, user_content)
                        trajectory.conversation_history.append({"role": "user", "content": user_prompt})
                        update_evaluation_row_messages()

                        # Check if user simulator signaled termination
                        if UserSimulator.is_stop(user_message):
                            trajectory.terminated = True
                            trajectory.termination_reason = TerminationReason.USER_STOP

                # In each turn: keep looping until assistant is ready to provide final response
                while not turn_completed and not trajectory.terminated:
                    tool_calls, usage_stats, finish_reason = await policy(
                        tool_schema, rollout_idx, trajectory.conversation_history
                    )
                    update_evaluation_row_messages()

                    # Update LLM usage stats if available; support both dict-like and attribute access
                    if usage_stats:
                        try:
                            prompt_tokens = (
                                usage_stats.get("prompt_tokens")
                                if isinstance(usage_stats, dict)
                                else usage_stats.prompt_tokens
                            )
                            completion_tokens = (
                                usage_stats.get("completion_tokens")
                                if isinstance(usage_stats, dict)
                                else usage_stats.completion_tokens
                            )
                            total_tokens = (
                                usage_stats.get("total_tokens")
                                if isinstance(usage_stats, dict)
                                else usage_stats.total_tokens
                            )
                            if isinstance(prompt_tokens, int):
                                trajectory.usage["prompt_tokens"] += prompt_tokens
                            if isinstance(completion_tokens, int):
                                trajectory.usage["completion_tokens"] += completion_tokens
                            if isinstance(total_tokens, int):
                                trajectory.usage["total_tokens"] += total_tokens
                        except Exception:
                            # Best-effort; ignore malformed usage stats
                            pass

                    # If no tool call is generated, turn is finished
                    if len(tool_calls) == 1:
                        # If there's a user simulator, no tool call means the policy is ready to provide final response on this turn
                        if tool_calls[0].tool_name == "_no_tool_call" and user_simulator:
                            turn_completed = True
                            break
                        # If there's no user simulator, then it marks the end of the episode as LLM think there is no tool call needed.
                        elif tool_calls[0].tool_name in ["_playback_terminate", "_no_tool_call"]:
                            trajectory.terminated = True
                            # Ensure finish_reason is a string before converting
                            trajectory.termination_reason = TerminationReason.from_str(str(finish_reason))
                            break

                    # Execute each tool call sequentially
                    for tool_call in tool_calls:
                        # Execute tool call for this environment
                        observation, reward, env_end, info = await envs.step(rollout_idx, tool_call)

                        tool_response = envs.format_tool_response(observation)

                        policy.add_tool_response(
                            rollout_idx,
                            tool_call,
                            tool_response,
                            trajectory.conversation_history,
                            reward,
                            env_end,
                            info,
                        )
                        update_evaluation_row_messages()

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
                                "terminated": env_end,
                                "info": info.get("control_plane", {}),
                                "tool_calls": [f"{tool_call.tool_name}({tool_call.arguments})"],
                                "num_tool_calls": 1,
                            }
                            print(f"🔍 control_plane_step: {control_plane_step}")
                            trajectory.conversation_history[-1]["control_plane_step"] = control_plane_step
                            trajectory.control_plane_steps.append(control_plane_step)

                            # Log conversation state for playback if in recording mode
                            if recording_mode:
                                policy.log_conversation_state_for_playback(
                                    rollout_idx, step - 1, trajectory.conversation_history
                                )

                        if env_end:
                            # if the env marks the end of the rollout, break the tool call loop
                            # but set the termination reason later after the final policy call
                            trajectory.terminated = True
                            break

                        if step >= steps:
                            trajectory.terminated = True
                            trajectory.termination_reason = TerminationReason.MAX_STEPS
                            break

                    # Update current observation for potential next turn
                    if observation is not None:
                        current_observation = observation

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
                        "terminated": env_end,
                        "info": info.get("control_plane", {}),
                        "tool_calls": tool_calls_summary,
                        "num_tool_calls": len(tool_calls),
                    }
                    trajectory.conversation_history[-1]["control_plane_step"] = control_plane_step
                    trajectory.control_plane_steps.append(control_plane_step)

                    # Log conversation state for playback if in recording mode
                    if recording_mode:
                        policy.log_conversation_state_for_playback(
                            rollout_idx, step - 1, trajectory.conversation_history
                        )

                # if the env marks end, update control plane summary and do one last policy call, then break the agent loop
                # this is to ensure each turn ends with an assistant message, which will align with the actual agentic llm behavior
                if env_end:
                    _, usage_stats, finish_reason = await policy(
                        tool_schema, rollout_idx, trajectory.conversation_history
                    )
                    update_evaluation_row_messages()
                    if usage_stats:
                        try:
                            prompt_tokens = (
                                usage_stats.get("prompt_tokens")
                                if isinstance(usage_stats, dict)
                                else usage_stats.prompt_tokens
                            )
                            completion_tokens = (
                                usage_stats.get("completion_tokens")
                                if isinstance(usage_stats, dict)
                                else usage_stats.completion_tokens
                            )
                            total_tokens = (
                                usage_stats.get("total_tokens")
                                if isinstance(usage_stats, dict)
                                else usage_stats.total_tokens
                            )
                            if isinstance(prompt_tokens, int):
                                trajectory.usage["prompt_tokens"] += prompt_tokens
                            if isinstance(completion_tokens, int):
                                trajectory.usage["completion_tokens"] += completion_tokens
                            if isinstance(total_tokens, int):
                                trajectory.usage["total_tokens"] += total_tokens
                        except Exception:
                            pass
                    trajectory.terminated = True
                    trajectory.termination_reason = TerminationReason.from_str(str(finish_reason))
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
                        if trajectory.conversation_history and len(trajectory.conversation_history) > 0:
                            openai_logger(
                                {
                                    "messages": trajectory.conversation_history,
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
                        f"🏁 Environmnet indicates rollout {rollout_idx} terminated at step {step} (reward: {trajectory.total_reward}) in thread {threading.current_thread().name}"
                    )
                    break

                # Progress logging
                if step % 10 == 0:
                    logger.debug(f"Rollout {rollout_idx} step {step}, reward: {trajectory.total_reward:.2f}")

            # Set termination reason if not already set (e.g., due to step limit)
            if not trajectory.termination_reason and step >= steps:
                trajectory.termination_reason = TerminationReason.MAX_STEPS

            # Add termination_reason to the final control_plane_step
            for msg in reversed(trajectory.conversation_history):
                if msg.get("control_plane_step"):
                    msg["control_plane_step"]["termination_reason"] = trajectory.termination_reason
                    break

            logger.info(
                f"✅ Rollout {rollout_idx} completed: {trajectory.steps} steps, reward: {trajectory.total_reward:.2f}, termination: {trajectory.termination_reason}, in thread {threading.current_thread().name}"
            )
        except NonSkippableException as e:
            # terminate the rollout right away, no retry and preserve the current trajectory history.
            # for other types of exceptions, keep propagate them to upper layers and handle them with retry handler.
            trajectory.terminated = True
            trajectory.termination_reason = TerminationReason.NON_SKIPPABLE_ERROR
            trajectory.control_plane_summary.update({"error_message": str(e)})
            logger.error(f"🚨 Rollout {rollout_idx} terminated due to non-skippable error: {str(e)}", exc_info=True)
        finally:
            try:
                await envs.connection_manager.reset_session(session)
            except Exception as e:
                logger.warning(f"Failed to reset session {session.session_id}: {type(e).__name__}: {e}", exc_info=True)
            try:
                await envs.connection_manager.close_session(session)
            except Exception as e:
                logger.warning(f"Failed to close session {session.session_id}: {type(e).__name__}: {e}", exc_info=True)
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
        # Lazy import vendor.tau2 types
        from vendor.tau2.data_model.message import AssistantMessage, UserMessage

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
