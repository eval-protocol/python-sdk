"""
Session Management and Vector Environment

Handles MCPSession management and vector environment operations.
Extracted from mcp_env.py to improve modularity.
"""

import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..execution.manager import ExecutionManager
from ..types import DatasetRow, MCPSession, MCPToolCall

logger = logging.getLogger(__name__)


class GeneralMCPVectorEnv:
    """
    General MCP vector environment that works with any MCP server.

    Manages on-demand MCP sessions for rollouts.
    Driven by dataset prompts and MCP tool discovery, not hardcoded logic.
    """

    def __init__(
        self,
        sessions: List[MCPSession],
        dataset_rows: List[DatasetRow],
        user_prompt_formatter: Optional[Callable] = None,
    ):
        """
        Initialize with dataset-driven configuration.

        Args:
            sessions: MCP sessions
            dataset_rows: Full dataset rows with prompts and context
            user_prompt_formatter: Callback to format user prompts dynamically
        """
        self.sessions = sessions
        self.dataset_rows = dataset_rows
        self.user_prompt_formatter = user_prompt_formatter or self._default_formatter
        self.n = len(sessions)
        self.tool_schemas = []  # Discovered from MCP servers
        self.execution_manager = ExecutionManager()

        if len(sessions) != len(dataset_rows):
            raise ValueError(
                f"Sessions ({len(sessions)}) and dataset rows ({len(dataset_rows)}) must have same length"
            )

    async def reset(self) -> Tuple[List[Any], List[List[Dict]], List[str]]:
        """
        Reset all environments and return observations, tools, and system prompts.

        Establishes persistent MCP sessions for each environment.
        Uses proper MCP pattern: get initial state from resources during session establishment.

        Returns:
            observations: Current state of each environment from MCP resources
            tool_schemas: Available MCP tools for each environment
            system_prompts: System prompts from dataset
        """
        print(f"🔄 Resetting {self.n} MCP environments...")

        async def reset_session(session: MCPSession) -> Tuple[Any, List[Dict]]:
            # Establish a persistent session for each environment.
            await self.execution_manager.connection_manager.initialize_session(session)

            # Get available tools from MCP server
            tool_schemas = await self.execution_manager.connection_manager.discover_tools(
                session
            )

            # PROPER MCP PATTERN: Get initial state from resources during session establishment
            initial_observation = (
                await self.execution_manager.connection_manager.get_initial_state(session)
            )

            # Update session state
            session.terminated = False
            session.last_observation = initial_observation
            return initial_observation, tool_schemas

        # Execute resets in parallel, each with its own isolated session.
        tasks = [reset_session(session) for session in self.sessions]
        results = await asyncio.gather(*tasks)

        observations, tool_schemas_list = zip(*results)
        self.tool_schemas = list(tool_schemas_list)

        # Extract system prompts from dataset
        system_prompts = [row.system_prompt for row in self.dataset_rows]

        return list(observations), self.tool_schemas, system_prompts

    async def step(
        self, tool_calls: List[MCPToolCall]
    ) -> Tuple[List[Any], List[float], List[bool], List[Dict]]:
        """
        Execute tool calls via MCP protocol using persistent sessions.

        Note: This uses MCP tools for actions/interactions during rollout.
        Initial state was obtained from MCP resources during reset() - different pattern.

        Args:
            tool_calls: Tool calls to execute in each environment

        Returns:
            observations, rewards, dones, infos
        """
        if len(tool_calls) != self.n:
            raise ValueError(f"Expected {self.n} tool calls, got {len(tool_calls)}")

        async def step_session(session: MCPSession, tool_call: MCPToolCall):
            if session.terminated:
                return session.last_observation, 0.0, True, {}

            # Handle special playback termination signal
            if tool_call.tool_name == "_playback_terminate":
                logger.info(
                    f"🎬 Session {session.session_id}: Received playback termination signal"
                )
                session.terminated = True
                return (
                    session.last_observation,
                    0.0,
                    True,
                    {"playback_terminated": True},
                )

            # Handle special no-tool-call signal (episode ended, no action needed)
            if tool_call.tool_name == "_no_tool_call":
                logger.info(
                    f"🏁 Session {session.session_id}: No tool call generated, episode likely ended"
                )
                session.terminated = True
                return (
                    session.last_observation,
                    0.0,
                    True,
                    {
                        "no_tool_call": True,
                        "reason": tool_call.arguments.get("reason", "unknown"),
                    },
                )

            # Execute the tool call via MCP protocol
            observation, reward, done, info = (
                await self.execution_manager.connection_manager.call_tool(
                    session, tool_call.tool_name, tool_call.arguments
                )
            )

            # Update session state
            session.last_observation = observation
            session.terminated = done

            return observation, reward, done, info

        # Execute steps in parallel using persistent sessions
        tasks = [
            step_session(session, tool_call)
            for session, tool_call in zip(self.sessions, tool_calls)
        ]
        results = await asyncio.gather(*tasks)

        observations, rewards, dones, infos = zip(*results)
        return list(observations), list(rewards), list(dones), list(infos)

    def format_user_prompts(self, observations: List[Any]) -> List[str]:
        """
        Format user prompts dynamically based on current observations.

        This is the callback pattern - prompts are generated based on current state.
        """
        user_prompts = []

        for obs, row in zip(observations, self.dataset_rows):
            # Use the callback to format the prompt
            prompt = self.user_prompt_formatter(
                row.user_prompt_template, obs, row.environment_context
            )
            user_prompts.append(prompt)

        return user_prompts

    def _default_formatter(self, template: str, observation: Any, context: Dict) -> str:
        """
        Default user prompt formatter.

        Extracts meaningful display data from MCP observations.
        For FrozenLake: extracts grid_layout if available, otherwise uses raw observation.
        """
        # Extract formatted display from observation if available
        display_observation = observation

        if isinstance(observation, dict):
            # For FrozenLake and similar games, prefer grid_layout for display
            if "grid_layout" in observation:
                display_observation = observation["grid_layout"]
            # For other structured observations, try to extract meaningful display
            elif (
                "observation" in observation
                and observation["observation"] != "default_initial_state"
            ):
                display_observation = observation["observation"]
            # If we still have default_initial_state, try to use position info
            elif (
                observation.get("observation") == "default_initial_state"
                and "session_id" in observation
            ):
                # This is the fallback case - we should have gotten the proper initial state from MCP resources
                display_observation = f"Initial game state (Session: {observation['session_id']})\nWaiting for grid data from server..."

        return template.format(observation=display_observation, **context)

    async def close(self):
        """Closes all MCP sessions."""
        print(f"🧹 Closing {self.n} MCP sessions...")
        await self.execution_manager.close_sessions(self.sessions)
        print(f"✅ All MCP sessions closed.")


# Keep the old MCPVectorEnv for backward compatibility
MCPVectorEnv = GeneralMCPVectorEnv