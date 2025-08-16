"""
Session Management and Vector Environment

Handles MCPSession management and vector environment operations.
Extracted from mcp_env.py to improve modularity.
"""

import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ...types import DatasetRow, MCPSession, MCPToolCall
from ..client.connection import MCPConnectionManager

logger = logging.getLogger(__name__)


# TODO: rename this file or the other manager.py
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
        self.connection_manager = MCPConnectionManager()
        self.usage_stats = {}  # llm usage stats for monitoring

        if len(sessions) != len(dataset_rows):
            raise ValueError(
                f"Sessions ({len(sessions)}) and dataset rows ({len(dataset_rows)}) must have same length"
            )

    async def reset(self, session: MCPSession) -> Tuple[Any, List[Dict]]:
        """
        Reset a single session - establish connection, get tools and initial state.

        This is thread-safe and can be called from worker threads.
        """
        await self.connection_manager.initialize_session(session)
        # Get available tools from MCP server
        tool_schemas = await self.connection_manager.discover_tools(session)

        if not self.tool_schemas:
            self.tool_schemas = tool_schemas

        # PROPER MCP PATTERN: Get initial state from resources during session establishment
        initial_observation = await self.connection_manager.get_initial_state(session)

        # Update session state
        session.terminated = False
        session.last_observation = initial_observation

        return initial_observation, tool_schemas

    async def step(self, env_index: int, tool_call: MCPToolCall) -> Tuple[Any, float, bool, Dict]:
        """
        Execute a tool call for a single environment.

        Args:
            env_index: Index of the environment to step
            tool_call: Tool call to execute

        Returns:
            observation: New observation after executing the tool call
            reward: Reward from the environment
            done: Whether the environment is terminated
            info: Additional info from the environment
        """
        if env_index >= self.n or env_index < 0:
            raise ValueError(f"Environment index {env_index} out of range [0, {self.n})")

        session = self.sessions[env_index]

        if session.terminated:
            return session.last_observation, 0.0, True, {}

        # Handle special playback termination signal
        if tool_call.tool_name == "_playback_terminate":
            logger.info(f"🎬 Session {session.session_id}: Received playback termination signal")
            session.terminated = True
            return session.last_observation, 0.0, True, {"playback_terminated": True}

        # Handle special no-tool-call signal
        if tool_call.tool_name == "_no_tool_call":
            logger.info(f"🏁 Session {session.session_id}: No tool call generated, episode likely ended")
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
        observation, reward, done, info = await self.connection_manager.call_tool(
            session, tool_call.tool_name, tool_call.arguments
        )

        # Update session state
        session.last_observation = observation
        session.terminated = done

        return observation, reward, done, info

    def format_user_prompt(self, env_index: int, observation: Any) -> Union[str, List[Dict[str, Any]]]:
        """
        Format user prompt dynamically for a single environment based on current observation.
        """
        if env_index >= self.n or env_index < 0:
            raise ValueError(f"Environment index {env_index} out of range [0, {self.n})")

        dataset_row = self.dataset_rows[env_index]

        # Use the callback to format the prompt
        prompt = self.user_prompt_formatter(
            dataset_row.user_prompt_template,
            observation,
            dataset_row.environment_context,
        )

        return prompt

    def format_tool_response(self, obs: Any) -> Union[str, List[Dict[str, Any]]]:
        """
        Format observation to tool response. If there's an image_url, it will be returned as a multimodal content. If not, it will be returned as a string.
        This is what gets filled in for the tool responses content.
        """

        if isinstance(obs, dict) and obs.get("image_url"):
            image_url = obs["image_url"]["url"]
            obs.pop("image_url")

            return [
                {
                    "type": "text",
                    "text": json.dumps(obs) if isinstance(obs, dict) else str(obs),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                    },
                },
            ]

        else:
            return json.dumps(obs) if isinstance(obs, dict) else str(obs)

    def _default_formatter(self, template: str, obs: Any, context: Dict) -> Union[str, List[Dict[str, Any]]]:
        """
        Default user prompt formatter.

        Extracts meaningful display data from MCP observations.
        For FrozenLake: extracts grid_layout if available, otherwise uses raw observation.
        For visual environments: returns multimodal content with both text and images.

        Returns:
            Either a string (text-only) or a dict (multimodal content)
        """
        # Extract formatted display from observation if available
        display_obs = obs
        image_dict = None

        if isinstance(obs, dict):
            # For visual environments like LunarLander, we have image_url
            if "image_url" in obs:
                image_dict = obs["image_url"]
                display_obs.pop("image_url")
            # For other structured observations, try to extract meaningful display
            elif "observation" in obs and obs["observation"] != "default_initial_state":
                display_obs = obs["observation"]
            # If we still have default_initial_state, try to use position info
            elif obs.get("observation") == "default_initial_state" and "session_id" in obs:
                # This is the fallback case - we should have gotten the proper initial state from MCP resources
                display_obs = (
                    f"Initial game state (Session: {obs['session_id']})\nWaiting for grid data from server..."
                )

        formatted_prompt = template.format(observation=display_obs, **context)

        # If we have image data, return multimodal content
        if image_dict:
            return [
                {
                    "type": "text",
                    "text": formatted_prompt,
                },
                {
                    "type": "image_url",
                    "image_url": image_dict,
                },
            ]

        return formatted_prompt

    async def close(self):
        """Closes all MCP sessions."""
        print(f"🧹 Resetting {self.n} MCP sessions in MCP server...")
        cleanup_tasks = [self.connection_manager.reset_session(session) for session in self.sessions]
        await asyncio.gather(*cleanup_tasks)
        print(f"🧹 Closing {self.n} MCP sessions...")
        tasks = [self.connection_manager.close_session(session) for session in self.sessions]
        await asyncio.gather(*tasks)
        print("✅ All MCP sessions closed.")
