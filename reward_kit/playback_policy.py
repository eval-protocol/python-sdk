"""
Playback policy base class for record-and-replay functionality.

This module implements the abstract base class that handles all playback logic,
allowing concrete policy classes to inherit replay functionality while focusing
on their specific implementation details.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from .mcp.types import MCPToolCall

logger = logging.getLogger(__name__)


class PlaybackPolicyBase(ABC):
    """
    Abstract base class for policies that support record-and-playback functionality.

    This class handles all playback logic including trajectory loading, parsing,
    and step management. Concrete policy classes inherit from this to get
    replay functionality while implementing their own live mode logic.
    """

    def __init__(
        self,
        _playback_actions: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        **kwargs,
    ):
        """
        Initialize policy with optional playback actions.

        Args:
            _playback_actions: Pre-parsed playback actions organized by environment.
                              Format: {env_index: [{"step": int, "messages": [...]}]}
            **kwargs: Additional arguments passed to concrete implementations
        """
        # Playback state management
        self._playback_actions = _playback_actions
        self._is_playback = _playback_actions is not None
        # print("show is playback: ", self._is_playback, _playback_actions)
        self._playback_step_counters = {}  # {env_index: current_step}

        # Environment variable override
        playback_file = os.environ.get("REWARD_KIT_PLAYBACK_FILE")
        if playback_file and not self._is_playback:
            logger.info(
                f"🎬 Auto-enabling playback mode from environment variable: {playback_file}"
            )
            self._playback_actions = self._load_trajectory_file(playback_file)
            self._is_playback = self._playback_actions is not None

        # Initialize step counters if in playback mode
        if self._is_playback and self._playback_actions:
            for env_index in self._playback_actions.keys():
                self._playback_step_counters[env_index] = 0

        logger.debug(
            f"PlaybackPolicyBase initialized: playback_mode={self._is_playback}"
        )

    @staticmethod
    def _load_trajectory_file(
        filepath: str,
    ) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """
        Load and parse trajectory file into organized playback actions.

        Expected JSONL format per design document:
        {"env_index": 0, "step": 0, "messages": [{..}, {..}]}
        {"env_index": 1, "step": 0, "messages": [{..}, {..}]}
        {"env_index": 0, "step": 1, "messages": [{..}, {..}]}

        Args:
            filepath: Path to trajectory JSONL file

        Returns:
            Organized playback actions: {env_index: [{"step": int, "messages": [...]}]}
        """
        if not os.path.exists(filepath):
            logger.error(f"Trajectory file not found: {filepath}")
            return None

        try:
            playback_actions = {}
            valid_entries = 0

            with open(filepath, "r") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        entry = json.loads(line)

                        # Validate required fields
                        if not isinstance(entry, dict):
                            logger.warning(
                                f"Line {line_num}: Entry is not a dictionary, skipping"
                            )
                            continue

                        env_index = entry.get("env_index")
                        step = entry.get("step")
                        messages = entry.get("messages")

                        if env_index is None or step is None or messages is None:
                            logger.warning(
                                f"Line {line_num}: Missing required fields (env_index, step, messages), skipping"
                            )
                            continue

                        # Convert env_index to string for consistent dictionary keys
                        env_key = str(env_index)

                        # Initialize environment list if needed
                        if env_key not in playback_actions:
                            playback_actions[env_key] = []

                        # Add step entry
                        playback_actions[env_key].append(
                            {"step": step, "messages": messages}
                        )

                        valid_entries += 1

                    except json.JSONDecodeError as e:
                        logger.warning(f"Line {line_num}: Invalid JSON - {e}")
                        continue

            # Sort each environment's actions by step
            for env_key in playback_actions:
                playback_actions[env_key].sort(key=lambda x: x["step"])

            if playback_actions:
                logger.info(
                    f"✅ Loaded {valid_entries} trajectory entries for {len(playback_actions)} environments"
                )
                return playback_actions
            else:
                logger.warning(
                    f"⚠️  Trajectory file {filepath} exists but contains no valid entries. "
                    f"Falling back to recording mode. Please check file format - expected JSONL with "
                    f"'env_index', 'step', and 'messages' fields."
                )
                return None

        except Exception as e:
            logger.error(f"Error loading trajectory file {filepath}: {e}")
            return None

    def _get_playback_messages(self, env_index: int) -> Optional[List[Dict[str, Any]]]:
        """
        Get the next playback messages for the specified environment.

        Args:
            env_index: Environment index

        Returns:
            Messages list for the current step, or None if no more steps
        """
        if not self._is_playback or not self._playback_actions:
            return None

        env_key = str(env_index)
        if env_key not in self._playback_actions:
            logger.warning(f"No playback data for environment {env_index}")
            return None

        current_step = self._playback_step_counters.get(str(env_index), 0)
        env_actions = self._playback_actions[env_key]

        # Find action for current step
        for action in env_actions:
            if action["step"] == current_step:
                # Increment step counter for next call
                self._playback_step_counters[str(env_index)] = current_step + 1
                logger.debug(
                    f"🎬 Environment {env_index}: Returning playback messages for step {current_step}"
                )
                return action["messages"]

        # No more recorded actions available
        logger.debug(
            f"🎬 Environment {env_index}: No more playback data (step {current_step})"
        )
        return None

    def has_more_playback_data(self, env_index: int) -> bool:
        """
        Check if there are more playback actions available for an environment.

        Args:
            env_index: Environment index

        Returns:
            True if more actions are available, False otherwise
        """
        if not self._is_playback or not self._playback_actions:
            return False

        env_key = str(env_index)
        if env_key not in self._playback_actions:
            return False

        current_step = self._playback_step_counters.get(str(env_index), 0)
        env_actions = self._playback_actions[env_key]

        # Check if there's an action for the current step
        return any(action["step"] == current_step for action in env_actions)

    @abstractmethod
    async def _generate_live_tool_calls(
        self,
        tool_schemas: List[List[Dict]],
        observations: List[Any],
        system_prompts: List[str],
        user_prompts: List[Union[str, List[Dict[str, Any]]]],
    ) -> List["MCPToolCall"]:
        """
        Generate tool calls in live mode. Concrete classes must implement this.

        Args:
            tool_schemas: Available tools for each environment
            observations: Current observations from environments
            system_prompts: System prompts for each environment
            user_prompts: User prompts for each environment

        Returns:
            List of ToolCall objects for each environment
        """
        pass

    async def __call__(
        self,
        tool_schema: List[Dict],
        env_index: int,
        conversation_history: List[Dict[str, Any]],
    ):
        """
        Main policy call method. Delegates to playback or live mode.

        Args:
            tool_schemas: Available tools for each environment
            observations: Current observations from environments
            system_prompts: System prompts for each environment
            user_prompts: User prompts for each environment

        Returns:
            List of ToolCall objects for each environment
        """
        # TODO: fix this entire class to reflect changes in the LLMBasePolicy class
        pass
        # if self._is_playback:
        #     return await self._handle_playback_mode(
        #         tool_schemas
        #     )
        # else:
        #     return await self._generate_live_tool_calls(
        #         tool_schemas, observations, system_prompts, user_prompts
        #     )

    async def _handle_playback_mode(
        self,
        tool_schemas: List[List[Dict]],
    ) -> List["MCPToolCall"]:
        """
        Handle policy calls in playback mode by extracting tool calls from recorded messages.

        Args:
            tool_schemas: Available tools for each environment (used for validation)
            observations: Current observations (not used in playback)
            system_prompts: System prompts (not used in playback)
            user_prompts: User prompts (not used in playback)

        Returns:
            List of ToolCall objects extracted from recorded messages
        """
        tool_calls = []
        n_envs = len(tool_schemas)

        for env_index in range(n_envs):
            # Get recorded messages for this environment and step
            messages = self._get_playback_messages(env_index)

            if messages is None:
                # No more recorded actions - signal early termination by using a special tool call
                # This will cause the environment to terminate (similar to how recording worked)
                tool_calls.append(
                    MCPToolCall(
                        "_playback_terminate", {"reason": "no_more_recorded_actions"}
                    )
                )
                logger.info(
                    f"🎬 Environment {env_index}: No more recorded actions, signaling termination"
                )
                continue

            # Extract tool call from the last assistant message with tool_calls
            tool_call = self._extract_tool_call_from_messages(messages, env_index)
            tool_calls.append(tool_call)

        return tool_calls

    def _extract_tool_call_from_messages(
        self, messages: List[Dict[str, Any]], env_index: int
    ) -> "MCPToolCall":
        """
        Extract tool call from recorded conversation messages.

        Args:
            messages: List of conversation messages
            env_index: Environment index for logging

        Returns:
            ToolCall object extracted from messages
        """
        # Look for the last assistant message with tool_calls
        for message in reversed(messages):
            if message.get("role") == "assistant" and message.get("tool_calls"):
                tool_calls = message["tool_calls"]
                if tool_calls and len(tool_calls) > 0:
                    tool_call = tool_calls[0]  # Use first tool call

                    # Extract function name and arguments
                    function = tool_call.get("function", {})
                    tool_name = function.get("name", "unknown")

                    # Parse arguments if they're a string
                    arguments = function.get("arguments", {})
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            logger.warning(
                                f"🎬 Environment {env_index}: Failed to parse tool call arguments: {arguments}"
                            )
                            arguments = {}

                    logger.debug(
                        f"🎬 Environment {env_index}: Extracted tool call {tool_name}({arguments})"
                    )
                    return MCPToolCall(tool_name, arguments)

        # Fallback if no tool calls found
        logger.warning(
            f"🎬 Environment {env_index}: No tool calls found in messages, using unknown tool"
        )
        return MCPToolCall("unknown", {})

    def is_playback_mode(self) -> bool:
        """
        Check if the policy is in playback mode.

        Returns:
            True if in playback mode, False otherwise
        """
        return self._is_playback

    def get_playback_progress(self) -> Dict[str, Any]:
        """
        Get playback progress information.

        Returns:
            Dictionary with playback progress details
        """
        if not self._is_playback:
            return {"playback_mode": False}

        progress = {
            "playback_mode": True,
            "environments": {},
            "total_environments": (
                len(self._playback_actions) if self._playback_actions else 0
            ),
        }

        if self._playback_actions:
            for env_key, actions in self._playback_actions.items():
                env_index = int(env_key)
                current_step = self._playback_step_counters.get(str(env_index), 0)
                total_steps = len(actions)

                progress["environments"][env_index] = {
                    "current_step": current_step,
                    "total_steps": total_steps,
                    "completed": current_step >= total_steps,
                }

        return progress

    def log_conversation_state_for_playback(self, env_index: int, step: int, conversation_history: List[Dict[str, Any]]):
        """
        Log the current conversation state in the format required for playback.
        
        Base implementation that subclasses can override with specific behavior.
        Expected format: {"env_index": 0, "step": 0, "messages": [{..}, {..}]}

        Args:
            env_index: Environment index
            step: Current step number
            conversation_history: List of conversation messages
        """
        # Use REWARD_KIT_PLAYBACK_FILE environment variable for recording
        playback_file = os.environ.get("REWARD_KIT_PLAYBACK_FILE")
        if not playback_file:
            return  # No recording file specified

        playback_entry = {
            "env_index": env_index,
            "step": step,
            "messages": conversation_history.copy(),
        }

        with open(playback_file, "a") as f:
            f.write(json.dumps(playback_entry) + "\n")