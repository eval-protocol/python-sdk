"""
General Static Policy for MCP Environment Testing

This policy provides a deterministic, non-LLM action sequence for fast iteration
across different MCP environments. It can be configured with custom tool names
and action sequences.

This is useful for:
- Fast testing of multi-session functionality
- Debugging environment behavior
- Performance testing without LLM overhead
"""

import asyncio
import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

# Import the base policy and types for proper recording functionality
from typing import Optional as _Optional

from ..playback_policy import PlaybackPolicyBase
from ..types import MCPToolCall

logger = logging.getLogger(__name__)


class StaticPolicy(PlaybackPolicyBase):
    """
    Static policy that follows a predetermined action sequence.

    Can be configured for different environments with custom tool names and actions.
    """

    def __init__(
        self,
        tool_name: str,
        action_sequence: Optional[List[str]] = None,
        available_actions: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize static policy with recording/playback support.

        Args:
            tool_name: Name of the tool to call for actions (e.g., "lake_move", "lander_action")
            action_sequence: List of actions to execute. If None, uses a default sequence.
            available_actions: List of all available actions for this environment.
            **kwargs: Additional arguments passed to PlaybackPolicyBase
        """
        # Initialize parent class for recording/playback functionality
        super().__init__(**kwargs)

        self.tool_name = tool_name
        self.available_actions = available_actions or []

        # Set default action sequence if not provided
        if action_sequence is None:
            if self.available_actions:
                # Use first few actions as default sequence
                self.action_sequence = self.available_actions[: min(6, len(self.available_actions))]
            else:
                self.action_sequence = ["DEFAULT_ACTION"]
        else:
            self.action_sequence = action_sequence

        self.step_counts = {}  # Track step count per environment

    async def _generate_live_tool_calls(
        self,
        tool_schemas: List[Dict],
        env_index: int,
        conversation_history: List[Dict[str, Any]],
    ) -> Tuple[List[MCPToolCall], Optional[Dict[str, int]], Optional[str]]:
        """
        Generate tool calls in live mode using the static action sequence.

        This implements the abstract method from PlaybackPolicyBase.

        Args:
            tool_schemas: Available tools for this environment
            env_index: Environment index
            conversation_history: Current conversation history for this environment

        Returns:
            List of MCPToolCall objects
        """
        # Get current step count for this environment
        step_count = self.step_counts.get(env_index, 0)

        # Determine action based on step count
        if step_count < len(self.action_sequence):
            action = self.action_sequence[step_count]
        else:
            # After sequence completes, repeat the last action
            action = self.action_sequence[-1]

        # Create tool call in MCPToolCall format
        tool_call = MCPToolCall(tool_name=self.tool_name, arguments={"action": action})

        # Update step count
        self.step_counts[env_index] = step_count + 1

        logger.debug(f"🎮 Env {env_index} step {step_count}: {action}")

        usage_stats: Optional[Dict[str, int]] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return [tool_call], usage_stats, None

    def add_tool_response(
        self,
        env_index: int,
        tool_call: MCPToolCall,
        tool_response: Union[str, List[Dict[str, Any]]],
        conversation_history: List[Dict[str, Any]],
        reward: float = 0.0,
        terminated: bool = False,
        info: Optional[Dict[str, Any]] = None,
    ):
        """Add tool call and response to conversation history for recording."""

        # Find the most recent assistant message with tool calls to get the correct call_id
        call_id = None
        for i in range(len(conversation_history) - 1, -1, -1):
            if conversation_history[i]["role"] == "assistant" and "tool_calls" in conversation_history[i]:
                # Find the tool call that matches our tool_name
                for tc in conversation_history[i]["tool_calls"]:
                    if tc["function"]["name"] == tool_call.tool_name:
                        call_id = tc["id"]
                        break
                if call_id:
                    break

        # Fallback if no matching tool call found
        if not call_id:
            call_id = f"call_{env_index}_{len(conversation_history)}"

        # Add tool response with control plane metadata
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

    def log_conversation_state_for_playback(
        self, env_index: int, step: int, conversation_history: List[Dict[str, Any]]
    ):
        """
        Log the current conversation state in the format required for playback.

        Expected format: {"env_index": 0, "step": 0, "messages": [{..}, {..}]}

        Args:
            env_index: Environment index
            step: Current step number
            conversation_history: List of conversation messages
        """
        # Use EP_PLAYBACK_FILE environment variable for recording
        playback_file = os.environ.get("EP_PLAYBACK_FILE")
        if not playback_file:
            return  # No recording file specified

        playback_entry = {
            "env_index": env_index,
            "step": step,
            "messages": conversation_history.copy(),
        }

        with open(playback_file, "a") as f:
            f.write(json.dumps(playback_entry) + "\n")

    @property
    def model_id(self) -> str:
        """Model identifier for static policy."""
        return f"static-policy-{self.tool_name}-v1"


class RandomPolicy(PlaybackPolicyBase):
    """
    Random policy that selects random actions.
    Useful for testing environment robustness.
    """

    def __init__(
        self,
        tool_name: str,
        available_actions: List[str],
        seed: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize random policy with recording/playback support.

        Args:
            tool_name: Name of the tool to call for actions
            available_actions: List of all available actions for this environment
            seed: Random seed for reproducibility
            **kwargs: Additional arguments passed to PlaybackPolicyBase
        """
        # Initialize parent class for recording/playback functionality
        super().__init__(**kwargs)

        self.tool_name = tool_name
        self.available_actions = available_actions
        self.random = random.Random(seed)

    async def _generate_live_tool_calls(
        self,
        tool_schemas: List[Dict],
        env_index: int,
        conversation_history: List[Dict[str, Any]],
    ) -> Tuple[List[MCPToolCall], Optional[Dict[str, int]], Optional[str]]:
        """
        Generate random tool calls in live mode.

        Args:
            tool_schemas: Available tools for this environment
            env_index: Environment index
            conversation_history: Current conversation history for this environment

        Returns:
            List of MCPToolCall objects
        """
        # Select random action
        action = self.random.choice(self.available_actions)

        # Create tool call
        tool_call = MCPToolCall(tool_name=self.tool_name, arguments={"action": action})

        logger.debug(f"🎲 Env {env_index}: {action}")

        usage_stats: Optional[Dict[str, int]] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return [tool_call], usage_stats, None

    def add_tool_response(
        self,
        env_index: int,
        tool_call: MCPToolCall,
        tool_response: Union[str, List[Dict[str, Any]]],
        conversation_history: List[Dict[str, Any]],
        reward: float = 0.0,
        terminated: bool = False,
        info: Optional[Dict[str, Any]] = None,
    ):
        """Add tool call and response to conversation history for recording."""

        # Find the most recent assistant message with tool calls
        call_id = None
        for i in range(len(conversation_history) - 1, -1, -1):
            if conversation_history[i]["role"] == "assistant" and "tool_calls" in conversation_history[i]:
                for tc in conversation_history[i]["tool_calls"]:
                    if tc["function"]["name"] == tool_call.tool_name:
                        call_id = tc["id"]
                        break
                if call_id:
                    break

        if not call_id:
            call_id = f"call_{env_index}_{len(conversation_history)}"

        # Add tool response with control plane metadata
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

    def log_conversation_state_for_playback(
        self, env_index: int, step: int, conversation_history: List[Dict[str, Any]]
    ):
        """Log the current conversation state for playback recording."""
        playback_file = os.environ.get("EP_PLAYBACK_FILE")
        if not playback_file:
            return

        playback_entry = {
            "env_index": env_index,
            "step": step,
            "messages": conversation_history.copy(),
        }

        with open(playback_file, "a") as f:
            f.write(json.dumps(playback_entry) + "\n")

    @property
    def model_id(self) -> str:
        """Model identifier for random policy."""
        return f"random-policy-{self.tool_name}-v1"
