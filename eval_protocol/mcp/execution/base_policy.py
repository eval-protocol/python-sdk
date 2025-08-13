"""
Base Policy for LLM Policies

This module contains the LLMBasePolicy abstract base class that provides
common functionality for all LLM-based policies (Fireworks, OpenAI, Anthropic, etc.)
"""

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from openai.types import CompletionUsage

from ...playback_policy import PlaybackPolicyBase
from ...types import MCPToolCall

logger = logging.getLogger(__name__)


class LLMBasePolicy(PlaybackPolicyBase, ABC):
    """
    Base class for LLM policies that work with MCP environments via tool calling.

    This abstraction enables shared code between FireworksPolicy and OpenAIPolicy.
    Maintains conversation history per environment for proper OpenAI-style trajectories.
    """

    def __init__(
        self,
        model_id: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        max_tools_per_turn: Optional[int] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize base policy with automatic record/playback detection.

        Args:
            model_id: Model identifier
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate per request
            max_tools_per_turn: Maximum number of tool calls per turn (None = unlimited, 1 = single tool)
        """
        # Initialize playback functionality (parent class handles EP_PLAYBACK_FILE automatically)
        super().__init__(**kwargs)

        # Store policy configuration
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_tools_per_turn = max_tools_per_turn
        self.base_url = base_url

        # Initialize conversation state tracking for proper OpenAI trajectories
        self.initialized = False

    @abstractmethod
    async def _make_llm_call(self, messages: List[Dict], tools: List[Dict]) -> Dict:
        """
        Make an LLM API call. Subclasses must implement this.

        Args:
            messages: Conversation messages
            tools: Available tools in OpenAI format

        Returns:
            LLM response with choices[0].message containing content and tool_calls
        """
        pass

    @abstractmethod
    def _convert_mcp_tools_to_llm_format(self, mcp_tools: List[Dict]) -> List[Dict]:
        """
        Convert MCP tool schemas to LLM-specific format.

        Args:
            mcp_tools: List of MCP tool definitions

        Returns:
            List of LLM-compatible tool definitions
        """
        pass

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
        """Add tool call and response to conversation history with control plane metadata."""
        # Use the preserved tool_call_id directly
        if tool_call.tool_call_id is None:
            raise ValueError("Tool call ID is required for tool response recording")

        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call.tool_call_id,
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

        # TODO: because we're using threads now, the ordering will be wrong.

        with open(playback_file, "a") as f:
            f.write(json.dumps(playback_entry) + "\n")

    async def _generate_live_tool_calls(
        self,
        tool_schemas: List[Dict],
        env_index: int,
        conversation_history: List[Dict[str, Any]],
    ) -> Tuple[List[MCPToolCall], CompletionUsage, str]:
        """
        Generate tool calls using conversation history for proper OpenAI trajectories.

        Args:
            tool_schemas: Available MCP tools for this environment
            env_index: Environment index
            user_prompt: Current user prompt with observation

        Returns:
            List of MCPToolCall objects, LLM usage stats, and finish reason
        """
        # Convert MCP tools to LLM format
        llm_tools = self._convert_mcp_tools_to_llm_format(tool_schemas)

        logger.debug(
            f"Environment {env_index} - Converted {len(tool_schemas)} MCP tools to {len(llm_tools)} LLM tools"
        )
        logger.debug(f"Environment {env_index} - Conversation length: {len(conversation_history)} messages")

        try:
            # Make API call with conversation history
            response = await self._make_llm_call(conversation_history, llm_tools)
        except Exception as e:
            logger.error(f"LLM API call failed for env {env_index}: {e}")
            raise e

        # ADD ASSISTANT MESSAGE TO ACTUAL CONVERSATION HISTORY
        # This is crucial for proper tool call ID management in add_tool_response
        assistant_message_for_history = {
            "role": "assistant",
            "content": response["choices"][0]["message"]["content"],
        }
        usage_stats = CompletionUsage(
            prompt_tokens=response["usage"]["prompt_tokens"],
            completion_tokens=response["usage"]["completion_tokens"],
            total_tokens=response["usage"]["total_tokens"],
        )

        finish_reason = response["choices"][0]["finish_reason"]

        # Extract tool call from response
        message = response["choices"][0]["message"]
        logger.debug(f"Environment {env_index} - Response message: {message}")

        # Add ALL tool calls if present with the actual API response IDs
        if message.get("tool_calls"):
            assistant_message_for_history["tool_calls"] = message["tool_calls"]

        # Add to actual conversation history
        conversation_history.append(assistant_message_for_history)

        if message.get("tool_calls") and len(message["tool_calls"]) > 0:
            tool_calls = message["tool_calls"]

            # Handle multiple tool calls - create MCPToolCall for each
            mcp_tool_calls = []
            for tool_call in tool_calls:
                mcp_tool_call = MCPToolCall(
                    tool_name=tool_call["function"]["name"],
                    arguments=json.loads(tool_call["function"]["arguments"]),
                    tool_call_id=tool_call["id"],
                )
                mcp_tool_calls.append(mcp_tool_call)

            if self.max_tools_per_turn:
                mcp_tool_calls = mcp_tool_calls[: self.max_tools_per_turn]

            return mcp_tool_calls, usage_stats, finish_reason
        else:
            # No tool calls in response - this is normal when episode ends or LLM provides only text
            logger.debug(f"No tool calls in response for env {env_index}, message content: {message.get('content')}")
            return (
                [
                    MCPToolCall(
                        tool_name="_no_tool_call",
                        arguments={
                            "reason": "no_tool_call_generated",
                        },
                    )
                ],
                usage_stats,
                finish_reason,
            )
