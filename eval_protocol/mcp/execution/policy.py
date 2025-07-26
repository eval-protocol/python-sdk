"""
LLM Policy Execution and Tool Calling

Base classes and implementations for LLM policies that work with MCP environments.
Extracted from mcp_env.py to improve modularity and enable OpenAI integration.
"""

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from concurrent.futures import ThreadPoolExecutor

from .base_policy import LLMBasePolicy
from ..types import LLMUsageStats, MCPToolCall

# Try to import FireworksPolicy from separate module - it's optional
try:
    from .fireworks_policy import FireworksPolicy
except ImportError:
    # FireworksPolicy not available (fireworks-ai package not installed)
    FireworksPolicy = None

logger = logging.getLogger(__name__)




class OpenAIPolicy(LLMBasePolicy):
    """
    OpenAI policy implementation that works with ANY MCP environment via tool calling.

    NO environment-specific logic - everything comes from MCP tools and dataset prompts.
    Supports both live mode (using OpenAI API) and playback mode (replaying recorded trajectories).
    """

    def __init__(
        self,
        model_id: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        max_tools_per_turn: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize OpenAI policy.

        Args:
            model_id: OpenAI model identifier (e.g., "gpt-4o", "gpt-4o-mini", "gpt-4-turbo")
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate per request
            max_tools_per_turn: Maximum number of tool calls per turn (None = unlimited, 1 = single tool)
        """
        super().__init__(model_id, temperature, max_tokens, max_tools_per_turn, **kwargs)

        # Only initialize OpenAI client in live mode (not in playback mode)
        if not self._is_playback:
            # Import OpenAI SDK - optional at module level
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError(
                    "The 'openai' package is required for OpenAIPolicy. " "Please install it with 'pip install openai'"
                )

            # Verify authentication
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable is required "
                    "to use OpenAIPolicy. Set this variable before running."
                )

            # Initialize the OpenAI client
            try:
                self.client = AsyncOpenAI(api_key=api_key)
                logger.info(f"✅ Initialized OpenAI client: {self.model_id}")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize OpenAI client for '{self.model_id}': {e}")
        else:
            # In playback mode, skip expensive client initialization
            self.client = None
            logger.info(f"🎬 Playback mode: Skipping OpenAI client initialization for performance")

    def _clean_messages_for_api(self, messages: List[Dict]) -> List[Dict]:
        """
        Clean messages by removing metadata fields that OpenAI API doesn't accept.

        Args:
            messages: Conversation messages with potential metadata

        Returns:
            Clean messages without metadata fields
        """
        clean_messages = []
        for msg in messages:
            clean_msg = msg.copy()
            # Remove metadata field if present
            if "metadata" in clean_msg:
                del clean_msg["metadata"]
            clean_messages.append(clean_msg)
        return clean_messages

    async def _make_llm_call(self, messages: List[Dict], tools: List[Dict]) -> Dict:
        """
        Make an OpenAI API call.

        Args:
            messages: Conversation messages (may contain metadata)
            tools: Available tools in OpenAI format

        Returns:
            API response in OpenAI format
        """
        # Clean messages by removing metadata before sending to API
        clean_messages = self._clean_messages_for_api(messages)

        current_request = {
            "model": self.model_id,
            "messages": clean_messages,
            "tools": tools,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if self.client is None:
            raise RuntimeError("OpenAI client not initialized")

        # Make the API call
        response = await self.client.chat.completions.create(**current_request)

        # Convert OpenAI response to standard format
        return {
            "choices": [
                {
                    "message": {
                        "content": response.choices[0].message.content,
                        "tool_calls": (
                            [
                                {
                                    "id": tc.id,
                                    "type": tc.type,
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in (response.choices[0].message.tool_calls or [])
                            ]
                            if response.choices[0].message.tool_calls
                            else []
                        ),
                    }
                }
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }

    def _convert_mcp_tools_to_llm_format(self, mcp_tools: List[Dict]) -> List[Dict]:
        """
        Convert MCP tool schemas to OpenAI function calling format.

        Args:
            mcp_tools: List of MCP tool definitions

        Returns:
            List of OpenAI-compatible tool definitions
        """
        openai_tools = []

        for mcp_tool in mcp_tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": mcp_tool["name"],
                    "description": mcp_tool.get("description", f"Execute {mcp_tool['name']} action"),
                    "parameters": mcp_tool.get(
                        "input_schema",
                        {"type": "object", "properties": {}, "required": []},
                    ),
                },
            }
            openai_tools.append(openai_tool)

        return openai_tools


class AnthropicPolicy(LLMBasePolicy):
    """
    Anthropic policy implementation that works with ANY MCP environment via tool calling.

    NO environment-specific logic - everything comes from MCP tools and dataset prompts.
    Supports both live mode (using Anthropic API) and playback mode (replaying recorded trajectories).
    """

    def __init__(
        self,
        model_id: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        max_tools_per_turn: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize Anthropic policy.

        Args:
            model_id: Anthropic model identifier (e.g., "claude-3-5-sonnet-20241022", "claude-3-opus-20240229")
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate per request
            max_tools_per_turn: Maximum number of tool calls per turn (None = unlimited, 1 = single tool)
        """
        super().__init__(model_id, temperature, max_tokens, max_tools_per_turn, **kwargs)

        # Only initialize Anthropic client in live mode (not in playback mode)
        if not self._is_playback:
            # Import Anthropic SDK - optional at module level
            try:
                from anthropic import AsyncAnthropic
            except ImportError:
                raise ImportError(
                    "The 'anthropic' package is required for AnthropicPolicy. "
                    "Please install it with 'pip install anthropic'"
                )

            # Verify authentication
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY environment variable is required "
                    "to use AnthropicPolicy. Set this variable before running."
                )

            # Initialize the Anthropic client
            try:
                self.client = AsyncAnthropic(api_key=api_key)
                logger.info(f"✅ Initialized Anthropic client: {self.model_id}")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Anthropic client for '{self.model_id}': {e}")
        else:
            # In playback mode, skip expensive client initialization
            self.client = None
            logger.info(f"🎬 Playback mode: Skipping Anthropic client initialization for performance")

    def _clean_messages_for_api(self, messages: List[Dict]) -> Tuple[List[Dict], Optional[str]]:
        """
        Clean messages by removing metadata fields, extracting system message, and converting tool messages.

        Anthropic handles system messages separately and doesn't support "tool" role messages.
        Tool results must be converted to "user" messages with tool_result content blocks.

        Args:
            messages: Conversation messages with potential metadata and system messages

        Returns:
            Tuple of (clean_messages_without_system, system_message_content)
        """
        clean_messages = []
        system_message = None

        for msg in messages:
            clean_msg = msg.copy()

            # Remove metadata field if present
            if "metadata" in clean_msg:
                del clean_msg["metadata"]

            # Extract system message separately - Anthropic handles it differently
            if clean_msg.get("role") == "system":
                system_message = clean_msg["content"]
            elif clean_msg.get("role") == "tool":
                # Convert tool message to user message with tool_result content
                # Anthropic expects tool results as content blocks in user messages
                tool_call_id = clean_msg.get("tool_call_id", "unknown")
                tool_result_content = clean_msg.get("content", "")

                converted_msg = {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": tool_call_id, "content": tool_result_content}],
                }
                clean_messages.append(converted_msg)
            elif clean_msg.get("role") == "assistant" and "tool_calls" in clean_msg:
                # Convert assistant message with tool_calls to Anthropic format
                # Anthropic uses content blocks instead of tool_calls field
                content_blocks = []

                # Add text content if present
                if clean_msg.get("content"):
                    content_blocks.append({"type": "text", "text": clean_msg["content"]})

                # Convert tool_calls to tool_use content blocks
                for tool_call in clean_msg.get("tool_calls", []):
                    if tool_call.get("type") == "function":
                        import json

                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "id": tool_call["id"],
                                "name": tool_call["function"]["name"],
                                "input": (
                                    json.loads(tool_call["function"]["arguments"])
                                    if isinstance(tool_call["function"]["arguments"], str)
                                    else tool_call["function"]["arguments"]
                                ),
                            }
                        )

                converted_msg = {"role": "assistant", "content": content_blocks}
                clean_messages.append(converted_msg)
            else:
                clean_messages.append(clean_msg)

        return clean_messages, system_message

    async def _make_llm_call(self, messages: List[Dict], tools: List[Dict]) -> Dict:
        """
        Make an Anthropic API call.

        Args:
            messages: Conversation messages (may contain metadata and system messages)
            tools: Available tools in Anthropic format

        Returns:
            API response in OpenAI-compatible format
        """
        # Clean messages and extract system message
        clean_messages, system_message = self._clean_messages_for_api(messages)

        current_request = {
            "model": self.model_id,
            "messages": clean_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        # Add system message if present
        if system_message:
            current_request["system"] = system_message

        # Add tools if present
        if tools:
            current_request["tools"] = tools

        if self.client is None:
            raise RuntimeError("Anthropic client not initialized")

        # Make the API call
        response = await self.client.messages.create(**current_request)

        # Convert Anthropic response to OpenAI-compatible format
        tool_calls = []
        if hasattr(response, "content"):
            for content_block in response.content:
                if hasattr(content_block, "type") and content_block.type == "tool_use":
                    tool_calls.append(
                        {
                            "id": content_block.id,
                            "type": "function",
                            "function": {
                                "name": content_block.name,
                                "arguments": json.dumps(content_block.input),
                            },
                        }
                    )

        # Get text content
        text_content = ""
        if hasattr(response, "content"):
            for content_block in response.content:
                if hasattr(content_block, "type") and content_block.type == "text":
                    text_content = content_block.text
                    break

        return {
            "choices": [
                {
                    "message": {
                        "content": text_content,
                        "tool_calls": tool_calls if tool_calls else None,
                    }
                }
            ],
            "usage": {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
        }

    def _convert_mcp_tools_to_llm_format(self, mcp_tools: List[Dict]) -> List[Dict]:
        """
        Convert MCP tool schemas to Anthropic tool calling format.

        Args:
            mcp_tools: List of MCP tool definitions

        Returns:
            List of Anthropic-compatible tool definitions
        """
        anthropic_tools = []

        for mcp_tool in mcp_tools:
            anthropic_tool = {
                "name": mcp_tool["name"],
                "description": mcp_tool.get("description", f"Execute {mcp_tool['name']} action"),
                "input_schema": mcp_tool.get(
                    "input_schema",
                    {"type": "object", "properties": {}, "required": []},
                ),
            }
            anthropic_tools.append(anthropic_tool)

        return anthropic_tools
