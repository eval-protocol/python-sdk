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

from ...playback_policy import PlaybackPolicyBase
from ..types import LLMUsageStats, MCPToolCall
from concurrent.futures import ThreadPoolExecutor

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
        # Initialize playback functionality (parent class handles REWARD_KIT_PLAYBACK_FILE automatically)
        super().__init__(**kwargs)

        # Store policy configuration
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_tools_per_turn = max_tools_per_turn

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
        # Use REWARD_KIT_PLAYBACK_FILE environment variable for recording
        playback_file = os.environ.get("REWARD_KIT_PLAYBACK_FILE")
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
    ) -> Tuple[List[MCPToolCall], LLMUsageStats]:
        """
        Generate tool calls using conversation history for proper OpenAI trajectories.

        Args:
            tool_schemas: Available MCP tools for this environment
            env_index: Environment index
            user_prompt: Current user prompt with observation

        Returns:
            List of MCPToolCall objects
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
        usage_stats = LLMUsageStats(
            prompt_tokens=response["usage"]["prompt_tokens"],
            completion_tokens=response["usage"]["completion_tokens"],
            total_tokens=response["usage"]["total_tokens"],
        )

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

            return mcp_tool_calls, usage_stats
        else:
            # No tool calls in response - this is normal when episode ends or LLM provides only text
            logger.info(f"No tool calls in response for env {env_index}, message content: {message.get('content')}")
            return [
                MCPToolCall(
                    tool_name="_no_tool_call",
                    arguments={
                        "reason": "no_tool_call_generated",
                    },
                )
            ], usage_stats


class FireworksPolicy(LLMBasePolicy):
    """
    Fireworks AI policy implementation that works with ANY MCP environment via tool calling.

    NO environment-specific logic - everything comes from MCP tools and dataset prompts.
    Supports both live mode (using Fireworks LLM) and playback mode (replaying recorded trajectories).
    """

    from fireworks import DeploymentTypeLiteral

    def __init__(
        self,
        model_id: str,
        temperature: float = 0.2,
        deployment_type: DeploymentTypeLiteral = "serverless",
        max_tokens: int = 4096,
        max_tools_per_turn: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize Fireworks policy.

        Args:
            model_id: Fireworks model identifier (e.g., "accounts/fireworks/models/qwen3-235b-a22b")
            temperature: Sampling temperature (0.0 to 2.0)
            deployment_type: "serverless", "on-demand", or "auto"
            max_tokens: Maximum tokens to generate per request
            max_tools_per_turn: Maximum number of tool calls per turn (None = unlimited, 1 = single tool)
        """
        super().__init__(model_id, temperature, max_tokens, max_tools_per_turn, **kwargs)

        self.deployment_type = deployment_type

        # Only initialize Fireworks LLM in live mode (not in playback mode)
        if not self._is_playback:
            # Import Fireworks Build SDK - optional at module level
            try:
                from fireworks import LLM
            except ImportError:
                raise ImportError(
                    "The 'fireworks-ai' package is required for FireworksPolicy. "
                    "Please install it with 'pip install fireworks-ai'"
                )

            # Verify authentication
            from ...auth import get_fireworks_api_key

            api_key = get_fireworks_api_key()
            if not api_key:
                raise ValueError(
                    "FIREWORKS_API_KEY environment variable or ~/.fireworks/auth.ini file is required "
                    "to use FireworksPolicy. See the reward-kit documentation for setup instructions."
                )

            # Set the API key for the Fireworks SDK
            os.environ["FIREWORKS_API_KEY"] = api_key

            # Initialize the LLM instance using Build SDK
            try:
                self.llm = LLM(
                    model=self.model_id,
                    deployment_type=self.deployment_type,
                    temperature=self.temperature,
                )
                logger.info(f"✅ Initialized Fireworks LLM: {self.model_id} ({self.deployment_type})")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Fireworks LLM '{self.model_id}': {e}")
            # Create dedicated executor for non-blocking LLM calls
            self.llm_executor = ThreadPoolExecutor(
                max_workers=16,  # Allow up to 16 concurrent LLM API calls
                thread_name_prefix="fireworks-api",
            )
        else:
            # In playback mode, skip expensive LLM initialization
            self.llm = None
            logger.info(f"🎬 Playback mode: Skipping Fireworks LLM initialization for performance")

    def __del__(self):
        """Clean up executor on garbage collection."""
        if hasattr(self, "llm_executor"):
            self.llm_executor.shutdown(wait=False)

    def _clean_messages_for_api(self, messages: List[Dict]) -> List[Dict]:
        """
        Clean messages by removing metadata fields that Fireworks API doesn't accept.

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
        Make a Fireworks API call.

        Args:
            messages: Conversation messages (may contain metadata)
            tools: Available tools in OpenAI format

        Returns:
            API response in OpenAI format
        """
        llm = self.llm
        if llm is None:
            raise RuntimeError("Fireworks LLM not initialized")

        # Clean messages by removing metadata before sending to API
        clean_messages = self._clean_messages_for_api(messages)

        current_request = {
            "messages": clean_messages,
            "tools": tools,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            self.llm_executor, lambda: llm.chat.completions.create(**current_request)
        )

        # Convert Fireworks response to standard format
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
        Convert MCP tool schemas to OpenAI function calling format for Fireworks.

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
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": tool_result_content
                        }
                    ]
                }
                clean_messages.append(converted_msg)
            elif clean_msg.get("role") == "assistant" and "tool_calls" in clean_msg:
                # Convert assistant message with tool_calls to Anthropic format
                # Anthropic uses content blocks instead of tool_calls field
                content_blocks = []
                
                # Add text content if present
                if clean_msg.get("content"):
                    content_blocks.append({
                        "type": "text",
                        "text": clean_msg["content"]
                    })
                
                # Convert tool_calls to tool_use content blocks
                for tool_call in clean_msg.get("tool_calls", []):
                    if tool_call.get("type") == "function":
                        import json
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tool_call["id"],
                            "name": tool_call["function"]["name"],
                            "input": json.loads(tool_call["function"]["arguments"]) if isinstance(tool_call["function"]["arguments"], str) else tool_call["function"]["arguments"]
                        })
                
                converted_msg = {
                    "role": "assistant",
                    "content": content_blocks
                }
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
        if hasattr(response, 'content'):
            for content_block in response.content:
                if hasattr(content_block, 'type') and content_block.type == 'tool_use':
                    tool_calls.append({
                        "id": content_block.id,
                        "type": "function",
                        "function": {
                            "name": content_block.name,
                            "arguments": json.dumps(content_block.input),
                        },
                    })

        # Get text content
        text_content = ""
        if hasattr(response, 'content'):
            for content_block in response.content:
                if hasattr(content_block, 'type') and content_block.type == 'text':
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
