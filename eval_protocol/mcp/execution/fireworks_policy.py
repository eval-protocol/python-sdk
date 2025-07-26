"""
Fireworks AI Policy Implementation

This module contains the FireworksPolicy class that integrates with Fireworks AI's LLM API
for tool calling and conversation management in MCP environments.
"""

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from .base_policy import LLMBasePolicy
from ..types import MCPToolCall

logger = logging.getLogger(__name__)


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
            deployment_type: "serverless", "on-demand", "auto", or "on-demand-lora"
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