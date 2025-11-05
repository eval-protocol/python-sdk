"""
Model clients for generating responses from various LLM APIs.
"""

import abc
import asyncio
import json  # For parsing content as JSON
import logging
import uuid  # For generating tool call IDs if not provided
from typing import Any, Dict, List, Optional

import aiohttp
from omegaconf import DictConfig
from pydantic import BaseModel  # Added for new models

from ..common_utils import get_user_agent

logger = logging.getLogger(__name__)


# Pydantic models for structured tool calls and generation results
class ToolCallFunction(BaseModel):
    name: str
    arguments: str  # Should be a JSON string


class ToolCall(BaseModel):
    id: str
    type: str = "function"  # OpenAI default
    function: ToolCallFunction


class GenerationResult(BaseModel):
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class ModelClient(abc.ABC):
    """Abstract base class for model clients."""

    def __init__(self, client_config: DictConfig):
        self.client_config = client_config
        self.model_name = client_config.get("model_name", "unknown_model")
        self.temperature = client_config.get("temperature", 0.0)
        self.max_tokens = client_config.get("max_tokens", 1024)
        self.top_p = client_config.get("top_p", 0.95)
        self.top_k = client_config.get("top_k", 20)
        self.min_p = client_config.get("min_p", 0.0)
        # Add reasoning_effort, defaulting to None if not specified in config
        self.reasoning_effort = client_config.get("reasoning_effort", None)

    @abc.abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        session: aiohttp.ClientSession,
        tools: Optional[List[Dict[str, Any]]] = None,  # Added tools parameter
    ) -> GenerationResult:  # Changed return type
        """Generates a response from the model given a list of messages."""
        pass


class FireworksModelClient(ModelClient):
    """Client for Fireworks AI models."""

    def __init__(self, client_config: DictConfig, api_key: str):
        super().__init__(client_config)
        self.api_key = api_key
        self.api_base = client_config.get("api_base", "https://api.fireworks.ai/inference/v1")
        # TODO: Initialize rate limiter, retry policy from client_config.api_params

    async def generate(
        self,
        messages: List[Dict[str, str]],
        session: aiohttp.ClientSession,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> GenerationResult:
        url = f"{self.api_base}/chat/completions"

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        # Include reasoning settings if configured (for reasoning-capable models)
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort

        if tools:
            payload["tools"] = tools
            # Fireworks API might use "function" or "any" or specific tool name for tool_choice.
            # "auto" is common for OpenAI. If Fireworks needs specific, this might need adjustment.
            # Or, if it's like older OpenAI, it might not use tool_choice if tools are present.
            # For now, let's assume "auto" or that it's implicit if "tools" is provided.
            # The user's log shows the LLM is attempting tool calls even with the simpler prompt,
            # implying the `tools` parameter is having an effect or the model is well-primed.
            payload["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": get_user_agent(),
        }

        debug_payload_log = json.loads(json.dumps(payload))
        if "messages" in debug_payload_log and debug_payload_log["messages"]:
            if debug_payload_log["messages"][-1].get("content"):  # Check if content exists
                debug_payload_log["messages"][-1]["content"] = (
                    str(debug_payload_log["messages"][-1]["content"])[:50] + "..."
                )
        logger.debug(f"Calling Fireworks API: {url}, Payload: {debug_payload_log}")

        try:
            for attempt in range(self.client_config.get("api_params", {}).get("max_retries", 3) + 1):
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("choices") and len(data["choices"]) > 0:
                            choice = data["choices"][0]
                            message = choice.get("message", {})

                            # 1. Check for native OpenAI-style tool_calls field
                            if message.get("tool_calls"):
                                tool_calls_data = message["tool_calls"]
                                parsed_tool_calls = []
                                for tc_data in tool_calls_data:
                                    if tc_data.get("type") == "function" and tc_data.get("function"):
                                        parsed_tool_calls.append(
                                            ToolCall(
                                                id=tc_data.get(
                                                    "id", f"call_{uuid.uuid4().hex[:8]}"
                                                ),  # Generate ID if missing
                                                type="function",
                                                function=ToolCallFunction(
                                                    name=tc_data["function"]["name"],
                                                    arguments=tc_data["function"]["arguments"],
                                                ),
                                            )
                                        )
                                if parsed_tool_calls:
                                    logger.debug(f"Parsed native tool_calls: {parsed_tool_calls}")
                                    return GenerationResult(tool_calls=parsed_tool_calls)

                            # 2. If no native tool_calls, check if content is a JSON string representing a tool call
                            # This handles the case where the LLM puts the tool call JSON into the content field.
                            # The user's log shows content like: "{\"type\": \"function\", \"name\": \"move_file\", ...}"
                            if message.get("content"):
                                content_str = message["content"]
                                try:
                                    # Attempt to parse content as JSON
                                    potential_tool_call_data = json.loads(content_str)

                                    # Check if it matches the OpenAI tool call structure (single call in content)
                                    # e.g., {"type": "function", "function": {"name": "...", "arguments": "{...}"}}
                                    # or the structure the LLM actually produced: {"type": "function", "name": "...", "parameters": {...}}

                                    parsed_tool_calls_from_content = []
                                    # Handle if content is a list of tool calls (less likely but possible)
                                    if isinstance(potential_tool_call_data, list):
                                        data_to_check = potential_tool_call_data
                                    else:  # Assume it's a single tool call object
                                        data_to_check = [potential_tool_call_data]

                                    for item in data_to_check:
                                        if isinstance(item, dict) and item.get("type") == "function":
                                            func_details = item.get("function")  # OpenAI style
                                            if func_details and "name" in func_details and "arguments" in func_details:
                                                parsed_tool_calls_from_content.append(
                                                    ToolCall(
                                                        id=item.get(
                                                            "id",
                                                            f"call_{uuid.uuid4().hex[:8]}",
                                                        ),
                                                        type="function",
                                                        function=ToolCallFunction(
                                                            name=func_details["name"],
                                                            arguments=func_details["arguments"],
                                                        ),
                                                    )
                                                )
                                                continue  # Found valid OpenAI style tool call

                                            # Check for the LLM's observed output format: {"type": "function", "name": ..., "parameters": ...}
                                            # This is slightly different from OpenAI's `function.arguments` being a string.
                                            # Here, `parameters` is an object. We need to dump it to string for `ToolCallFunction.arguments`.
                                            llm_name = item.get("name")
                                            llm_params = item.get("parameters")
                                            if llm_name and isinstance(llm_params, dict):
                                                parsed_tool_calls_from_content.append(
                                                    ToolCall(
                                                        id=item.get(
                                                            "id",
                                                            f"call_{uuid.uuid4().hex[:8]}",
                                                        ),  # Generate an ID
                                                        type="function",
                                                        function=ToolCallFunction(
                                                            name=llm_name,
                                                            arguments=json.dumps(llm_params),
                                                        ),
                                                    )
                                                )
                                                continue  # Found valid LLM-specific style tool call

                                    if parsed_tool_calls_from_content:
                                        logger.debug(
                                            f"Parsed tool_calls from content field: {parsed_tool_calls_from_content}"
                                        )
                                        return GenerationResult(tool_calls=parsed_tool_calls_from_content)

                                    # If JSON but not a recognized tool call, it's just JSON content
                                    logger.debug(
                                        "Content was JSON, but not a recognized tool call structure. Treating as text."
                                    )
                                    return GenerationResult(content=content_str.strip())

                                except json.JSONDecodeError:
                                    # Content is not JSON, so it's a regular text response
                                    logger.debug("Content is not JSON. Treating as text.")
                                    return GenerationResult(content=content_str.strip())

                        # If neither tool_calls nor parsable content that looks like a tool call
                        logger.warning(f"Fireworks API response malformed or no actionable content/tool_calls: {data}")
                        return GenerationResult()

                    # ... (rest of the error handling as before) ...
                    elif response.status == 429:  # Rate limit
                        retry_after = int(response.headers.get("Retry-After", "5"))
                        logger.warning(f"Rate limited. Retrying after {retry_after}s (attempt {attempt + 1}).")
                        await asyncio.sleep(retry_after)
                    elif response.status in [401, 403]:  # Auth errors
                        error_text = await response.text()
                        logger.error(f"Fireworks API Auth Error ({response.status}): {error_text}")
                        return GenerationResult()  # Empty result on auth error
                    elif response.status >= 500:  # Server errors
                        logger.warning(
                            f"Fireworks API Server Error ({response.status}). Retrying (attempt {attempt + 1})."
                        )
                        await asyncio.sleep(2**attempt)
                    else:  # Other client errors
                        error_text = await response.text()
                        logger.error(f"Fireworks API request failed ({response.status}): {error_text}")
                        return GenerationResult()  # Empty result
            logger.error("Max retries reached for Fireworks API call.")
            return GenerationResult()
        except aiohttp.ClientError as e:
            logger.error(f"AIOHTTP client error: {e}")
            return GenerationResult()
        except Exception as e:
            logger.error(f"Unexpected error in FireworksModelClient: {e}", exc_info=True)
            return GenerationResult()
