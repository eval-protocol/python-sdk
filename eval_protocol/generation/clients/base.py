from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import aiohttp
from omegaconf import DictConfig
from pydantic import BaseModel, Field


class ToolCallFunction(BaseModel):
    name: str
    arguments: str  # Should be a JSON string


class ToolCall(BaseModel):
    id: str
    type: str = "function"  # OpenAI default is "function"
    function: ToolCallFunction


class GenerationResult(BaseModel):
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None

    # Add a validator to ensure that not both content and tool_calls are None,
    # and not both are set, if that's a desired constraint.
    # For now, allowing flexibility.


class ModelClient(ABC):
    """Abstract base class for model clients."""

    def __init__(self, client_config: DictConfig, api_key: Optional[str] = None):
        self.model_name = client_config.get("model_name", "unknown")
        self.temperature = client_config.get("temperature", 0.0)
        self.top_p = client_config.get("top_p", 1.0)
        self.top_k = client_config.get("top_k", None)  # Optional, None if not used
        self.min_p = client_config.get("min_p", None)  # Optional, None if not used
        self.max_tokens = client_config.get("max_tokens", 1024)
        self.reasoning_effort = client_config.get("reasoning_effort", None)  # Optional
        self.api_key = api_key
        self.client_config = client_config  # Store the raw config for other params

    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        session: aiohttp.ClientSession,
        tools: Optional[List[Dict[str, Any]]] = None,  # For OpenAI-style tool definitions
        **kwargs: Any,  # For additional model-specific parameters
    ) -> GenerationResult:
        """
        Generates a response from the model.

        Args:
            messages: A list of messages comprising the conversation history.
            session: An aiohttp.ClientSession for making HTTP requests.
            tools: Optional list of tool definitions to provide to the model.
            **kwargs: Additional keyword arguments for model-specific parameters.

        Returns:
            A GenerationResult object containing either text content or tool calls.
        """
        pass

    @property
    def name(self) -> str:
        return self.model_name
