"""
LLM Policy Execution and Tool Calling with LiteLLM

Base classes and implementations for LLM policies that work with MCP environments.
Rewritten to use LiteLLM for unified retry logic, caching, and provider support.
"""

import logging
import os
from typing import Any, Dict, List, Literal, Optional

import litellm
from litellm import acompletion
from litellm.types.utils import ModelResponse
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.caching.caching import Cache
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache

from .base_policy import LLMBasePolicy

logger = logging.getLogger(__name__)


class LiteLLMPolicy(LLMBasePolicy):
    """
    Unified LiteLLM policy implementation that works with ANY MCP environment via tool calling.

    Supports OpenAI, Anthropic, Fireworks AI
    Includes built-in retry logic and caching.
    NO environment-specific logic - everything comes from MCP tools and dataset prompts.
    """

    def __init__(
        self,
        model_id: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        max_tools_per_turn: Optional[int] = None,
        base_url: Optional[str] = None,
        # LiteLLM-specific parameters
        use_caching: bool = True,
        cache_type: Literal["memory", "redis", "dual", "s3", "disk"] = "memory",
        redis_url: Optional[str] = None,
        num_retries: int = 8,
        retry_strategy: Literal["exponential_backoff_retry", "constant_retry"] = "exponential_backoff_retry",
        **kwargs,
    ):
        """
        Initialize LiteLLM policy with caching and retry logic.

        Args:
            model_id: Model identifier (e.g., "gpt-4o", "anthropic/claude-3-5-sonnet-20241022", "fireworks_ai/accounts/fireworks/models/llama-v3p2-3b-instruct")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate per request
            max_tools_per_turn: Maximum number of tool calls per turn
            use_caching: Enable response caching
            cache_type: Type of cache (literal: "memory", "redis", "dual", "s3", "disk")
            redis_url: Redis URL for distributed caching
            num_retries: Number of retries for failed requests
            retry_strategy: Retry strategy (literal: "exponential_backoff_retry", "constant_retry")
        """
        super().__init__(model_id, temperature, max_tokens, max_tools_per_turn, base_url, **kwargs)

        self.num_retries = num_retries
        self.retry_strategy = retry_strategy

        # Store additional API parameters from kwargs
        self.additional_params = kwargs

        # Only initialize LiteLLM in live mode (not in playback mode)
        if not self._is_playback:
            self._setup_litellm_caching(use_caching, cache_type, redis_url)
            logger.info(f"✅ Initialized LiteLLM policy: {self.model_id}")
        else:
            logger.info("🎬 Playback mode: Skipping LiteLLM initialization for performance")

    def _setup_litellm_caching(
        self, use_caching: bool, cache_type: Literal["memory", "redis", "dual", "s3", "disk"], redis_url: Optional[str]
    ):
        """Setup LiteLLM caching based on configuration."""
        if not use_caching:
            litellm.cache = None
            return

        try:
            if cache_type == "memory":
                litellm.cache = Cache()
                logger.info("🗄️ Initialized in-memory caching")

            elif cache_type == "redis":
                if redis_url:
                    redis_cache = RedisCache(url=redis_url)
                else:
                    redis_cache = RedisCache()
                litellm.cache = redis_cache
                logger.info("🗄️ Initialized Redis caching")

            elif cache_type == "dual":
                # Best performance: in-memory + Redis
                # TODO: further optimize by using this, but requires
                in_memory_cache = InMemoryCache()
                if redis_url:
                    redis_cache = RedisCache(url=redis_url)
                else:
                    redis_cache = RedisCache()
                dual_cache = DualCache(in_memory_cache=in_memory_cache, redis_cache=redis_cache)
                litellm.cache = dual_cache
                logger.info("🗄️ Initialized dual caching (memory + Redis)")

            elif cache_type == "disk":
                from litellm.caching.disk_cache import DiskCache

                litellm.cache = DiskCache()
                logger.info("🗄️ Initialized disk caching")

            elif cache_type == "s3":
                try:
                    from litellm.caching.s3_cache import S3Cache

                    # Some versions require positional or named 's3_bucket_name'
                    s3_bucket_name = os.getenv("LITELLM_S3_BUCKET")
                    if not s3_bucket_name:
                        raise ValueError("Missing LITELLM_S3_BUCKET for S3 cache")
                    # Use explicit arg name expected by basedpyright
                    litellm.cache = S3Cache(s3_bucket_name=s3_bucket_name)
                    logger.info("🗄️ Initialized S3 caching for bucket %s", s3_bucket_name)
                except Exception as e:
                    logger.warning(f"Failed to initialize S3 cache ({e}); falling back to in-memory cache")
                    litellm.cache = Cache()

        except Exception as e:
            logger.warning(f"Failed to setup {cache_type} caching: {e}. Falling back to in-memory cache.")
            litellm.cache = Cache()

    def _clean_messages_for_api(self, messages: List[Dict]) -> List[Dict]:
        """
        Clean messages by keeping only OpenAI API compatible fields.
        LiteLLM handles provider-specific message format conversion automatically.

        Args:
            messages: Conversation messages with potential metadata

        Returns:
            Clean messages with only OpenAI API compatible fields
        """
        # Standard OpenAI message fields
        allowed_fields = {"role", "content", "tool_calls", "tool_call_id", "name"}

        clean_messages = []
        for msg in messages:
            # Only keep allowed fields
            clean_msg = {k: v for k, v in msg.items() if k in allowed_fields}
            clean_messages.append(clean_msg)
        return clean_messages

    async def _make_llm_call(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Make an LLM API call with retry logic and caching.

        Args:
            messages: Conversation messages (may contain metadata)
            tools: Available tools in OpenAI format (LiteLLM converts automatically)

        Returns:
            API response in OpenAI format
        """
        # Clean messages by removing metadata before sending to API
        clean_messages = self._clean_messages_for_api(messages)

        # Prepare request parameters
        request_params: Dict[str, Any] = {
            "messages": clean_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "caching": True,
            "num_retries": self.num_retries,
            "retry_strategy": self.retry_strategy,
            "base_url": self.base_url,
        }

        # Add additional parameters from kwargs (like reasoning_effort)
        if self.additional_params:
            request_params.update(self.additional_params)

            # Tell LiteLLM to allow reasoning_effort if it's present
            if "reasoning_effort" in self.additional_params:
                request_params["allowed_openai_params"] = ["reasoning_effort"]

        # Add tools if provided
        if tools:
            request_params["tools"] = tools

        try:
            if request_params.get("stream") is True:
                chunks = []
                stream = await acompletion(model=self.model_id, **request_params)

                assert isinstance(stream, CustomStreamWrapper), "Stream should be a CustomStreamWrapper"

                async for chunk in stream:  # pyright: ignore[reportGeneralTypeIssues]
                    chunks.append(chunk)
                response = litellm.stream_chunk_builder(chunks, messages)
            else:
                response = await acompletion(model=self.model_id, **request_params)

            assert response is not None, "Response is None"
            assert isinstance(response, ModelResponse), "Response should be ModelResponse"

            # Log cache hit/miss for monitoring
            hidden = getattr(response, "_hidden_params", {})
            cache_hit = hidden.get("cache_hit", False) if isinstance(hidden, dict) else False
            if cache_hit:
                logger.debug(f"🎯 Cache hit for model: {self.model_id}")
            else:
                logger.debug(f"🔄 API call for model: {self.model_id}")

            # LiteLLM already returns OpenAI-compatible format
            return {
                "choices": [
                    {
                        "message": {
                            "role": getattr(getattr(response.choices[0], "message", object()), "role", "assistant"),
                            "content": getattr(getattr(response.choices[0], "message", object()), "content", None),
                            "tool_calls": (
                                [
                                    {
                                        "id": getattr(tc, "id", None),
                                        "type": getattr(tc, "type", "function"),
                                        "function": {
                                            "name": getattr(getattr(tc, "function", None), "name", "tool"),
                                            "arguments": getattr(getattr(tc, "function", None), "arguments", "{}"),
                                        },
                                    }
                                    for tc in (
                                        getattr(getattr(response.choices[0], "message", object()), "tool_calls", [])
                                        or []
                                    )
                                ]
                                if getattr(getattr(response.choices[0], "message", object()), "tool_calls", None)
                                else []
                            ),
                        },
                        "finish_reason": getattr(response.choices[0], "finish_reason", None),
                    }
                ],
                "usage": {
                    "prompt_tokens": getattr(getattr(response, "usage", {}), "prompt_tokens", 0),
                    "completion_tokens": getattr(getattr(response, "usage", {}), "completion_tokens", 0),
                    "total_tokens": getattr(getattr(response, "usage", {}), "total_tokens", 0),
                },
            }

        except Exception as e:
            logger.error(
                f"❌ LLM call FAILED after all retries ({self.num_retries}) for model {self.model_id}: {type(e).__name__}: {e}"
            )
            raise

    def _convert_mcp_tools_to_llm_format(self, mcp_tools: List[Dict]) -> List[Dict]:
        """
        Convert MCP tool schemas to OpenAI function calling format.
        LiteLLM handles provider-specific format conversion automatically.

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


class OpenAIPolicy(LiteLLMPolicy):
    """OpenAI-specific policy using LiteLLM."""

    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id=model_id, **kwargs)


class AnthropicPolicy(LiteLLMPolicy):
    """Anthropic-specific policy using LiteLLM."""

    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id=model_id, **kwargs)


class FireworksPolicy(LiteLLMPolicy):
    """Fireworks AI-specific policy using LiteLLM."""

    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id=f"fireworks_ai/{model_id}", **kwargs)


class LocalPolicy(LiteLLMPolicy):
    """Local policy using LiteLLM for local model endpoints."""

    def __init__(self, model_id: str, base_url: str, **kwargs):
        """Initialize LocalPolicy for local model endpoints."""
        super().__init__(model_id=model_id, base_url=base_url, **kwargs)


# Export the policies
__all__ = [
    "LiteLLMPolicy",
    "OpenAIPolicy",
    "AnthropicPolicy",
    "FireworksPolicy",
    "LocalPolicy",
]
