"""
LiteLLM client - handles all LLM calls directly via LiteLLM SDK with Langfuse OTEL integration.
"""

import json
import base64
import logging
from uuid6 import uuid7
from fastapi import Request, Response, HTTPException
from fastapi.responses import StreamingResponse
import redis
import openai
from litellm import acompletion

from .redis_utils import register_insertion_id
from .models import ProxyConfig, ChatParams

logger = logging.getLogger(__name__)


async def handle_chat_completion(
    config: ProxyConfig,
    redis_client: redis.Redis,
    request: Request,
    params: ChatParams,
) -> Response:
    """
    Handle chat completion requests using LiteLLM SDK directly with Langfuse OTEL.

    If metadata IDs (rollout_id, etc.) are provided, they'll be added as tags
    and the assistant message count will be tracked in Redis.

    If encoded_base_url is provided, it will be decoded and used as api_base.
    """
    body = await request.body()
    data = json.loads(body) if body else {}

    if config.preprocess_chat_request:
        data, params = config.preprocess_chat_request(data, request, params)

    project_id = params.project_id
    rollout_id = params.rollout_id
    invocation_id = params.invocation_id
    experiment_id = params.experiment_id
    run_id = params.run_id
    row_id = params.row_id
    encoded_base_url = params.encoded_base_url

    # Use default project if not specified
    if project_id is None:
        project_id = config.default_project_id

    # Decode and add base_url if provided
    if encoded_base_url:
        try:
            decoded_bytes = base64.urlsafe_b64decode(encoded_base_url)
            data["base_url"] = decoded_bytes.decode("utf-8")
            logger.debug(f"Decoded base_url: {data['base_url']}")
        except Exception as e:
            logger.error(f"Failed to decode base_url: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid encoded_base_url: {str(e)}")

    # Extract API key from Authorization header and add to data
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        data["api_key"] = auth_header.replace("Bearer ", "").strip()

    # Build metadata with tags for Langfuse
    insertion_id = None
    metadata = data.pop("metadata", {}) or {}
    tags = list(metadata.pop("tags", []) or [])

    if rollout_id is not None:
        insertion_id = str(uuid7())
        tags.extend(
            [
                f"rollout_id:{rollout_id}",
                f"insertion_id:{insertion_id}",
                f"invocation_id:{invocation_id}",
                f"experiment_id:{experiment_id}",
                f"run_id:{run_id}",
                f"row_id:{row_id}",
            ]
        )

    # Build Langfuse metadata (tags + user if present)
    # Convert user_id (from preprocess hook) to trace_user_id for Langfuse
    user_id = metadata.pop("user_id", None) or data.get("user")
    litellm_metadata = {"tags": tags, **metadata}
    if user_id:
        litellm_metadata["trace_user_id"] = user_id

    langfuse_keys = config.langfuse_keys[project_id]

    # Check if streaming is requested
    is_streaming = data.get("stream", False)

    # Pop fields that we pass explicitly to avoid duplicate kwarg errors
    request_timeout = data.pop("timeout", None) or config.request_timeout
    data.pop("langfuse_public_key", None)
    data.pop("langfuse_secret_key", None)

    try:
        # Make the completion call - pass all params through
        # Note: langfuse_host is set via LANGFUSE_HOST env var at startup; OTEL doesn't support per-request host override
        response = await acompletion(
            **data,
            metadata=litellm_metadata,
            timeout=request_timeout,
            langfuse_public_key=langfuse_keys["public_key"],
            langfuse_secret_key=langfuse_keys["secret_key"],
        )

        if is_streaming:
            # For streaming, return a StreamingResponse with SSE format
            # Register insertion_id only after stream completes successfully
            async def stream_generator():
                async for chunk in response:  # type: ignore[union-attr]
                    yield f"data: {chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                # Stream completed successfully - now register
                if insertion_id is not None and rollout_id is not None:
                    register_insertion_id(redis_client, rollout_id, insertion_id)

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            # Non-streaming: register insertion_id on success
            if insertion_id is not None and rollout_id is not None:
                register_insertion_id(redis_client, rollout_id, insertion_id)

            return Response(
                content=response.model_dump_json(),
                status_code=200,
                media_type="application/json",
            )

    except HTTPException:
        raise
    except openai.APIError as e:
        # Convert to HTTPException and let FastAPI handle it
        raise HTTPException(
            status_code=getattr(e, "status_code", 500),
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
