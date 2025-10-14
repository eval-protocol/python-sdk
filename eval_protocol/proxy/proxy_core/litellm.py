"""
LiteLLM client - handles all communication with LiteLLM service.
"""

import json
import base64
import httpx
import logging
from uuid6 import uuid7
from fastapi import Request, Response, HTTPException
import redis
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
    Handle chat completion requests and forward to LiteLLM.

    If metadata IDs (rollout_id, etc.) are provided, they'll be added as tags
    and the assistant message count will be tracked in Redis.

    If encoded_base_url is provided, it will be decoded and added to the request.
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
            # Decode from URL-safe base64
            decoded_bytes = base64.urlsafe_b64decode(encoded_base_url)
            base_url = decoded_bytes.decode("utf-8")
            data["base_url"] = base_url
            logger.debug(f"Decoded base_url: {base_url}")
        except Exception as e:
            logger.error(f"Failed to decode base_url: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid encoded_base_url: {str(e)}")

    # Extract API key from Authorization header and inject into request body
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        api_key = auth_header.replace("Bearer ", "").strip()
        # Only inject API key if model is a Fireworks model
        model = data.get("model")
        if model and isinstance(model, str) and model.startswith("fireworks_ai"):
            data["api_key"] = api_key

    # If metadata IDs are provided, add them as tags
    insertion_id = None
    if rollout_id is not None:
        insertion_id = str(uuid7())

        if "metadata" not in data:
            data["metadata"] = {}
        if "tags" not in data["metadata"]:
            data["metadata"]["tags"] = []

        # Add extracted IDs as tags
        data["metadata"]["tags"].extend(
            [
                f"rollout_id:{rollout_id}",
                f"insertion_id:{insertion_id}",
                f"invocation_id:{invocation_id}",
                f"experiment_id:{experiment_id}",
                f"run_id:{run_id}",
                f"row_id:{row_id}",
            ]
        )

    # Add Langfuse configuration
    data["langfuse_public_key"] = config.langfuse_keys[project_id]["public_key"]
    data["langfuse_secret_key"] = config.langfuse_keys[project_id]["secret_key"]
    data["langfuse_host"] = config.langfuse_host

    # Forward to LiteLLM's standard /chat/completions endpoint
    # Set longer timeout for LLM API calls (LLMs can be slow)
    timeout = httpx.Timeout(config.request_timeout)
    async with httpx.AsyncClient(timeout=timeout) as client:
        # Copy headers from original request but exclude content-length (httpx will set it correctly)
        headers = dict(request.headers)
        headers.pop("host", None)
        headers.pop("content-length", None)  # Let httpx calculate the correct length
        headers["content-type"] = "application/json"

        # Forward to LiteLLM
        litellm_url = f"{config.litellm_url}/chat/completions"

        response = await client.post(
            litellm_url,
            json=data,  # httpx will serialize and set correct Content-Length
            headers=headers,
        )

        # Register insertion_id in Redis only on successful response
        if response.status_code == 200 and insertion_id is not None and rollout_id is not None:
            register_insertion_id(redis_client, rollout_id, insertion_id)

        # Return the response
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers),
        )


async def proxy_to_litellm(config: ProxyConfig, path: str, request: Request) -> Response:
    """
    Catch-all proxy: Forward any request to LiteLLM, extracting API key from Authorization header.
    """
    # Set longer timeout for LLM API calls (LLMs can be slow)
    timeout = httpx.Timeout(config.request_timeout)
    async with httpx.AsyncClient(timeout=timeout) as client:
        # Copy headers
        headers = dict(request.headers)
        headers.pop("host", None)
        headers.pop("content-length", None)

        # Get body
        body = await request.body()

        # Pass through API key from Authorization header
        if request.method in ["POST", "PUT", "PATCH"] and body:
            try:
                data = json.loads(body)

                auth_header = request.headers.get("authorization", "")
                if auth_header.startswith("Bearer "):
                    api_key = auth_header.replace("Bearer ", "").strip()
                    data["api_key"] = api_key

                # Re-serialize
                body = json.dumps(data).encode()
            except json.JSONDecodeError:
                pass

        # Forward to LiteLLM
        litellm_url = f"{config.litellm_url}/{path}"

        response = await client.request(
            method=request.method,
            url=litellm_url,
            headers=headers,
            content=body,
        )

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers),
        )
