"""
Consolidated Fireworks client factory.

This module provides a single point of instantiation for the Fireworks SDK client,
ensuring consistent handling of environment variables and configuration across the
eval_protocol codebase.

Environment variables:
    FIREWORKS_API_KEY: API key for authentication (required)
    FIREWORKS_ACCOUNT_ID: Account ID (optional, can be derived from API key)
    FIREWORKS_API_BASE: Base URL for the API (default: https://api.fireworks.ai)
    FIREWORKS_EXTRA_HEADERS: JSON-encoded extra headers to include in requests
        Example: '{"X-Custom-Header": "value", "X-Another": "another-value"}'
"""

import json
import logging
import os
from typing import Mapping, Optional

from fireworks import Fireworks

from eval_protocol.auth import (
    get_fireworks_account_id,
    get_fireworks_api_base,
    get_fireworks_api_key,
)

logger = logging.getLogger(__name__)


def get_fireworks_extra_headers() -> Optional[Mapping[str, str]]:
    """
    Retrieves extra headers from the FIREWORKS_EXTRA_HEADERS environment variable.

    The value should be a JSON-encoded object mapping header names to values.
    Example: '{"X-Custom-Header": "value"}'

    Returns:
            A mapping of header names to values if set and valid, otherwise None.
    """
    extra_headers_str = os.environ.get("FIREWORKS_EXTRA_HEADERS")
    if not extra_headers_str or not extra_headers_str.strip():
        return None

    try:
        headers = json.loads(extra_headers_str)
        if not isinstance(headers, dict):
            logger.warning(
                "FIREWORKS_EXTRA_HEADERS must be a JSON object, got %s. Ignoring.",
                type(headers).__name__,
            )
            return None
        # Validate all keys and values are strings
        for k, v in headers.items():
            if not isinstance(k, str) or not isinstance(v, str):
                logger.warning(
                    "FIREWORKS_EXTRA_HEADERS contains non-string key or value: %s=%s. Ignoring all extra headers.",
                    k,
                    v,
                )
                return None
        logger.debug("Using FIREWORKS_EXTRA_HEADERS: %s", list(headers.keys()))
        return headers
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse FIREWORKS_EXTRA_HEADERS as JSON: %s. Ignoring.", e)
        return None


def create_fireworks_client(
    *,
    api_key: Optional[str] = None,
    account_id: Optional[str] = None,
    base_url: Optional[str] = None,
    extra_headers: Optional[Mapping[str, str]] = None,
) -> Fireworks:
    """
    Create a Fireworks client with consistent configuration.

    This factory function centralizes the logic for creating Fireworks clients,
    ensuring that environment variables are handled consistently across the codebase.

    Resolution order for each parameter:
            1. Explicit argument passed to this function
            2. Environment variable (via auth module helpers)
            3. SDK defaults (for base_url only)

    Args:
            api_key: Fireworks API key. If not provided, resolves from FIREWORKS_API_KEY.
            account_id: Fireworks account ID. If not provided, resolves from FIREWORKS_ACCOUNT_ID
                    or derives from the API key via the verifyApiKey endpoint.
            base_url: Base URL for the Fireworks API. If not provided, resolves from
                    FIREWORKS_API_BASE or defaults to https://api.fireworks.ai.
            extra_headers: Additional headers to include in all requests. If not provided,
                    resolves from FIREWORKS_EXTRA_HEADERS environment variable (JSON-encoded).

    Returns:
            A configured Fireworks client instance.

    Raises:
            fireworks.FireworksError: If api_key is not provided and FIREWORKS_API_KEY
                    environment variable is not set.
    """
    # Resolve parameters from environment if not explicitly provided
    resolved_api_key = api_key or get_fireworks_api_key()
    resolved_account_id = account_id or get_fireworks_account_id()
    resolved_base_url = base_url or get_fireworks_api_base()

    # Merge extra headers: env var headers first, then explicit headers override
    env_extra_headers = get_fireworks_extra_headers()
    merged_headers: Optional[Mapping[str, str]] = None
    if env_extra_headers or extra_headers:
        merged = {}
        if env_extra_headers:
            merged.update(env_extra_headers)
        if extra_headers:
            merged.update(extra_headers)
        merged_headers = merged if merged else None

    logger.debug(
        "Creating Fireworks client: base_url=%s, account_id=%s, extra_headers=%s",
        resolved_base_url,
        resolved_account_id,
        list(merged_headers.keys()) if merged_headers else None,
    )

    return Fireworks(
        api_key=resolved_api_key,
        account_id=resolved_account_id,
        base_url=resolved_base_url,
        default_headers=merged_headers,
    )
