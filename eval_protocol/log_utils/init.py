import logging
import os
from typing import Optional

from eval_protocol.log_utils.fireworks_tracing_http_handler import (
    FireworksTracingHttpHandler,
)
from eval_protocol.log_utils.elasticsearch_direct_http_handler import (
    ElasticsearchDirectHttpHandler,
)
from eval_protocol.log_utils.rollout_context import ContextRolloutIdFilter
from eval_protocol.types.remote_rollout_processor import ElasticsearchConfig


_INITIALIZED = False


def _get_env(name: str) -> Optional[str]:
    val = os.getenv(name)
    return val if val and val.strip() else None


def init_external_logging_from_env() -> None:
    """
    Initialize external logging sinks (Fireworks tracing, optional Elasticsearch) from env vars.

    Idempotent: safe to call multiple times.

    Environment variables:
      - FW_TRACING_GATEWAY_BASE_URL: enable Fireworks tracing handler when set
      - EP_ELASTICSEARCH_URL, EP_ELASTICSEARCH_API_KEY, EP_ELASTICSEARCH_INDEX: enable ES when all set
    """
    global _INITIALIZED
    if _INITIALIZED:
        return

    root_logger = logging.getLogger()

    # Ensure we do not add duplicate handlers if already present
    existing_handler_types = {type(h).__name__ for h in root_logger.handlers}

    # Fireworks tracing: prefer if FIREWORKS_API_KEY is present; default base URL if not provided
    fw_key = _get_env("FIREWORKS_API_KEY")
    # Allow remote validation gateway to act as tracing base when provided
    fw_url = _get_env("FW_TRACING_GATEWAY_BASE_URL") or _get_env("GATEWAY_URL") or "https://tracing.fireworks.ai"
    if fw_key and "FireworksTracingHttpHandler" not in existing_handler_types:
        fw_handler = FireworksTracingHttpHandler(gateway_base_url=fw_url)
        fw_handler.setLevel(logging.INFO)
        fw_handler.addFilter(ContextRolloutIdFilter())
        root_logger.addHandler(fw_handler)

    # Elasticsearch
    es_url = _get_env("EP_ELASTICSEARCH_URL")
    es_api_key = _get_env("EP_ELASTICSEARCH_API_KEY")
    es_index = _get_env("EP_ELASTICSEARCH_INDEX")
    if (
        not fw_key
        and es_url
        and es_api_key
        and es_index
        and "ElasticsearchDirectHttpHandler" not in existing_handler_types
    ):
        es_config = ElasticsearchConfig(url=es_url, api_key=es_api_key, index_name=es_index)
        es_handler = ElasticsearchDirectHttpHandler(elasticsearch_config=es_config)
        es_handler.setLevel(logging.INFO)
        es_handler.addFilter(ContextRolloutIdFilter())
        root_logger.addHandler(es_handler)

    _INITIALIZED = True
