"""
Models for the LiteLLM Metadata Proxy.
"""

from pydantic import BaseModel
from typing import Optional, List, Any, Dict, TypeAlias, Callable
from fastapi import Request


ChatRequestHook: TypeAlias = Callable[[Dict[str, Any], Request, "ChatParams"], tuple[Dict[str, Any], "ChatParams"]]
TracesRequestHook: TypeAlias = Callable[[Request, "TracesParams"], "TracesParams"]


class AccountInfo(BaseModel):
    """Account information returned from authentication."""

    account_id: str


class ChatParams(BaseModel):
    """Typed container for chat completion URL path parameters."""

    project_id: Optional[str] = None
    rollout_id: Optional[str] = None
    invocation_id: Optional[str] = None
    experiment_id: Optional[str] = None
    run_id: Optional[str] = None
    row_id: Optional[str] = None
    encoded_base_url: Optional[str] = None


class TracesParams(BaseModel):
    """Typed container for traces query parameters and controls."""

    tags: Optional[List[str]] = None
    project_id: Optional[str] = None
    limit: int = 100
    sample_size: Optional[int] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    name: Optional[str] = None
    environment: Optional[str] = None
    version: Optional[str] = None
    release: Optional[str] = None
    fields: Optional[str] = None
    hours_back: Optional[int] = None
    from_timestamp: Optional[str] = None
    to_timestamp: Optional[str] = None
    sleep_between_gets: float = 2.5
    max_retries: int = 3


class ProxyConfig(BaseModel):
    """Configuration model for the LiteLLM Metadata Proxy"""

    litellm_url: str
    request_timeout: float = 300.0
    langfuse_host: str
    langfuse_keys: Dict[str, Dict[str, str]]
    default_project_id: str
    preprocess_chat_request: Optional[ChatRequestHook] = None
    preprocess_traces_request: Optional[TracesRequestHook] = None


class ObservationResponse(BaseModel):
    """Response model for a single observation within a trace"""

    id: str
    type: Optional[str] = None
    name: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    input: Optional[Any] = None
    output: Optional[Any] = None
    parent_observation_id: Optional[str] = None


class TraceResponse(BaseModel):
    """Response model for a single trace"""

    id: str
    name: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    tags: List[str] = []
    timestamp: Optional[str] = None
    input: Optional[Any] = None
    output: Optional[Any] = None
    metadata: Optional[Any] = None
    observations: List[ObservationResponse] = []


class LangfuseTracesResponse(BaseModel):
    """Response model for the /traces endpoint"""

    project_id: str
    total_traces: int
    traces: List[TraceResponse]
