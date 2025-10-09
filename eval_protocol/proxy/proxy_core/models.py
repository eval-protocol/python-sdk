"""
Models for the LiteLLM Metadata Proxy.
"""

from pydantic import BaseModel
from typing import Optional, List, Any, Dict


class ProxyConfig(BaseModel):
    """Configuration model for the LiteLLM Metadata Proxy"""

    litellm_url: str
    request_timeout: float = 300.0
    langfuse_keys: Dict[str, Dict[str, str]]
    default_project_id: str


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
