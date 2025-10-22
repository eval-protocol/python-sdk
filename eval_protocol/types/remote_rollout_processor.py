"""
Request and response models for remote rollout processor servers.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from urllib.parse import urlparse
from eval_protocol.models import Message, Status


class ElasticsearchConfig(BaseModel):
    """
    Configuration for Elasticsearch.
    """

    url: str
    api_key: str
    index_name: str

    @property
    def verify_ssl(self) -> bool:
        """Infer verify_ssl from URL scheme."""
        parsed_url = urlparse(self.url)
        return parsed_url.scheme == "https"


class RolloutMetadata(BaseModel):
    """Metadata for rollout execution."""

    invocation_id: str
    experiment_id: str
    rollout_id: str
    run_id: str
    row_id: str


class DataLoaderConfig(BaseModel):
    """Configuration passed to output_data_loader functions."""

    rollout_id: str
    model_base_url: Optional[str] = None


class InitRequest(BaseModel):
    """Request model for POST /init endpoint."""

    completion_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Completion parameters including model and optional model_kwargs, temperature, etc.",
    )
    messages: Optional[List[Message]] = None
    tools: Optional[List[Dict[str, Any]]] = None

    model_base_url: Optional[str] = None
    """
    A Base URL that the remote server can use to make LLM calls. This is useful
    to configure on the eval-protocol side for flexibility in
    development/traning.
    """

    metadata: RolloutMetadata
    api_key: Optional[str] = None


class StatusResponse(BaseModel):
    """Response model for GET /status endpoint."""

    terminated: bool
    info: Optional[Dict[str, Any]] = None

    status: Optional[Status] = None
    """
    Optional status indicator for the rollout to be used by eval-protocol. This
    is useful to distinguish between successful and failed rollouts.
    """


def create_langfuse_config_tags(init_request: InitRequest) -> List[str]:
    """Create Langfuse tags from InitRequest metadata."""
    metadata = init_request.metadata
    return [
        f"invocation_id:{metadata.invocation_id}",
        f"experiment_id:{metadata.experiment_id}",
        f"rollout_id:{metadata.rollout_id}",
        f"run_id:{metadata.run_id}",
        f"row_id:{metadata.row_id}",
    ]
