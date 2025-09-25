"""
Request and response models for remote rollout processor servers.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from eval_protocol.models import Message


class RolloutMetadata(BaseModel):
    """Metadata for rollout execution."""

    invocation_id: str
    experiment_id: str
    rollout_id: str
    run_id: str
    row_id: str


class InitRequest(BaseModel):
    """Request model for POST /init endpoint."""

    rollout_id: str
    model: str
    messages: List[Message] = Field(min_length=1)
    tools: Optional[List[Dict[str, Any]]] = None
    metadata: RolloutMetadata


class StatusResponse(BaseModel):
    """Response model for GET /status endpoint."""

    terminated: bool
    info: Optional[Dict[str, Any]] = None


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
