"""
Request and response models for remote rollout processor servers.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from eval_protocol.models import Message, Status


class RolloutMetadata(BaseModel):
    """Metadata for rollout execution."""

    invocation_id: str
    experiment_id: str
    rollout_id: str
    run_id: str
    row_id: str


class InitRequest(BaseModel):
    """Request model for POST /init endpoint."""

    model: str
    messages: Optional[List[Message]] = None
    tools: Optional[List[Dict[str, Any]]] = None

    model_base_url: Optional[str] = None
    """
    A Base URL that the remote server can use to make LLM calls. This is useful
    to configure on the eval-protocol side for flexibility in
    development/traning.
    """

    metadata: RolloutMetadata


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
