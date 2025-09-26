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

    model: str
    messages: List[Message] = Field(min_length=1)
    tools: Optional[List[Dict[str, Any]]] = None

    model_base_url: Optional[str] = None
    """
    A Base URL that the remote server can use to make LLM calls. This is useful
    to configure on the eval-protocol side for flexibility in
    development/training.

    The RemoteRolloutProcessor automatically enhances this URL by attaching
    rollout metadata as query parameters (rollout_id, invocation_id, experiment_id,
    run_id, row_id) before sending it to the remote server. This passes along
    rollout context to the remote server for use in LLM API calls.

    Example:
        If model_base_url is "https://api.openai.com/v1", it will be enhanced to:
        "https://api.openai.com/v1?rollout_id=abc123&invocation_id=def456&experiment_id=ghi789&run_id=jkl012&row_id=mno345"
    """

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
