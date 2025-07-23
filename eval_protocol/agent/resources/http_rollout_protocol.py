"""
HTTP Rollout Protocol - Standardized types for HTTP rollout communication.

This module defines the standard request/response models for HTTP rollout servers
and clients, ensuring consistent communication across different implementations.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class StartEpisodeRequest(BaseModel):
    """Request to start a new episode."""

    class Config:
        extra = "allow"  # Allow arbitrary extra fields (like seed)


class StartEpisodeResponse(BaseModel):
    """Response from starting a new episode."""

    episode_id: str
    observation: Dict[str, Any]


class StepRequest(BaseModel):
    """Request to take a step in the environment."""

    episode_id: str
    action: Any  # Can be int, str, dict, etc. depending on environment


class StepResponse(BaseModel):
    """Response from taking a step in the environment."""

    observation: Dict[str, Any]
    is_done: bool
    info: Optional[Dict[str, Any]] = None


class EndEpisodeRequest(BaseModel):
    """Request to end an episode."""

    episode_id: str


class EndEpisodeResponse(BaseModel):
    """Response from ending an episode."""

    message: str


class HealthResponse(BaseModel):
    """Response from health check endpoint."""

    status: str
    game: Optional[str] = None
    version: Optional[str] = None


class HttpRolloutConfig(BaseModel):
    """Configuration for HTTP rollout resource."""

    base_url: str
    start_episode_endpoint: str = "/start_episode"
    step_endpoint: str = "/step"
    end_episode_endpoint: str = "/end_episode"
    health_endpoint: str = "/health"
    timeout: float = 30.0
    max_retries: int = 3


# Observation structure for game environments
class GameObservation(BaseModel):
    """Standard observation structure for game environments."""

    position: Optional[List[int]] = None
    current_cell: Optional[str] = None
    done: bool = False
    won: bool = False
    visual: Optional[str] = None
    message: Optional[str] = None
    step_count: Optional[int] = None
    max_steps: Optional[int] = None
