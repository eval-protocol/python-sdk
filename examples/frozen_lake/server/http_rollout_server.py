"""
HTTP rollout server for Frozen Lake game.

This server implements the standard HTTP rollout protocol using the reward-kit
library's standardized types for consistent client/server communication.
"""

import os
import sys
import uuid
from typing import Dict

from fastapi import FastAPI, HTTPException

# Add parent directory to path to import gymnasium frozen lake server
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gymnasium_frozen_lake_server import GymnasiumFrozenLakeGame as FrozenLakeGame

# Additional models for responses
from pydantic import BaseModel

# Import standardized HTTP rollout protocol types from reward-kit
from eval_protocol.agent.resources import (
    EndEpisodeRequest,
    EndEpisodeResponse,
    HealthResponse,
    StartEpisodeRequest,
    StartEpisodeResponse,
    StepRequest,
    StepResponse,
)

# FastAPI app
app = FastAPI(title="Frozen Lake HTTP Rollout Server")

# Store active episodes
episodes: Dict[str, FrozenLakeGame] = {}


@app.post("/start_episode", response_model=StartEpisodeResponse)
async def start_episode(
    request: StartEpisodeRequest = StartEpisodeRequest(),
) -> StartEpisodeResponse:
    """Start a new episode of the Frozen Lake game."""
    episode_id = str(uuid.uuid4())

    # Extract request data to pass to the game constructor
    request_data = request.dict() if hasattr(request, "dict") else request.model_dump()

    # Create Gymnasium-based game with configuration from request
    # Default values maintained for backward compatibility
    game_config = {
        "map_name": "4x4",
        "is_slippery": True,  # Enable stochastic behavior to demonstrate seed effect
        "render_mode": None,
    }
    # Override with any values from the request (like seed)
    game_config.update(request_data)

    game = FrozenLakeGame(**game_config)
    observation = game.reset()
    episodes[episode_id] = game

    return StartEpisodeResponse(episode_id=episode_id, observation=observation)


@app.post("/step", response_model=StepResponse)
async def step(req: StepRequest) -> StepResponse:
    """Take a step in the specified episode."""
    if req.episode_id not in episodes:
        raise HTTPException(status_code=404, detail="Episode not found")

    game = episodes[req.episode_id]
    observation, is_done = game.step(req.action)

    return StepResponse(observation=observation, is_done=is_done)


@app.post("/end_episode", response_model=EndEpisodeResponse)
async def end_episode(req: EndEpisodeRequest) -> EndEpisodeResponse:
    """End the specified episode."""
    if req.episode_id not in episodes:
        raise HTTPException(status_code=404, detail="Episode not found")

    del episodes[req.episode_id]
    return EndEpisodeResponse(message=f"Episode {req.episode_id} ended successfully")


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="healthy", game="frozen_lake_gymnasium")


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Frozen Lake HTTP Rollout Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the server on")
    args = parser.parse_args()

    uvicorn.run(app, host="0.0.0.0", port=args.port)
