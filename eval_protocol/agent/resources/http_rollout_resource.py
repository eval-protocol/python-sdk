"""
HTTP Rollout Resource implementation for the agent evaluation framework.

This resource bridges the HTTP rollout protocol with the ForkableResource interface,
allowing HTTP-based environments to be used in agent evaluations.
"""

import json
import uuid
from typing import Any, Dict, List, Optional

import httpx

from ..resource_abc import ForkableResource
from .http_rollout_protocol import (
    EndEpisodeRequest,
    GameObservation,
    HttpRolloutConfig,
    StartEpisodeRequest,
    StartEpisodeResponse,
    StepRequest,
    StepResponse,
)


class HttpRolloutResource(ForkableResource):
    """
    A ForkableResource implementation that communicates with HTTP rollout servers.

    This resource allows the agent evaluation framework to interact with
    HTTP-based environments through a standardized rollout protocol.
    """

    def __init__(self):
        """Initialize the HTTP rollout resource."""
        super().__init__()
        self.config: Optional[HttpRolloutConfig] = None
        self.episode_id: Optional[str] = None
        self.current_observation: Optional[Dict[str, Any]] = None
        self.is_episode_active = False
        self.client: Optional[httpx.Client] = None

        # Set up logging
        import logging

        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    async def setup(self, config: Dict[str, Any]) -> None:
        """
        Set up the resource with the provided configuration.

        Args:
            config: Configuration dictionary from the task definition
        """
        self.config = HttpRolloutConfig(**config)
        self.client = httpx.Client(timeout=self.config.timeout)

    async def fork(self) -> "HttpRolloutResource":
        """
        Create a new independent instance of this resource.

        For HTTP rollout, forking means creating a new resource instance
        that will start its own episode when initialized.
        """
        if not self.config:
            raise RuntimeError("Resource not set up. Call setup() first.")

        # Create a new instance with the same config
        new_resource = HttpRolloutResource()
        await new_resource.setup(self.config.model_dump())
        return new_resource

    async def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the resource.

        Returns the current observation and episode metadata.
        """
        return {
            "episode_id": self.episode_id,
            "observation": self.current_observation,
            "is_episode_active": self.is_episode_active,
            "type": "http_rollout",
        }

    async def initialize(self, **kwargs) -> None:
        """
        Initialize the resource by starting a new episode.
        Passes any provided kwargs (like seed) to the server in the request body.
        """
        try:
            url = f"{self.config.base_url}{self.config.start_episode_endpoint}"

            # Include any sample data (like seed) in the request body
            if kwargs:
                self.logger.info(f"Sending initialization data to server: {kwargs}")
                response = self.client.post(url, json=kwargs)
            else:
                response = self.client.post(url)
            response.raise_for_status()

            episode_data = response.json()
            self.episode_id = episode_data["episode_id"]
            self.current_observation = episode_data["observation"]
            self.is_episode_active = True

        except Exception as e:
            raise RuntimeError(f"Failed to start HTTP rollout episode: {e}")

    async def get_initial_state_description(self) -> str:
        """
        Get a formatted description of the initial game state for the agent.
        Uses the observation from start_episode to build the prompt.
        """
        # Start episode to get current game state
        if not self.is_episode_active:
            await self.initialize()

        if not self.current_observation:
            return "No initial state available."

        obs = self.current_observation

        # Build comprehensive game prompt
        content = """🎮 FROZEN LAKE GAME - AUTONOMOUS PLAY MODE

🎯 OBJECTIVE: Navigate from S to G without hitting H

📋 GAME RULES: S=start, F=safe, H=hole(death), G=goal(win)

🤖 AUTONOMOUS MODE INSTRUCTIONS:
- You are playing this game AUTONOMOUSLY until completion
- KEEP MAKING MOVES using the step tool until you reach G or hit H
- DO NOT ask for user input or wait for confirmation
- DO NOT stop after one move - continue until the game ends
- Each move should be followed immediately by another move
- Game only ends when you reach G (win) or hit H (lose)

🎮 ACTION: Use step tool with: "left", "right", "up", or "down"

⚡ START NOW - Make your first move and continue until the game is complete!"""

        description_parts = [content]

        if obs.get("message"):
            description_parts.append(f"\nEnvironment: {obs['message']}")

        if obs.get("visual"):
            description_parts.append(f"\nGame Board:\n{obs['visual']}")

        if obs.get("position"):
            description_parts.append(f"\nStarting Position: {obs['position']}")

        description_parts.append("\nGame Rules:")
        description_parts.append("- S = Start position")
        description_parts.append("- F = Frozen (safe to step on)")
        description_parts.append("- H = Hole (game over if you step here)")
        description_parts.append("- G = Goal (reach this to win)")
        description_parts.append("- [X] = Your current position")

        return "\n".join(description_parts)

    async def cleanup(self) -> None:
        """
        Clean up the resource by ending the current episode.
        """
        if self.is_episode_active and self.episode_id:
            try:
                url = f"{self.config.base_url}{self.config.end_episode_endpoint}"
                response = self.client.post(url, json={"episode_id": self.episode_id})
                response.raise_for_status()

            except Exception as e:
                # Log but don't raise - cleanup should be best effort
                print(f"Warning: Failed to properly end episode {self.episode_id}: {e}")

            finally:
                self.episode_id = None
                self.current_observation = None
                self.is_episode_active = False

        # Close the HTTP client
        self.client.close()

    async def get_tools_spec(self) -> List[Dict[str, Any]]:
        """
        Get the list of available tools for this resource.

        For HTTP rollout, this returns the 'step' tool that allows
        the agent to take actions in the environment.
        """
        return [
            {
                "name": "step",
                "description": "Take a step in the Frozen Lake game by choosing a direction to move",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["left", "down", "right", "up"],
                            "description": "The direction to move in the game: 'left', 'down', 'right', or 'up'",
                        }
                    },
                    "required": ["action"],
                },
            }
        ]

    async def step(self, action_name: str, action_params: Dict[str, Any]) -> Any:
        """
        Execute a tool call on this resource.

        For HTTP rollout, this handles the 'step' tool by sending
        the action to the HTTP rollout server.
        """
        if not self.is_episode_active or not self.episode_id:
            # If no active episode, start one first
            await self.initialize()

        if action_name == "step":
            action = action_params.get("action")
            return await self._handle_step_tool(action)
        else:
            raise ValueError(f"Unknown action: {action_name}")

    async def get_observation(self) -> Any:
        """
        Get the current observation from the environment.
        """
        if self.current_observation:
            return self.current_observation
        else:
            return {"message": "No observation available. Start an episode first."}

    async def checkpoint(self) -> Dict[str, Any]:
        """
        Create a checkpoint of the current resource state.

        For HTTP rollout, this saves the episode ID and current observation.
        """
        return {
            "episode_id": self.episode_id,
            "current_observation": self.current_observation,
            "is_episode_active": self.is_episode_active,
        }

    async def restore(self, state_data: Dict[str, Any]) -> None:
        """
        Restore the resource state from a checkpoint.

        Note: This is limited for HTTP rollout since we can't restore
        arbitrary server-side state.
        """
        self.episode_id = state_data.get("episode_id")
        self.current_observation = state_data.get("current_observation")
        self.is_episode_active = state_data.get("is_episode_active", False)

    async def close(self) -> None:
        """
        Clean up and close the resource.
        """
        await self.cleanup()

    async def _handle_step_tool(self, action: Any) -> Dict[str, Any]:
        """
        Handle the 'step' tool by sending an action to the HTTP rollout server.
        """
        try:
            # Convert string action to integer for the server
            action_map = {"left": 0, "down": 1, "right": 2, "up": 3}

            if isinstance(action, str):
                if action.lower() not in action_map:
                    raise ValueError(
                        f"Invalid action '{action}'. Must be one of: left, down, right, up"
                    )
                numeric_action = action_map[action.lower()]
            else:
                # Backward compatibility with numeric actions
                numeric_action = action

            url = f"{self.config.base_url}{self.config.step_endpoint}"
            step_data = {"episode_id": self.episode_id, "action": numeric_action}

            response = self.client.post(url, json=step_data)
            response.raise_for_status()

            step_result = response.json()
            self.current_observation = step_result["observation"]

            # If the episode is done, mark it as inactive
            if step_result.get("is_done", False):
                self.is_episode_active = False

            # Format the response for the agent
            observation = step_result["observation"]
            message = observation.get("message", "")
            visual = observation.get("visual", "")

            # Create a comprehensive response
            response_content = []
            if message:
                response_content.append(f"Environment: {message}")
            if visual:
                response_content.append(f"Visual State:\n{visual}")

            # Add structured data
            response_content.append(
                f"Position: {observation.get('position', 'unknown')}"
            )
            response_content.append(f"Done: {step_result.get('is_done', False)}")

            if step_result.get("is_done", False):
                won = observation.get("won", False)
                response_content.append(f"Result: {'Victory!' if won else 'Game Over'}")

            return {"content": [{"type": "text", "text": "\n".join(response_content)}]}

        except Exception as e:
            raise RuntimeError(f"Failed to execute step: {e}")

    def __del__(self):
        """Ensure cleanup on deletion."""
        if hasattr(self, "client") and self.client:
            try:
                self.client.close()
            except Exception:
                pass  # Ignore cleanup errors during deletion
