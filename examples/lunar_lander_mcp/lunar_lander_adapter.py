#!/usr/bin/env python3
"""
LunarLander Adapter for Gymnasium Environment

This adapter handles the specific mechanics of the LunarLander environment
including discrete action space, rendering, and state interpretation.
"""

import base64
import io
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from PIL import Image

from gymnasium.envs.box2d.lunar_lander import LunarLander

from reward_kit.mcp.adapter import EnvironmentAdapter


class LunarLanderAdapter(EnvironmentAdapter):
    """LunarLander adapter for MCP-Gym framework."""

    def __init__(self):
        # Discrete action mapping for LunarLander
        self.action_map = {
            "NOTHING": 0,
            "FIRE_LEFT": 1,
            "FIRE_MAIN": 2,
            "FIRE_RIGHT": 3,
        }

    def create_environment(self, config: Optional[Dict[str, Any]] = None) -> LunarLander:
        """Create and configure the LunarLander environment."""
        env_config = self.get_default_config()
        if config:
            env_config.update(config)
        
        env = LunarLander(**env_config)
        return env

    def create_environment_with_seed(
        self, config: Optional[Dict[str, Any]] = None, seed: Optional[int] = None
    ) -> Tuple[LunarLander, Any, Dict[str, Any]]:
        """Create and configure the LunarLander environment with a specific seed."""
        env = self.create_environment(config)
        obs, info = env.reset(seed=seed)
        return env, obs, info

    def reset_environment(
        self, env: LunarLander, seed: Optional[int] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment."""
        if seed is not None:
            return env.reset(seed=seed)
        return env.reset()

    def step_environment(
        self, env: LunarLander, action: int
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        return env.step(action)

    def close_environment(self, env: LunarLander) -> None:
        """Clean up environment resources."""
        env.close()

    def parse_action(self, action: str) -> int:
        """Parse discrete action string to environment action integer."""
        if action not in self.action_map:
            raise ValueError(f"Invalid action '{action}'. Valid actions: {list(self.action_map.keys())}")
        
        return self.action_map[action]

    def format_observation(self, obs: np.ndarray) -> Dict[str, Any]:
        """Format observation array into structured data."""
        # LunarLander observation is 8-dimensional:
        # [x, y, vx, vy, angle, angular_velocity, leg1_contact, leg2_contact]

        if len(obs) != 8:
            raise ValueError(f"Expected 8-dimensional observation, got {len(obs)}")

        return {
            "position": {"x": float(obs[0]), "y": float(obs[1])},
            "velocity": {"x": float(obs[2]), "y": float(obs[3])},
            "orientation": {"angle": float(obs[4]), "angular_velocity": float(obs[5])},
            "legs": {"left_contact": bool(obs[6]), "right_contact": bool(obs[7])},
        }

    def get_action_space_description(self) -> Dict[str, Any]:
        """Get description of valid actions for this environment."""
        return {
            "type": "discrete",
            "n": 4,
            "actions": {
                "NOTHING": {"value": 0, "description": "Do nothing"},
                "FIRE_LEFT": {"value": 1, "description": "Fire left orientation engine"},
                "FIRE_MAIN": {"value": 2, "description": "Fire main engine"},
                "FIRE_RIGHT": {"value": 3, "description": "Fire right orientation engine"},
            },
            "examples": {
                "idle": "NOTHING",
                "descend": "FIRE_MAIN",
                "turn_left": "FIRE_LEFT",
                "turn_right": "FIRE_RIGHT",
            }
        }

    def get_observation_space_info(self) -> Dict[str, Any]:
        """Get information about the observation space."""
        return {
            "type": "box",
            "shape": [8],
            "description": {
                "0": "x coordinate",
                "1": "y coordinate",
                "2": "x velocity",
                "3": "y velocity",
                "4": "angle",
                "5": "angular velocity",
                "6": "left leg contact",
                "7": "right leg contact",
            },
        }

    def render_frame(self, env: LunarLander) -> Optional[str]:
        """Render the current environment state as base64 encoded image."""
        try:
            # Get RGB array from environment
            rgb_array = env.render()

            if rgb_array is None:
                return None

            # Convert numpy array to PIL Image
            if isinstance(rgb_array, np.ndarray):
                image = Image.fromarray(rgb_array.astype(np.uint8))
            else:
                # Handle case where render returns a list or other format
                return None

            # Convert to base64
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)

            # Encode as base64 string
            image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            return f"data:image/png;base64,{image_b64}"

        except Exception as e:
            print(f"Error rendering frame: {e}")
            return None

    def is_successful_landing(
        self, obs: np.ndarray, reward: float, terminated: bool
    ) -> bool:
        """Check if the landing was successful."""
        if not terminated:
            return False

        # Successful landing typically gives positive reward
        # and both legs should be in contact
        legs_contact = obs[6] and obs[7] if len(obs) >= 8 else False

        return reward > 0 and legs_contact

    def get_landing_status(
        self, obs: np.ndarray, reward: float, terminated: bool, truncated: bool
    ) -> str:
        """Get a human-readable status of the landing attempt."""
        if not (terminated or truncated):
            return "🚀 Flying"

        if truncated:
            return "⏰ Time limit reached"

        if self.is_successful_landing(obs, reward, terminated):
            return "🎯 Successful landing!"
        elif reward < -50:  # Crashed
            return "💥 Crashed!"
        else:
            return "❌ Landing failed"

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for LunarLander environment."""
        return {
            "continuous": False,
            "gravity": -10.0,
            "enable_wind": False,
            "wind_power": 15.0,
            "turbulence_power": 1.5,
            "render_mode": "rgb_array"
        }