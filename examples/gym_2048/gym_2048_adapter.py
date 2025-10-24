"""
2048 Environment Adapter

This adapter implements the EnvironmentAdapter interface for 2048 environments,
enabling integration with the MCP-Gym framework.
"""

from typing import Any, Dict, Optional, Tuple

import gym
import gym_2048
from gym_2048.envs.game2048_env import Game2048Env

from eval_protocol.mcp.adapter import EnvironmentAdapter


class Gym2048Adapter(EnvironmentAdapter):
    """2048 adapter for MCP-Gym framework."""

    ACTION_NAMES = ["UP", "RIGHT", "DOWN", "LEFT"]

    def create_environment(self, config: Optional[Dict[str, Any]] = None) -> Game2048Env:
        """
        Create 2048 environment.

        Args:
            config: config is not used in this implementation

        Returns:
            2048 environment instance
        """
        print(f"🔍 Gym2048Adapter.create_environment: config: {config}")
        env = gym.make('2048-v0')
        return env

    def create_environment_with_seed(
        self, config: Optional[Dict[str, Any]] = None, seed: Optional[int] = None
    ) -> Tuple[Game2048Env, int, Dict[str, Any]]:
        """
        Create 2048 environment with seed and return initial state.

        Args:
            config: config is not used in this implementation
            seed: Seed for reproducible environments

        Returns:
            Tuple of (environment, initial_observation, initial_info)
        """
        print(f"🔍 Gym2048Adapter.create_environment_with_seed: seed: {seed}")
        config = config or {}

        # Add seed to config for environment creation
        env_config = {**config, "seed": seed}
        print(f"🔍 Gym2048Adapter.create_environment_with_seed: env_config: {env_config}")

        env = self.create_environment(env_config)
        print(f"🔍 Gym2048Adapter.create_environment_with_seed: created env, calling reset with seed: {seed}")
        obs, info = env.reset(seed=seed)
        print(f"🔍 Gym2048Adapter.create_environment_with_seed: reset returned obs: {obs}, info: {info}")

        return env, obs, info

    def reset_environment(self, env: Game2048Env, seed: Optional[int] = None) -> Tuple[int, Dict[str, Any]]:
        """
        Reset environment.

        Args:
            env: Environment instance
            seed: Optional seed for reset

        Returns:
            Tuple of (observation, info)
        """
        return env.reset(seed=seed)

    def step_environment(self, env: Game2048Env, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """
        Execute environment step.

        Args:
            env: Environment instance
            action: Action index

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        return env.step(action)

    def close_environment(self, env: Game2048Env) -> None:
        """
        Close environment.

        Args:
            env: Environment instance
        """
        env.close()

    def parse_action(self, action_str: str) -> int:
        """
        Parse action string to integer.

        Args:
            action_str: Action string (UP, RIGHT, DOWN, LEFT)

        Returns:
            Action index

        Raises:
            ValueError: If action is invalid
        """
        action_str = action_str.strip().upper()
        if action_str not in self.ACTION_NAMES:
            raise ValueError(f"Invalid action '{action_str}'. Valid actions: {self.ACTION_NAMES}")
        return self.ACTION_NAMES.index(action_str)

    def format_observation(self, observation: int) -> int:
        """
        Format observation for JSON serialization.

        Args:
            observation: Raw observation from environment

        Returns:
            Formatted observation
        """
        return int(observation)

    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration.

        Returns:
            Default configuration dictionary
        """
        return {
            "is_slippery": False,
        }
