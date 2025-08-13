"""
Blackjack Environment Adapter

This adapter implements the EnvironmentAdapter interface for Blackjack environments,
enabling integration with the MCP-Gym framework.
"""

from typing import Any, Dict, Optional, Tuple

from gymnasium.envs.toy_text.blackjack import BlackjackEnv

from eval_protocol.mcp.adapter import EnvironmentAdapter


class BlackjackAdapter(EnvironmentAdapter):
    """Blackjack adapter for MCP-Gym framework."""

    ACTION_NAMES = ["STICK", "HIT"]

    def create_environment(self, config: Optional[Dict[str, Any]] = None) -> BlackjackEnv:
        """
        Create Blackjack environment.

        Args:
            config: Configuration dictionary with optional 'natural' and 'sab'

        Returns:
            Blackjack environment instance
        """
        print(f"🔍 BlackjackAdapter.create_environment: config: {config}")
        config = config or {}
        natural = config.get("natural")
        if natural is None:
            natural = False
            print("🔍 BlackjackAdapter.create_environment: natural is not set in the config, use False by default")
        if isinstance(natural, str):
            natural = natural.lower() == "true"
            print(f"🔍 BlackjackAdapter.create_environment: natural is a string, convert to boolean: {natural}")
        else:
            natural = bool(natural)

        sab = config.get("sab", False)
        if sab is None:
            sab = False
            print("🔍 BlackjackAdapter.create_environment: sab is not set in the config, use False by default")
        if isinstance(sab, str):
            sab = sab.lower() == "true"
            print(f"🔍 BlackjackAdapter.create_environment: sab is a string, convert to boolean: {sab}")
        else:
            sab = bool(sab)

        env = BlackjackEnv(render_mode="ansi", natural=natural, sab=sab)
        print("🔍 BlackjackAdapter.create_environment: Created BlackjackEnv")
        return env

    def create_environment_with_seed(
        self, config: Optional[Dict[str, Any]] = None, seed: Optional[int] = None
    ) -> Tuple[BlackjackEnv, int, Dict[str, Any]]:
        """
        Create Blackjack environment with seed and return initial state.

        Args:
            config: Configuration dictionary
            seed: Seed for reproducible environments

        Returns:
            Tuple of (environment, initial_observation, initial_info)
        """
        print(f"🔍 BlackjackAdapter.create_environment_with_seed: config: {config}, seed: {seed}")
        config = config or {}

        # Add seed to config for environment creation
        env_config = {**config, "seed": seed}
        print(f"🔍 BlackjackAdapter.create_environment_with_seed: env_config: {env_config}")

        env = self.create_environment(env_config)
        print(f"🔍 BlackjackAdapter.create_environment_with_seed: created env, calling reset with seed: {seed}")
        obs, info = env.reset(seed=seed)
        print(f"🔍 BlackjackAdapter.create_environment_with_seed: reset returned obs: {obs}, info: {info}")

        return env, obs, info

    def reset_environment(self, env: BlackjackEnv, seed: Optional[int] = None) -> Tuple[int, Dict[str, Any]]:
        """
        Reset environment.

        Args:
            env: Environment instance
            seed: Optional seed for reset

        Returns:
            Tuple of (observation, info)
        """
        return env.reset(seed=seed)

    def step_environment(
        self, env: BlackjackEnv, action: int
    ) -> Tuple[Tuple[int, int, int], float, bool, bool, Dict[str, Any]]:
        """
        Execute environment step.

        Args:
            env: Environment instance
            action: Action index

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        return env.step(action)

    def close_environment(self, env: BlackjackEnv) -> None:
        """
        Close environment.

        Args:
            env: Environment instance
        """
        return env.close()

    def parse_action(self, action_str: str) -> int:
        """
        Parse action string to integer.

        Args:
            action_str: Action string (STICK, HIT)

        Returns:
            Action index

        Raises:
            ValueError: If action is invalid
        """
        action_str = action_str.strip().upper()
        if action_str not in self.ACTION_NAMES:
            raise ValueError(f"Invalid action '{action_str}'. Valid actions: {self.ACTION_NAMES}")
        return self.ACTION_NAMES.index(action_str)

    def format_observation(self, observation: Tuple[int, int, int]) -> Dict[str, int]:
        """
        Format observation for JSON serialization.

        Args:
            observation: Raw observation from environment

        Returns:
            Formatted observation
        """
        return {
            "player_sum": observation[0],
            "dealer_card": observation[1],
            "usable_ace": observation[2],
        }

    def get_default_config(self) -> Dict[str, bool]:
        """
        Get default configuration.

        Returns:
            Default configuration dictionary
        """
        return {
            "natural": False,
            "sab": False,
        }
