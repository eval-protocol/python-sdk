"""
Environment Adapter Interface

This defines the interface that users implement to connect their
environments to the MCP framework. It also provides default implementations
that work with most gymnasium-style and complex environments.
"""

import json
from typing import Any, Dict, Optional, Tuple


class EnvironmentAdapter:
    """
    Environment adapter with default implementations.

    Users can either use this class directly by providing an env_class,
    or inherit from it to customize specific methods for their environment.
    This provides a clean separation between the MCP protocol layer
    and the environment implementation.
    """

    def __init__(self, env_class: Any = None, default_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the environment adapter.

        Args:
            env_class: The environment class to instantiate (required for default implementation)
            default_config: Default configuration for environment creation
        """
        self.env_class = env_class
        self.default_config = default_config or {}

    def create_environment(self, config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Create and return a new environment instance.

        Args:
            config: Optional configuration dict for environment creation

        Returns:
            Environment instance (type depends on the specific implementation)
        """
        if self.env_class is None:
            raise NotImplementedError("env_class must be provided or create_environment must be overridden")

        env_config = self.get_default_config()
        if config:
            env_config.update(config)

        env = self.env_class(config=env_config)
        return env

    def create_environment_with_seed(
        self, config: Optional[Dict[str, Any]] = None, seed: Optional[int] = None
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Create and return a new environment instance with a specific seed.
        """
        env = self.create_environment(config)
        obs, info = env.reset(seed=seed)

        return env, obs, info

    def reset_environment(self, env: Any, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Args:
            env: Environment instance
            seed: Optional seed for reproducibility

        Returns:
            Tuple of (initial_observation, info_dict)
        """
        return env.reset(seed=seed)

    def step_environment(self, env: Any, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            env: Environment instance
            action: Action to execute (type depends on environment)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        return env.step(action)

    def close_environment(self, env: Any) -> None:
        """
        Clean up environment resources.

        Args:
            env: Environment instance to close
        """
        env.close()

    def parse_action(self, action_str: str) -> Any:
        """
        Parse action string from MCP tool call into environment action.

        Args:
            action_str: Action string from MCP client

        Returns:
            Action in format expected by environment
        """
        return json.loads(action_str)

    def format_observation(self, observation: Any) -> Any:
        """
        Format environment observation for MCP response.

        Args:
            observation: Raw observation from environment

        Returns:
            JSON-serializable observation data
        """
        return observation

    def get_default_config(self) -> Dict[str, Any]:
        """
        Get the default environment configuration.

        Returns:
            Dict describing the default environment configuration
        """
        return self.default_config
