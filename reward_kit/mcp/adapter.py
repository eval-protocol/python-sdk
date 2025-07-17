"""
Environment Adapter Interface

This defines the interface that users implement to connect their
environments to the MCP framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple


class EnvironmentAdapter(ABC):
    """
    Abstract base class for environment adapters.

    Users implement this interface to connect their environments
    to the MCP server framework. This provides a clean separation
    between the MCP protocol layer and the environment implementation.
    """

    @abstractmethod
    def create_environment(self, config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Create and return a new environment instance.

        Args:
            config: Optional configuration dict for environment creation

        Returns:
            Environment instance (type depends on the specific implementation)
        """
        pass

    @abstractmethod
    def create_environment_with_seed(self, config: Optional[Dict[str, Any]] = None, seed: Optional[int] = None) -> Tuple[Any, int, Dict[str, Any]]:
        """
        Create and return a new environment instance with a specific seed.
        """
        pass

    @abstractmethod
    def reset_environment(
        self, env: Any, seed: Optional[int] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Args:
            env: Environment instance
            seed: Optional seed for reproducibility

        Returns:
            Tuple of (initial_observation, info_dict)
        """
        pass

    @abstractmethod
    def step_environment(
        self, env: Any, action: Any
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            env: Environment instance
            action: Action to execute (type depends on environment)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        pass

    @abstractmethod
    def close_environment(self, env: Any) -> None:
        """
        Clean up environment resources.

        Args:
            env: Environment instance to close
        """
        pass

    @abstractmethod
    def parse_action(self, action_str: str) -> Any:
        """
        Parse action string from MCP tool call into environment action.

        Args:
            action_str: Action string from MCP client

        Returns:
            Action in format expected by environment
        """
        pass

    @abstractmethod
    def format_observation(self, observation: Any) -> Any:
        """
        Format environment observation for MCP response.

        Args:
            observation: Raw observation from environment

        Returns:
            JSON-serializable observation data
        """
        pass

    @abstractmethod
    def get_action_space_description(self) -> Dict[str, Any]:
        """
        Get description of valid actions for this environment.

        Returns:
            Dict describing the action space (for tool schema generation)
        """
        pass

    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get the default environment configuration.

        Returns:
            Dict describing the default environment configuration
        """
        pass