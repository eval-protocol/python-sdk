"""
Abstract Base Class for Forkable Resources in the Agent Evaluation Framework V2.
"""

from abc import ABC, abstractmethod
from typing import (  # Callable removed as not directly used in ABC signatures
    Any,
    Dict,
    List,
    Optional,
)


class ForkableResource(ABC):
    """
    Abstract base class defining the interface for a forkable, checkpointable,
    and interactive environment resource for agent evaluation.
    """

    @abstractmethod
    async def setup(self, config: Dict[str, Any]) -> None:
        """
        Initializes the resource with a given configuration.
        This method should prepare the resource for its first use or fork.
        For example, setting up a database schema, creating a base file system,
        or starting a base Docker container.
        """
        pass

    @abstractmethod
    async def fork(self) -> "ForkableResource":
        """
        Creates and returns a new, independent instance of this resource
        with an identical copy of the current state of the resource it was forked from.
        This new instance is typically an EpisodeResource, used for a single agent rollout.
        """
        pass

    @abstractmethod
    async def checkpoint(self) -> Any:
        """
        Returns a serializable representation of the resource's current state.
        The format of this state (e.g., bytes, dict, path to a file) is specific
        to the resource implementation but must be restorable by `restore()`.
        """
        pass

    @abstractmethod
    async def restore(self, state_data: Any) -> None:
        """
        Restores the resource's state from previously checkpointed `state_data`.
        The resource should be in the same state as when `checkpoint()` was called.
        """
        pass

    @abstractmethod
    async def step(self, action_name: str, action_params: Dict[str, Any]) -> Any:
        """
        Executes a named action with given parameters on the resource.
        This typically modifies the resource's state.
        Returns an observation or result of the action, specific to the resource and action.
        """
        pass

    @abstractmethod
    async def get_observation(self) -> Any:
        """
        Returns the current observable state of the resource for the agent.
        The format of the observation is resource-specific.
        """
        pass

    @abstractmethod
    async def get_tools_spec(self) -> List[Dict[str, Any]]:
        """
        Returns a list of tool specifications (e.g., OpenAI function calling format)
        that are currently available or applicable to this resource's state.
        This can be dynamic, changing based on the resource's current state.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Performs any necessary cleanup for the resource.
        This includes releasing acquired resources like database connections,
        stopping containers, deleting temporary files or directories, etc.
        """
        pass
