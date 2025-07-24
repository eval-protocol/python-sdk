"""
PythonStateResource: A ForkableResource that manages state as a Python dictionary.
"""

import copy
import pickle
from typing import Any, Dict, List, Optional

from ..resource_abc import ForkableResource


class PythonStateResource(ForkableResource):
    """
    A ForkableResource that manages its state as an in-memory Python dictionary.

    This resource is useful for tasks where the environment's state can be
    represented and manipulated directly as Python objects.

    Attributes:
        _state (Dict[str, Any]): The internal dictionary holding the resource's state.
        _config (Dict[str, Any]): The configuration passed during setup.
    """

    def __init__(self) -> None:
        self._state: Dict[str, Any] = {}
        self._config: Dict[str, Any] = {}

    async def setup(self, config: Dict[str, Any]) -> None:
        """
        Initializes the resource with a given configuration.

        The configuration can specify an 'initial_state' dictionary.

        Args:
            config: Configuration dictionary.
                    Expected keys:
                    - 'initial_state' (Optional[Dict[str, Any]]):
                      A dictionary to set as the initial state.
        """
        self._config = copy.deepcopy(config)
        self._state = copy.deepcopy(self._config.get("initial_state", {}))

    async def fork(self) -> "PythonStateResource":
        """
        Creates and returns a new, independent instance of this resource
        with an identical copy of the current state.
        """
        forked_resource = PythonStateResource()
        forked_resource._config = copy.deepcopy(self._config)
        forked_resource._state = copy.deepcopy(self._state)
        return forked_resource

    async def checkpoint(self) -> bytes:
        """
        Returns a serializable representation of the resource's current state
        using pickle.
        """
        return pickle.dumps(self._state)

    async def restore(self, state_data: bytes) -> None:
        """
        Restores the resource's state from previously checkpointed state_data
        (pickle format).
        """
        self._state = pickle.loads(state_data)

    async def step(self, action_name: str, action_params: Dict[str, Any]) -> Any:
        """
        Executes a named action with given parameters on the resource.

        This implementation provides a generic 'update_state' action
        that merges action_params into the current state.
        Subclasses could override this for more specific actions.

        Args:
            action_name: The name of the action to perform.
                         Currently supports 'update_state'.
            action_params: A dictionary of parameters for the action.
                           For 'update_state', these are key-value pairs
                           to update in the state.

        Returns:
            A copy of the updated state.

        Raises:
            NotImplementedError: If action_name is not 'update_state'.
        """
        if action_name == "update_state":
            self._state.update(action_params)
            return copy.deepcopy(self._state)
        elif action_name == "get_value":
            key = action_params.get("key")
            if key is None:
                raise ValueError("Missing 'key' in action_params for 'get_value'")
            return self._state.get(key)
        else:
            raise NotImplementedError(f"Action '{action_name}' is not implemented for PythonStateResource.")

    async def get_observation(self) -> Dict[str, Any]:
        """
        Returns a deep copy of the current observable state of the resource.
        """
        return copy.deepcopy(self._state)

    def get_state(self) -> Dict[str, Any]:
        """
        Returns a deep copy of the current state dictionary.
        This is a synchronous version of get_observation for compatibility with test tasks.
        """
        return copy.deepcopy(self._state)

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Sets the resource's state to the provided dictionary.

        Args:
            state: A dictionary containing the new state.
        """
        self._state = copy.deepcopy(state)

    async def get_tools_spec(self) -> List[Dict[str, Any]]:
        """
        Returns a list of tool specifications available for this resource.

        Provides generic 'update_state' and 'get_value' tools.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "update_state",
                    "description": "Updates the current state dictionary with the provided key-value pairs.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "updates": {
                                "type": "object",
                                "description": "Key-value pairs to update in the state.",
                            }
                        },
                        "required": ["updates"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_value",
                    "description": "Retrieves a value from the state dictionary for a given key.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "key": {
                                "type": "string",
                                "description": "The key of the value to retrieve.",
                            }
                        },
                        "required": ["key"],
                    },
                },
            },
        ]

    async def close(self) -> None:
        """
        Performs any necessary cleanup for the resource.
        For PythonStateResource, this is a no-op as state is in-memory.
        """
        self._state = {}
        self._config = {}
