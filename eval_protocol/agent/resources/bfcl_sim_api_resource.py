import copy
import importlib
import sys
from pathlib import Path

# Import BFCL File and Directory for isinstance checks from local implementation
from .bfcl_envs.gorilla_file_system import Directory as BFCLDirectory, File as BFCLFile

BFCL_TYPES_AVAILABLE = True
import gc
import inspect
import json
from typing import Any, Dict, List, Optional, Tuple, Union

from ..resource_abc import ForkableResource


class BFCLSimAPIResource(ForkableResource):
    CLASS_FILE_PATH_MAPPING = {
        "GorillaFileSystem": "eval_protocol.agent.resources.bfcl_envs.gorilla_file_system",
        "MathAPI": "eval_protocol.agent.resources.bfcl_envs.math_api",
        "TwitterAPI": "eval_protocol.agent.resources.bfcl_envs.posting_api",
        # Add these back when implemented:
        # "MessageAPI": "eval_protocol.agent.resources.bfcl_envs.message_api",
        # "TicketAPI": "eval_protocol.agent.resources.bfcl_envs.ticket_api",
        # "TradingBot": "eval_protocol.agent.resources.bfcl_envs.trading_bot",
        # "TravelAPI": "eval_protocol.agent.resources.bfcl_envs.travel_booking",
        # "VehicleControlAPI": "eval_protocol.agent.resources.bfcl_envs.vehicle_control",
    }
    STATELESS_CLASSES = ["MathAPI"]

    def _serialize_bfcl_file(self, file_obj: BFCLFile) -> Dict[str, Any]:
        """Serializes a BFCL File object into a canonical dictionary."""
        return {
            "type": "file",  # Add a type hint for clarity, though not in original __eq__
            "name": file_obj.name,
            "content": file_obj.content,
        }

    def _serialize_bfcl_directory(self, dir_obj: BFCLDirectory) -> Dict[str, Any]:
        """Serializes a BFCL Directory object into a canonical dictionary."""
        serialized_contents: Dict[str, Any] = {}
        # Sort keys for canonical representation, crucial for reliable comparison
        for item_name, item_value in sorted(dir_obj.contents.items()):
            if BFCL_TYPES_AVAILABLE and isinstance(item_value, BFCLFile):
                serialized_contents[item_name] = self._serialize_bfcl_file(item_value)
            elif BFCL_TYPES_AVAILABLE and isinstance(item_value, BFCLDirectory):
                serialized_contents[item_name] = self._serialize_bfcl_directory(item_value)
            else:
                # Fallback for other types if any, or if BFCL types weren't imported
                try:
                    json.dumps(item_value)
                    serialized_contents[item_name] = item_value
                except (TypeError, OverflowError):
                    serialized_contents[item_name] = str(item_value)
        return {
            "type": "directory",  # Add a type hint
            "name": dir_obj.name,
            "contents": serialized_contents,
            # Parent is intentionally excluded to match original Directory.__eq__
        }

    def __init__(self, env_instances: Optional[Dict[str, Any]] = None):
        self._env_instances = env_instances if env_instances is not None else {}
        self._initial_config: Dict[str, Any] = {}  # To store initial configuration for forking

    async def setup(self, config: Dict[str, Any]) -> None:
        """Initializes the resource with a given configuration."""
        self._initial_config = copy.deepcopy(config)
        involved_classes = config.get("involved_classes", [])
        initial_config_data = config.get("initial_config", {})

        for class_name in involved_classes:
            if class_name not in self._env_instances:
                module_name = self.CLASS_FILE_PATH_MAPPING[class_name]
                module = importlib.import_module(module_name)
                class_ = getattr(module, class_name)
                instance = class_()

                if class_name not in self.STATELESS_CLASSES:
                    class_initial_config = initial_config_data.get(class_name, {})
                    instance._load_scenario(copy.deepcopy(class_initial_config))

                self._env_instances[class_name] = instance

    async def fork(self) -> "ForkableResource":
        """Creates and returns a new, independent instance of this resource
        with an identical copy of the current state.
        """
        # Deep copy the environment instances to create an independent fork
        forked_instances = copy.deepcopy(self._env_instances)
        new_resource = BFCLSimAPIResource(env_instances=forked_instances)
        new_resource._initial_config = copy.deepcopy(
            self._initial_config
        )  # Copy initial config for potential re-setup
        return new_resource

    async def checkpoint(self) -> Dict[str, Any]:
        """Returns a serializable representation of the resource's current state."""
        # Use get_comparable_state for checkpointing
        state_data = self.get_comparable_state()
        return state_data

    async def restore(self, state_data: Dict[str, Any]) -> None:
        """Restores the resource's state from a previously checkpointed state_data."""
        # Re-initialize based on initial config
        await self.setup(self._initial_config)
        # Restore state from the provided state_data using _set_comparable_state
        self._set_comparable_state(state_data)

    async def step(self, action_name: str, action_params: Dict[str, Any]) -> Any:
        """Executes a named action with given parameters on the resource."""
        # Find the correct environment instance and call the method
        for instance in self._env_instances.values():
            if hasattr(instance, action_name):
                try:
                    # Convert tuple back to list if needed by the tool function
                    for key, value in action_params.items():
                        if isinstance(value, tuple):
                            action_params[key] = list(value)
                    result = getattr(instance, action_name)(**action_params)
                    # BFCL envs might return results directly or modify state
                    if isinstance(result, str):
                        # Convert string result to dict if needed by type checker
                        try:
                            parsed_result = json.loads(result)
                            if isinstance(parsed_result, dict):
                                return parsed_result
                        except json.JSONDecodeError:
                            pass
                    return result
                except Exception as e:
                    return {"error": f"Error executing tool {action_name}: {e}"}
        return {"error": f"Tool {action_name} not found in available resources."}

    async def get_observation(self) -> Dict[str, Any]:
        """Returns the current observable state of the resource for the agent."""
        # This needs to be defined based on what the agent should observe from the BFCL envs.
        # It might be a summary of the environment state or specific attributes.
        # For now, return a placeholder or a simple representation.
        observation = self.get_comparable_state()  # Return comparable state as observation for now
        return observation

    async def get_tools_spec(self) -> List[Dict[str, Any]]:
        """Returns a list of tool specifications (e.g., OpenAPI format)
        that are currently available or applicable to this resource's state.
        """
        # This needs to generate tool specifications from the methods of the BFCL env instances.
        # It can adapt the logic from verifiers.envs.tool_env.infer_schema_from_function
        tool_specs = []
        for instance in self._env_instances.values():
            # Inspect methods of the instance
            for name, method in inspect.getmembers(instance, predicate=inspect.ismethod):
                if not name.startswith("_"):  # Exclude private methods
                    # Infer schema from method signature
                    try:
                        schema = self._infer_schema_from_method(method)
                        tool_specs.append(schema)
                    except Exception as e:
                        print(f"Could not infer schema for {name}: {e}")
        return tool_specs

    async def close(self) -> None:
        """Performs any necessary cleanup for the resource."""
        self._env_instances.clear()
        gc.collect()

    def get_comparable_state(self) -> Dict[str, Any]:
        """
        Returns a serializable representation of the resource's state for comparison.
        This method is synchronous for use in reward functions.
        """
        state = {}
        for class_name, instance in self._env_instances.items():
            instance_state = {}
            # Specifically handle GorillaFileSystem's root attribute if it's the one
            # This is a bit of a special case due to its recursive nature and importance.
            if (
                class_name == "GorillaFileSystem"
                and hasattr(instance, "root")
                and BFCL_TYPES_AVAILABLE
                and isinstance(instance.root, BFCLDirectory)
            ):
                # Serialize 'root' attribute using the new method
                instance_state["root"] = self._serialize_bfcl_directory(instance.root)  # type: ignore[assignment]
                # Serialize other public attributes normally
                for attr_name, value in vars(instance).items():
                    if not attr_name.startswith("_") and attr_name != "root":
                        if BFCL_TYPES_AVAILABLE and isinstance(value, BFCLDirectory):
                            instance_state[attr_name] = self._serialize_bfcl_directory(value)
                        elif BFCL_TYPES_AVAILABLE and isinstance(value, BFCLFile):
                            instance_state[attr_name] = self._serialize_bfcl_file(value)
                        else:
                            try:
                                json.dumps(value)
                                instance_state[attr_name] = value
                            except (TypeError, OverflowError):
                                instance_state[attr_name] = str(  # type: ignore[assignment]
                                    value
                                )  # Convert non-serializable objects to string
            else:  # For other classes or if GorillaFileSystem doesn't have 'root' or types unavailable
                for attr_name, value in vars(instance).items():
                    if not attr_name.startswith("_"):
                        # Check if value is an instance of BFCLDirectory or BFCLFile first
                        if BFCL_TYPES_AVAILABLE and isinstance(value, BFCLDirectory):
                            instance_state[attr_name] = self._serialize_bfcl_directory(value)
                        elif BFCL_TYPES_AVAILABLE and isinstance(value, BFCLFile):
                            instance_state[attr_name] = self._serialize_bfcl_file(value)
                        else:
                            try:
                                json.dumps(value)
                                instance_state[attr_name] = value
                            except (TypeError, OverflowError):
                                instance_state[attr_name] = str(  # type: ignore[assignment]
                                    value
                                )  # Convert non-serializable objects to string
            state[class_name] = instance_state
        return state

    def _set_comparable_state(self, state_data: Dict[str, Any]) -> None:
        """Helper to set state on BFCL environment instances from a comparable state dict."""
        for class_name, state in state_data.items():
            if class_name in self._env_instances:
                instance = self._env_instances[class_name]
                for attr_name, value in state.items():
                    if hasattr(instance, attr_name):
                        try:
                            setattr(instance, attr_name, value)
                        except Exception as e:
                            print(f"Could not set attribute {attr_name} on {instance.__class__.__name__}: {e}")

    def _infer_schema_from_method(self, method: Any) -> Dict[str, Any]:
        """Helper to infer tool schema from a method signature."""
        # This is a simplified version, can be expanded based on verifiers.envs.tool_env.infer_schema_from_function
        schema = {
            "name": method.__name__,
            "description": method.__doc__ if method.__doc__ else "",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }
        sig = inspect.signature(method)
        type_mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            List: "array",
            dict: "object",
            Dict: "object",
            Any: "string",  # Default to string for Any or unknown
            type(None): "null",  # For Optional[str] = None
        }

        for name, param in sig.parameters.items():
            if name == "self":
                continue

            param_type_annotation = param.annotation
            json_type = "string"  # Default

            if param_type_annotation != inspect.Parameter.empty:
                # Handle Optional types like Optional[str]
                if hasattr(param_type_annotation, "__origin__") and param_type_annotation.__origin__ is Union:
                    # Get the first non-None type from Union for Optional[T]
                    union_args = [arg for arg in param_type_annotation.__args__ if arg is not type(None)]
                    if union_args:
                        actual_type = union_args[0]
                        json_type = type_mapping.get(actual_type, "string")
                        # Handle List[str] etc.
                        if hasattr(actual_type, "__origin__") and actual_type.__origin__ in [list, List]:
                            json_type = "array"
                            # Try to infer item type for List[T]
                            if hasattr(actual_type, "__args__") and actual_type.__args__:
                                item_type_annotation = actual_type.__args__[0]
                                item_json_type = type_mapping.get(item_type_annotation, "string")
                                schema["parameters"]["properties"][name] = {
                                    "type": "array",
                                    "items": {"type": item_json_type},
                                }
                            else:  # Fallback if item type can't be inferred
                                schema["parameters"]["properties"][name] = {
                                    "type": "array",
                                    "items": {"type": "string"},
                                }
                            if param.default == inspect.Parameter.empty:
                                schema["parameters"]["required"].append(name)
                            continue  # Skip default property assignment below
                    else:  # Should not happen for valid Optional[T]
                        json_type = "string"
                elif hasattr(param_type_annotation, "__origin__") and param_type_annotation.__origin__ in [list, List]:
                    json_type = "array"
                    if hasattr(param_type_annotation, "__args__") and param_type_annotation.__args__:
                        item_type_annotation = param_type_annotation.__args__[0]
                        item_json_type = type_mapping.get(item_type_annotation, "string")
                        schema["parameters"]["properties"][name] = {
                            "type": "array",
                            "items": {"type": item_json_type},
                        }
                    else:  # Fallback
                        schema["parameters"]["properties"][name] = {
                            "type": "array",
                            "items": {"type": "string"},
                        }
                    if param.default == inspect.Parameter.empty:
                        schema["parameters"]["required"].append(name)
                    continue  # Skip default property assignment
                else:
                    json_type = type_mapping.get(param_type_annotation, "string")

            schema["parameters"]["properties"][name] = {"type": json_type}
            if param.default == inspect.Parameter.empty:
                schema["parameters"]["required"].append(name)
        return schema
