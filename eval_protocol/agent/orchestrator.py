# mypy: ignore-errors
"""
Orchestrator for the Agent Evaluation Framework V2.
Manages the lifecycle of a task using ForkableResources.
"""

import asyncio
import importlib
import inspect
import json
import logging
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type, cast
import importlib.util as _importlib_util

# Determine OpenAI availability without importing symbols for typing
OPENAI_AVAILABLE = _importlib_util.find_spec("openai") is not None

# Expose AsyncOpenAI/OpenAI at module level for tests/patching, even if we import lazily elsewhere
if OPENAI_AVAILABLE:
    try:
        from openai import AsyncOpenAI as AsyncOpenAI, OpenAI as OpenAI  # type: ignore[import-not-found]
    except Exception:

        class AsyncOpenAI:  # type: ignore[no-redef]
            def __init__(self, **_: Any) -> None:
                pass

        class OpenAI:  # type: ignore[no-redef]
            def __init__(self, **_: Any) -> None:
                pass
else:

    class AsyncOpenAI:  # type: ignore[no-redef]
        def __init__(self, **_: Any) -> None:
            pass

    class OpenAI:  # type: ignore[no-redef]
        def __init__(self, **_: Any) -> None:
            pass


# Max steps for the inner loop within a single user turn
MAX_STEPS_PER_USER_TURN = 10

from ..models import Message, TaskDefinitionModel
from .resource_abc import ForkableResource

# Import specific resource types for type checking if needed, or handle dynamically
from .resources import (
    BFCLSimAPIResource,
    DockerResource,
    FileSystemResource,
    PythonStateResource,
    SQLResource,
)


class Orchestrator:
    def __init__(self, task_definition: TaskDefinitionModel):
        self.task_definition = task_definition
        self.base_resource: Optional[ForkableResource] = None
        self.tools_module: Optional[Any] = None
        self.reward_function: Optional[Callable[..., Any]] = None
        self.logger = logging.getLogger(f"Orchestrator.{self.task_definition.name}")
        self.logger.setLevel(logging.DEBUG)  # Ensure debug logs are processed
        self.logger.info(f"Orchestrator initialized for task: {self.task_definition.name}")
        # Use Any here to avoid pyright stubs mismatches across openai versions
        self._openai_client: Optional[Any] = None

    def _initialize_openai_client(self):
        """Initializes the AsyncOpenAI client if available and not already initialized."""
        if not OPENAI_AVAILABLE:
            self.logger.warning("OpenAI library not available. Cannot use OpenAI models.")
            return
        if self._openai_client is None:
            try:
                from openai import AsyncOpenAI  # type: ignore[import-not-found]

                self._openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))  # type: ignore[call-arg]
                self.logger.info("AsyncOpenAI client initialized.")
            except Exception as e:
                self.logger.error(f"Failed to initialize AsyncOpenAI client: {e}")
                self._openai_client = None  # Ensure it's None if init fails

    def _initialize_fireworks_client(self):
        """Initializes the Fireworks client using OpenAI-compatible interface."""
        if not OPENAI_AVAILABLE:
            self.logger.warning("OpenAI library not available. Cannot use Fireworks models.")
            return
        if self._openai_client is None:
            try:
                from openai import AsyncOpenAI  # type: ignore[import-not-found]

                self._openai_client = AsyncOpenAI(  # type: ignore[call-arg]
                    api_key=os.environ.get("FIREWORKS_API_KEY"),
                    base_url="https://api.fireworks.ai/inference/v1",
                )
                self.logger.info("Fireworks client initialized.")
            except Exception as e:
                self.logger.error(f"Failed to initialize Fireworks client: {e}")
                self._openai_client = None

    def _validate_conversation_messages(self, conversation_messages: List[Dict[str, Any]]) -> None:
        """
        Validate and fix conversation messages to ensure OpenAI API compliance.

        OpenAI requires that tool messages must be preceded by an assistant message with tool_calls.
        This method detects and fixes cases where tool messages are orphaned.
        """
        if not conversation_messages:
            return

        for i, msg in enumerate(conversation_messages):
            if msg.get("role") == "tool":
                # Check if previous message is assistant with tool_calls
                if i == 0:
                    # Tool message at start - this is always invalid
                    self.logger.error(f"Found orphaned tool message at start of conversation: {msg}")
                    raise ValueError("Tool message cannot be the first message in conversation")

                prev_msg = conversation_messages[i - 1]
                if prev_msg.get("role") != "assistant" or not prev_msg.get("tool_calls"):
                    # Found orphaned tool message - log and remove it
                    self.logger.warning(
                        f"Found orphaned tool message without preceding assistant tool_calls at index {i}: {msg}"
                    )
                    self.logger.warning(
                        "This suggests a bug in conversation history management - removing invalid tool message"
                    )
                    conversation_messages.pop(i)
                    # Recursively validate again since we modified the list
                    return self._validate_conversation_messages(conversation_messages)

    def _load_module_and_function(self, full_path: str) -> Optional[Callable[..., Any]]:
        try:
            module_path, function_name = full_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            func = getattr(module, function_name)

            # Check if the attribute exists but might not be directly callable due to decoration
            # For example, bfcl_reward is defined in the module but wrapped with @reward_function
            if hasattr(module, function_name):
                # For attributes that are or contain callable objects
                attr = getattr(module, function_name)
                if callable(attr):
                    self.logger.info(f"Successfully loaded function '{function_name}' from module '{module_path}'.")
                    return attr
                # For module-level objects that might wrap callable functions
                elif hasattr(attr, "__call__"):
                    self.logger.info(
                        f"Successfully loaded callable object '{function_name}' from module '{module_path}'."
                    )
                    return attr.__call__
                else:
                    self.logger.error(f"Loaded attribute '{function_name}' from '{module_path}' is not callable.")
            else:
                self.logger.error(f"Attribute '{function_name}' not found in module '{module_path}'.")
            return None
        except (ImportError, AttributeError, ValueError) as e:
            self.logger.error(f"Failed to load function from '{full_path}': {e}")
            return None

    async def _load_task_components(self) -> bool:
        if self.task_definition.tools_module_path:
            try:
                self.tools_module = importlib.import_module(self.task_definition.tools_module_path)
                self.logger.info(f"Successfully loaded tools module: {self.task_definition.tools_module_path}")
            except ImportError as e:
                self.logger.error(f"Failed to import tools module '{self.task_definition.tools_module_path}': {e}")
                return False
        else:
            self.logger.info("No 'tools_module_path' specified. Tools may only come from resource.get_tools_spec().")

        # Load reward function
        if self.task_definition.reward_function_path:
            try:
                # First try direct import
                self.reward_function = self._load_module_and_function(self.task_definition.reward_function_path)

                if not self.reward_function:
                    # If that failed, check if we need to import from eval_protocol.rewards
                    if "." not in self.task_definition.reward_function_path:
                        # Try importing from rewards directly as a fallback
                        fallback_path = f"eval_protocol.rewards.{self.task_definition.reward_function_path}"
                        self.logger.info(f"Attempting fallback import from: {fallback_path}")
                        self.reward_function = self._load_module_and_function(fallback_path)

                    # If still no function, try importing from __init__ exports
                    if (
                        not self.reward_function
                        and "eval_protocol.rewards" in self.task_definition.reward_function_path
                    ):
                        # Extract the function name from the path
                        func_name = self.task_definition.reward_function_path.split(".")[-1]
                        self.logger.debug(f"Attempting to get function by name: {func_name}")
                        try:
                            import eval_protocol.rewards

                            self.logger.debug(f"Available in rewards module: {dir(eval_protocol.rewards)}")
                            if hasattr(eval_protocol.rewards, func_name):
                                self.reward_function = getattr(eval_protocol.rewards, func_name)
                                self.logger.info(f"Found reward function {func_name} in eval_protocol.rewards")
                                self.logger.debug(f"Loaded function type: {type(self.reward_function)}")
                                self.logger.debug(f"Is callable: {callable(self.reward_function)}")
                            else:
                                self.logger.error(f"Function {func_name} not found in eval_protocol.rewards")
                        except (ImportError, AttributeError) as e:
                            self.logger.error(f"Error importing from rewards module: {e}")

                if self.reward_function:
                    self.logger.info(
                        f"Successfully loaded reward function: {self.task_definition.reward_function_path}"
                    )
                    return True
                else:
                    self.logger.error(
                        f"Failed to load reward function from '{self.task_definition.reward_function_path}'"
                    )
                    return False
            except Exception as e:
                self.logger.error(f"Error loading reward function: {e}", exc_info=True)
                return False
        else:
            self.logger.error("Reward function path is mandatory but missing.")
            return False
        return True

    def _get_resource_class(self, resource_type_name: str) -> Type[ForkableResource]:
        # This method will now need to look into eval_protocol.agent_v2.resources
        # For example: from .resources import SQLResource, PythonStateResource etc.
        # And then map resource_type_name string to the class.
        # For now, a placeholder that would need specific imports or a registry.

        # Option 1: Direct mapping (requires importing all known resource types here)
        # from .resources import PythonStateResource, SQLResource, FileSystemResource, DockerResource # noqa

        mapping = {
            "PythonStateResource": PythonStateResource,
            "SQLResource": SQLResource,
            "FileSystemResource": FileSystemResource,
            "DockerResource": DockerResource,
            "BFCLSimAPIResource": BFCLSimAPIResource,  # Add BFCLSimAPIResource to mapping
        }
        resource_class = mapping.get(resource_type_name)

        if resource_class is None:
            raise ValueError(
                f"Resource class '{resource_type_name}' not found or not mapped in Orchestrator._get_resource_class."
            )
        # No need to check issubclass here if mapping is correct and types are imported.
        return cast(Type[ForkableResource], resource_class)

    async def setup_base_resource(self) -> None:
        resource_type = self.task_definition.resource_type
        base_config = self.task_definition.base_resource_config

        self.logger.info(f"Attempting to set up base resource of type '{resource_type}'...")
        try:
            ResourceClass = self._get_resource_class(resource_type)
            self.base_resource = ResourceClass()
            await self.base_resource.setup(base_config)
            self.logger.info(f"Base resource '{resource_type}' setup complete.")
        except ValueError as e_val:
            self.logger.error(f"Could not get resource class '{resource_type}'. {e_val}")
            self.base_resource = None
        except Exception as e_setup:
            self.logger.error(
                f"Failed to setup base resource '{resource_type}'. {e_setup}",
                exc_info=True,
            )
            self.base_resource = None

    async def _get_available_tools(self, episode_resource: ForkableResource) -> Dict[str, Callable[..., Any]]:
        available_tools: Dict[str, Callable[..., Any]] = {}
        if episode_resource:
            resource_tool_specs = await episode_resource.get_tools_spec()
            self.logger.debug(f"Raw tool specs from resource.get_tools_spec(): {resource_tool_specs}")
            for tool_spec in resource_tool_specs:
                # Corrected logic based on BFCLSimAPIResource._infer_schema_from_method output
                tool_name = tool_spec.get("name")
                if tool_name:
                    # Create an async adapter function that calls episode_resource.step
                    async def resource_tool_adapter(
                        params: Dict[str, Any],
                        bound_tool_name=tool_name,
                        bound_resource=episode_resource,
                    ):
                        # Ensure params are passed correctly to step
                        return await bound_resource.step(action_name=bound_tool_name, action_params=params)

                    available_tools[tool_name] = resource_tool_adapter
                    self.logger.debug(f"Added tool '{tool_name}' from resource spec.")
                else:
                    self.logger.warning(f"Skipping resource tool spec due to missing 'name': {tool_spec}")

        # Check for tools defined using ToolRegistry (more common pattern)
        if self.tools_module:
            self.logger.debug(f"Inspecting tools_module: {self.tools_module} (type: {type(self.tools_module)})")

            # First, try to find a ToolRegistry instance
            registry_instances = []
            for name, member in inspect.getmembers(self.tools_module):
                # Skip if it starts with underscore or is not a ToolRegistry
                if name.startswith("_"):
                    continue

                if hasattr(member, "get_tools") and callable(member.get_tools):
                    registry_instances.append((name, member))
                    self.logger.debug(f"Found ToolRegistry instance: {name}")

            if registry_instances:
                # Use the first registry instance found
                registry_name, registry = registry_instances[0]
                self.logger.info(f"Using ToolRegistry '{registry_name}' from module")

                # Get all tools from the registry
                registry_tools = registry.get_tools()
                for tool_name, tool_func in registry_tools.items():
                    # Create an adapter that will pass the resource to the tool
                    def create_tool_adapter(tool_func):
                        async def adapter(params: Dict[str, Any], bound_resource=episode_resource):
                            # Handle both sync and async functions
                            if asyncio.iscoroutinefunction(tool_func):
                                result = await tool_func(resource=bound_resource, **params)
                            else:
                                result = tool_func(resource=bound_resource, **params)
                            return result

                        return adapter

                    available_tools[tool_name] = create_tool_adapter(tool_func)
                    self.logger.debug(f"Added tool '{tool_name}' from registry {registry_name}")

                # If we found and used a registry, we're done
                if available_tools:
                    self.logger.info(f"Found {len(available_tools)} tools from ToolRegistry")
                    self.logger.debug(f"Tool names: {list(available_tools.keys())}")

            # If no registry tools were found, fall back to module inspection
            if not available_tools:
                self.logger.debug("No ToolRegistry found or no tools in registry. Falling back to module inspection.")

                members_to_inspect = []
                if inspect.ismodule(self.tools_module):
                    self.logger.debug("tools_module is a module. Using inspect.getmembers.")
                    members_to_inspect = inspect.getmembers(self.tools_module)
                elif hasattr(self.tools_module, "__dict__"):
                    self.logger.debug("tools_module is an object with __dict__. Iterating __dict__.items().")
                    members_to_inspect = self.tools_module.__dict__.items()
                else:
                    self.logger.debug("Falling back to inspect.getmembers.")
                    members_to_inspect = inspect.getmembers(self.tools_module)

                for name, member in members_to_inspect:
                    self.logger.debug(
                        f"Found member in tools_module: '{name}', type: {type(member)}, callable: {callable(member)}"
                    )
                    if name.startswith("_") or not callable(member):
                        self.logger.debug(f"Skipping member '{name}' (startswith_ or not callable).")
                        continue

                    # Check if it's a sync or async function
                    is_async = asyncio.iscoroutinefunction(member)
                    self.logger.debug(f"Member '{name}' is {'async' if is_async else 'sync'} function.")

                    try:
                        sig = inspect.signature(member)
                        resource_param_name = next(
                            (pname for pname in ["resource", "db_resource"] if pname in sig.parameters),
                            None,
                        )

                        if resource_param_name:

                            async def module_tool_adapter(
                                params: Dict[str, Any],
                                bound_tool_func=member,
                                bound_resource=episode_resource,
                                res_param_name=resource_param_name,
                                is_async=is_async,
                            ):
                                tool_kwargs = {res_param_name: bound_resource, **params}
                                if is_async:
                                    return await bound_tool_func(**tool_kwargs)
                                else:
                                    return bound_tool_func(**tool_kwargs)

                            available_tools[name] = module_tool_adapter
                            self.logger.debug(f"Added tool '{name}' from tools_module directly.")
                        else:
                            self.logger.debug(
                                f"Skipping module tool '{name}': no 'resource' or 'db_resource' parameter in signature '{sig}'."
                            )
                    except ValueError as e_sig:
                        self.logger.debug(f"Skipping module tool '{name}': could not get signature. Error: {e_sig}")
        self.logger.info(f"Combined available tools: {list(available_tools.keys())}")
        return available_tools

    async def execute_task_poc(self, sample_data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        if not await self._load_task_components():
            self.logger.error("Failed to load task components.")
            return None
        if not self.base_resource:
            await self.setup_base_resource()
        if not self.base_resource:
            self.logger.error("Base resource setup failed or not performed.")
            return None
        if not self.reward_function:
            self.logger.error("Reward function not loaded.")
            return None  # Should be caught by _load_task_components

        self.logger.info(f"Starting execution for task '{self.task_definition.name}'...")
        episode_resource: Optional[ForkableResource] = None
        evaluation_result: Optional[Dict[str, Any]] = None

        all_user_turns_successful_function_calls: List[
            List[Dict[str, Any]]
        ] = []  # Track successful calls for reward fn, list of lists (per user turn)
        conversation_messages: List[Dict[str, Any]] = []  # Use dicts for API compatibility

        # --- Agent Model Setup ---
        agent_model_name = os.environ.get("MODEL_AGENT")
        if not agent_model_name:
            self.logger.error("MODEL_AGENT environment variable not set.")
            return None
        if agent_model_name.startswith("openai/"):
            self._initialize_openai_client()
            if not self._openai_client:
                self.logger.error("OpenAI client failed to initialize. Cannot proceed.")
                return None
            agent_model_name = agent_model_name.split("openai/", 1)[1]  # Get actual model name
            self.logger.info(f"Using OpenAI model: {agent_model_name}")
        elif agent_model_name.startswith("fireworks/") or agent_model_name.startswith("accounts/fireworks"):
            self._initialize_fireworks_client()
            if not self._openai_client:
                self.logger.error("Fireworks client failed to initialize. Cannot proceed.")
                return None
            # Remove prefix if it exists
            if agent_model_name.startswith("fireworks/"):
                agent_model_name = agent_model_name.split("fireworks/", 1)[1]
            # If it starts with accounts/fireworks, keep the full model name
            self.logger.info(f"Using Fireworks model: {agent_model_name}")
        else:
            # Placeholder for other model providers if needed in the future
            self.logger.error(f"Unsupported model provider for MODEL_AGENT: {agent_model_name}")
            return None

        try:
            # --- Task Setup ---
            if not await self._load_task_components():
                self.logger.error("Failed to load task components.")
                return None
            if not self.base_resource:
                await self.setup_base_resource()
            if not self.base_resource:
                self.logger.error("Base resource setup failed or not performed.")
                return None
            if not self.reward_function:
                self.logger.error("Reward function not loaded.")
                return None

            self.logger.info("Forking base resource for episode...")
            episode_resource = await self.base_resource.fork()
            self.logger.info(f"Episode resource forked: {type(episode_resource).__name__}")

            # Initialize the episode resource with sample data if provided
            if sample_data:
                self.logger.info(f"Initializing episode resource with sample data: {sample_data}")
                initializer = getattr(episode_resource, "initialize", None)
                if callable(initializer):
                    await initializer(**sample_data)  # type: ignore[misc]
                else:
                    self.logger.warning(
                        f"Episode resource {type(episode_resource).__name__} does not have initialize method"
                    )

            # Get initial state for injection into first prompt (for HTTP rollout)
            initial_state_description = None
            get_init_state = getattr(episode_resource, "get_initial_state_description", None)
            if callable(get_init_state):
                try:
                    initial_state_description = await get_init_state()  # type: ignore[misc]
                    self.logger.info("Retrieved initial state description for first prompt")
                except Exception as e:
                    self.logger.warning(f"Failed to get initial state description: {e}")

            # --- Initial Conversation State ---
            # The conversation_messages list will be built turn by turn.
            # We need a copy of the user turns from the task definition.
            user_turns_from_task: List[Dict[str, Any]] = []
            if self.task_definition.messages:
                for msg_data in self.task_definition.messages:
                    if isinstance(msg_data, dict) and msg_data.get("role") == "user":
                        # Ensure it's a dict and has a role, content can be complex
                        user_turns_from_task.append(msg_data)
                    elif isinstance(msg_data, Message) and msg_data.role == "user":
                        user_turns_from_task.append(msg_data.model_dump(exclude_none=True))
                    else:
                        self.logger.warning(
                            f"Skipping non-user message or invalid message type in task definition's messages: {msg_data}"
                        )

            if not user_turns_from_task:
                self.logger.error("No user turns found in task definition's messages. Cannot proceed.")
                return None

            # --- Interaction Loop ---
            # Loop through the user turns defined in the task or up to poc_max_turns
            num_defined_user_turns = len(user_turns_from_task)
            max_interaction_turns = min(self.task_definition.poc_max_turns, num_defined_user_turns)

            current_user_turn_index = 0

            for turn_num in range(1, max_interaction_turns + 1):  # Outer loop for user turns
                self.logger.info(
                    f"--- User Turn {turn_num}/{max_interaction_turns} (Overall Index {current_user_turn_index + 1}/{num_defined_user_turns}) ---"
                )

                current_user_turn_accumulated_successful_calls: List[Dict[str, Any]] = []

                # Add the current user turn's message(s) to the conversation history
                if current_user_turn_index < num_defined_user_turns:
                    current_user_turn_message = user_turns_from_task[
                        current_user_turn_index
                    ].copy()  # Make a copy to avoid modifying the original

                    # Inject initial state into first user message
                    if current_user_turn_index == 0 and initial_state_description:
                        original_content = current_user_turn_message.get("content", "")
                        enhanced_content = f"{original_content}\n\n{initial_state_description}"
                        current_user_turn_message["content"] = enhanced_content
                        self.logger.info("Injected initial state into first user prompt")

                    # The user message content might be a string or a list of content blocks (e.g. for multi-modal)
                    # For BFCL, it's a string that might represent a JSON list of user messages for that turn.
                    # We need to parse it if it's a JSON string representing a list of messages.
                    try:
                        # Attempt to parse content if it's a string that looks like a JSON list
                        if isinstance(current_user_turn_message.get("content"), str):
                            parsed_content = json.loads(current_user_turn_message["content"])
                            if isinstance(parsed_content, list):
                                for sub_msg_dict in parsed_content:
                                    if (
                                        isinstance(sub_msg_dict, dict)
                                        and "role" in sub_msg_dict
                                        and "content" in sub_msg_dict
                                    ):
                                        conversation_messages.append(sub_msg_dict)
                                    else:
                                        self.logger.warning(
                                            f"Skipping sub-message in user turn due to invalid format: {sub_msg_dict}"
                                        )
                                        conversation_messages.append(
                                            current_user_turn_message
                                        )  # Fallback to original if parsing fails partially
                                        break  # Stop processing sub-messages for this turn
                                else:  # If loop completed without break
                                    pass  # Successfully processed all sub-messages
                            else:  # Content is a JSON string but not a list
                                conversation_messages.append(current_user_turn_message)
                        else:  # Content is not a string or already a complex object
                            conversation_messages.append(current_user_turn_message)
                    except json.JSONDecodeError:  # Content is a string but not valid JSON
                        conversation_messages.append(current_user_turn_message)

                    current_user_turn_index += 1
                else:
                    self.logger.info("No more user turns defined by task. Ending interaction.")
                    break  # Break outer loop if no more user messages from task def

                # 1. Get available tools for this user turn (can be dynamic based on resource state)
                # For BFCL, tools are generally static for the episode, but good practice to refresh.
                resource_tool_specs = await episode_resource.get_tools_spec()
                available_tools_adapters = await self._get_available_tools(
                    episode_resource
                )  # Get adapters for execution

                # Format tools for OpenAI API (should be done once per user turn, or if tools change)
                openai_tools: List[Dict[str, Any]] = []
                if OPENAI_AVAILABLE:
                    # First add tools from the resource
                    for spec in resource_tool_specs:
                        # Ensure spec has the structure with name and parameters
                        if "name" in spec and "parameters" in spec:
                            openai_tools.append(
                                {
                                    "type": "function",
                                    "function": {
                                        "name": spec["name"],
                                        "description": spec.get("description", ""),
                                        "parameters": spec["parameters"],  # Assuming OpenAI-compatible schema
                                    },
                                }
                            )
                        else:
                            self.logger.warning(f"Skipping tool spec due to missing name/parameters: {spec}")

                    # Now add tools from the registry
                    if (
                        self.tools_module
                        and hasattr(self.tools_module, "R")
                        and hasattr(self.tools_module.R, "get_openai_tools")
                    ):
                        registry_tools = self.tools_module.R.get_openai_tools()
                        for tool_spec in registry_tools:
                            openai_tools.append(
                                {
                                    "type": "function",
                                    "function": {
                                        "name": tool_spec["name"],
                                        "description": tool_spec.get("description", ""),
                                        "parameters": tool_spec["parameters"],
                                    },
                                }
                            )
                else:
                    self.logger.warning("OpenAI not available, cannot format tools for API.")

                if not available_tools_adapters and not openai_tools:  # If no tools can be formed or executed
                    self.logger.info(
                        "No tools available from resource or module for this turn. Agent cannot make tool calls."
                    )
                    # Agent might still respond textually. Let the loop proceed for one LLM call.

                # Inner loop for multi-step tool use within this single user turn
                current_inner_step = 0
                while current_inner_step < MAX_STEPS_PER_USER_TURN:
                    current_inner_step += 1
                    self.logger.info(
                        f"--- User Turn {turn_num}, Inner Step {current_inner_step}/{MAX_STEPS_PER_USER_TURN} ---"
                    )

                    # 2. Call the LLM (OpenAI)
                    try:
                        # Validate conversation messages for OpenAI API compliance
                        self._validate_conversation_messages(conversation_messages)

                        self.logger.debug(
                            f"Calling OpenAI: model={agent_model_name}, messages_FULL_HISTORY={json.dumps(conversation_messages, indent=2)}, tools={openai_tools}"
                        )  # Log full message history
                        if not self._openai_client:
                            raise Exception("OpenAI client not initialized")

                        # type: ignore[reportUnknownMemberType]
                        response = await self._openai_client.chat.completions.create(
                            model=agent_model_name,
                            messages=conversation_messages,  # type: ignore
                            tools=openai_tools if openai_tools else None,
                            tool_choice="auto" if openai_tools else None,
                            max_tokens=4096,
                            temperature=0.0,
                        )
                        response_message = response.choices[0].message
                        self.logger.debug(f"OpenAI response message: {response_message}")

                    except Exception as e_openai:
                        self.logger.error(f"Error calling OpenAI API: {e_openai}", exc_info=True)
                        # Break inner loop on API error, then outer loop will decide to continue or break.
                        # For now, let's break the outer loop as well to prevent cascading errors.
                        # TODO: Consider more nuanced error handling for outer loop.
                        evaluation_result = {"error": f"OpenAI API error: {e_openai}"}
                        # Clean up and return
                        if episode_resource:
                            await episode_resource.close()
                        if self.base_resource:
                            await self.base_resource.close()
                            self.base_resource = None
                        return evaluation_result

                    # 3. Process LLM Response
                    # Append assistant's response (content and tool calls) to history
                    conversation_messages.append(response_message.model_dump(exclude_none=True))

                    tool_calls = response_message.tool_calls
                    if tool_calls:
                        self.logger.info(f"Assistant requested {len(tool_calls)} tool calls in this step.")
                        current_llm_response_successful_calls: List[Dict[str, Any]] = []
                        for tool_call in tool_calls:
                            function_name = tool_call.function.name
                            function_args_str = tool_call.function.arguments
                            self.logger.info(f"Attempting tool call: {function_name}({function_args_str})")

                            tool_adapter = available_tools_adapters.get(function_name)
                            if tool_adapter:
                                try:
                                    function_args = json.loads(function_args_str)
                                    print("show function args: ", function_args)
                                    function_response = await tool_adapter(function_args)
                                    self.logger.info(
                                        f"Tool '{function_name}' result: {str(function_response)[:200]}..."
                                    )
                                    conversation_messages.append(
                                        {
                                            "tool_call_id": tool_call.id,
                                            "role": "tool",
                                            "name": function_name,
                                            "content": json.dumps(function_response),
                                        }
                                    )
                                    current_llm_response_successful_calls.append(
                                        {
                                            "name": function_name,
                                            "args": function_args,
                                        }
                                    )
                                except json.JSONDecodeError:
                                    self.logger.error(
                                        f"Failed to parse arguments for tool '{function_name}': {function_args_str}"
                                    )
                                    conversation_messages.append(
                                        {
                                            "tool_call_id": tool_call.id,
                                            "role": "tool",
                                            "name": function_name,
                                            "content": json.dumps({"error": "Invalid JSON arguments"}),
                                        }
                                    )
                                except Exception as e_tool_exec:
                                    self.logger.error(
                                        f"Error executing tool '{function_name}': {e_tool_exec}",
                                        exc_info=True,
                                    )
                                    conversation_messages.append(
                                        {
                                            "tool_call_id": tool_call.id,
                                            "role": "tool",
                                            "name": function_name,
                                            "content": json.dumps({"error": f"Execution failed: {e_tool_exec}"}),
                                        }
                                    )
                            else:
                                self.logger.error(
                                    f"Tool '{function_name}' requested by model but not found in available tools."
                                )
                                conversation_messages.append(
                                    {
                                        "tool_call_id": tool_call.id,
                                        "role": "tool",
                                        "name": function_name,
                                        "content": json.dumps({"error": "Tool not found"}),
                                    }
                                )

                        if current_llm_response_successful_calls:
                            current_user_turn_accumulated_successful_calls.extend(
                                current_llm_response_successful_calls
                            )

                        # If tool calls were made, continue the inner loop for the LLM to react to tool results.
                        if not openai_tools and not available_tools_adapters:  # No tools were ever available
                            self.logger.info(
                                "No tools were available, but LLM hallucinated tool calls. Breaking inner loop."
                            )
                            break  # Break inner loop
                    else:
                        # No tool calls from LLM in this step, means assistant provided a final textual response for this user turn.
                        self.logger.info(
                            "Assistant did not request tool calls in this step. Ending inner loop for this user turn."
                        )
                        break  # Break the inner while loop
                else:  # Inner while loop finished due to max_steps_per_user_turn
                    self.logger.warning(
                        f"Reached max steps ({MAX_STEPS_PER_USER_TURN}) for user turn {turn_num}. Ending inner loop."
                    )
                # End of inner while loop for multi-step tool use

                if current_user_turn_accumulated_successful_calls:
                    all_user_turns_successful_function_calls.append(current_user_turn_accumulated_successful_calls)
            # End of outer for loop for user turns

            # --- Evaluation ---
            self.logger.info("Evaluating task outcome...")
            task_achieved = False  # Reset task_achieved, as PoC logic is gone
            eval_criteria = self.task_definition.evaluation_criteria

            # Log evaluation_criteria and its relevant fields before calling reward function
            self.logger.debug(f"Evaluation criteria object: {eval_criteria}")
            if eval_criteria:
                self.logger.debug(
                    f"Evaluation criteria ground_truth_function_calls: {getattr(eval_criteria, 'ground_truth_function_calls', 'AttributeError or None')}"
                )
                self.logger.debug(
                    f"Evaluation criteria ground_truth_comparable_state: {getattr(eval_criteria, 'ground_truth_comparable_state', 'AttributeError or None')}"
                )

            # Check if episode_resource is SQLResource for final_state_query
            # from .resources import SQLResource # Would be needed here for isinstance
            if eval_criteria and eval_criteria.final_state_query:  # and isinstance(episode_resource, SQLResource):
                if hasattr(episode_resource, "step"):  # Generic check
                    query_res_step = await episode_resource.step(
                        "fetch_val_sql", {"query": eval_criteria.final_state_query}
                    )
                    if query_res_step.get("status") == "success":
                        outcome = query_res_step.get("result")
                        if eval_criteria.expected_query_result_transform:
                            try:
                                transform_func = eval(eval_criteria.expected_query_result_transform)
                                task_achieved = bool(transform_func(outcome))
                            except Exception as e_tf:
                                self.logger.error(f"Error applying transform: {e_tf}")
                        else:
                            task_achieved = bool(outcome)
                        self.logger.info(f"Final state query outcome: {outcome}, Task achieved: {task_achieved}")
                    else:
                        self.logger.error(f"Failed to execute final_state_query: {query_res_step.get('message')}")

            # TODO: Re-evaluate how task_achieved should be determined without PoC logic
            # Maybe based on final observation, specific tool calls, or reward function logic itself?

            # Log evaluation_criteria and its relevant fields before calling reward function
            self.logger.debug(f"Evaluation criteria object: {eval_criteria}")
            if eval_criteria:
                self.logger.debug(
                    f"Evaluation criteria ground_truth_function_calls: {getattr(eval_criteria, 'ground_truth_function_calls', 'AttributeError or None')}"
                )
                self.logger.debug(
                    f"Evaluation criteria ground_truth_comparable_state: {getattr(eval_criteria, 'ground_truth_comparable_state', 'AttributeError or None')}"
                )

            # Prepare ground_truth dictionary for the reward function
            ground_truth_for_reward = None
            if eval_criteria:
                ground_truth_for_reward = {
                    "function_calls": getattr(eval_criteria, "ground_truth_function_calls", None),
                    "comparable_state": getattr(eval_criteria, "ground_truth_comparable_state", None),
                }

            # Prepare state dictionary for reward function
            state_for_reward = {
                "resource": episode_resource,
                "successful_func_calls": all_user_turns_successful_function_calls,
                # Add other relevant state info if needed
            }

            # Prepare eval_args dictionary
            eval_args = {
                "messages": conversation_messages,  # Pass final conversation history (as dicts)
                "state": state_for_reward,
                "task_achieved": task_achieved,  # Still needs proper determination
                "task_definition_name": self.task_definition.name,
            }

            # Add ground_truth as a single parameter (not unpacked)
            if ground_truth_for_reward:
                eval_args["ground_truth"] = ground_truth_for_reward

            # Call the reward function
            self.logger.info("=== CALLING REWARD FUNCTION DEBUG ===")
            self.logger.info(f"Reward function type: {type(self.reward_function)}")
            self.logger.info(f"Eval args keys: {list(eval_args.keys())}")
            self.logger.info(f"Task achieved: {eval_args.get('task_achieved', 'NOT_SET')}")
            self.logger.info(f"Messages count: {len(eval_args.get('messages', []))}")
            evaluation_result = self.reward_function(**eval_args)
            self.logger.info("=== REWARD FUNCTION RESULT ===")
            self.logger.info(f"Reward function result: {evaluation_result}")
            self.logger.info(f"Result type: {type(evaluation_result)}")
            self.logger.info("=== END REWARD FUNCTION DEBUG ===")

            # Return both the evaluation result and the inputs for trajectory capture
            return {
                "evaluation_result": evaluation_result,
                "reward_function_inputs": {
                    "messages": conversation_messages,
                    "state": state_for_reward,
                    "task_achieved": task_achieved,
                    "task_definition_name": self.task_definition.name,
                    "ground_truth": ground_truth_for_reward,
                },
            }

        except Exception as e_lifecycle:
            self.logger.error(f"Exception during task lifecycle: {e_lifecycle}", exc_info=True)
            return {
                "evaluation_result": {"error": str(e_lifecycle)},
                "reward_function_inputs": None,
            }
        finally:
            if episode_resource:
                await episode_resource.close()
                self.logger.info("Episode resource closed.")
            if self.base_resource:
                await self.base_resource.close()
                self.base_resource = None
                self.logger.info("Base resource closed.")
        self.logger.info(f"Execution for task '{self.task_definition.name}' finished.")
        # This should not be reached normally since we return earlier, but handle edge case
        return {
            "evaluation_result": {"error": "Unexpected execution path"},
            "reward_function_inputs": None,
        }
