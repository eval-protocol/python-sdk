import inspect
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    overload,
)

from pydantic import TypeAdapter, ValidationError

# EvaluateResult and StepOutput are now extended/defined in models.py
from .models import (  # Removed StepOutput as it's not used here directly
    EvaluateResult,
    Message,
)

# Import resource types
from .resources import ResourceDict

_single_res_adapter = TypeAdapter(EvaluateResult)
_list_res_adapter = TypeAdapter(List[EvaluateResult])

# Define a type for the mode parameter
EvaluationMode = Literal["pointwise", "batch"]

# TypeVar for the function being decorated, to preserve its signature as much as possible.
F = TypeVar("F", bound=Callable[..., Any])


# Precise overloads help static type checkers preserve the original function signature.
@overload
def reward_function(
    _func: F,
    *,
    mode: EvaluationMode = "pointwise",
    id: Optional[str] = None,
    requirements: Optional[List[str]] = None,
    resources: Optional[ResourceDict] = None,
    concurrency: Optional[int] = None,
    timeout: Optional[int] = None,
) -> F: ...


@overload
def reward_function(
    _func: None = ...,  # when used as @reward_function(...)
    *,
    mode: EvaluationMode = "pointwise",
    id: Optional[str] = None,
    requirements: Optional[List[str]] = None,
    resources: Optional[ResourceDict] = None,
    concurrency: Optional[int] = None,
    timeout: Optional[int] = None,
) -> Callable[[F], F]: ...


def reward_function(
    _func: Optional[F] = None,
    *,
    mode: EvaluationMode = "pointwise",
    id: Optional[str] = None,
    requirements: Optional[List[str]] = None,  # Changed to List[str]
    resources: Optional[ResourceDict] = None,  # Resource management
    concurrency: Optional[int] = None,
    timeout: Optional[int] = None,
) -> Union[F, Callable[[F], F]]:
    """
    Decorator for user-defined reward and evaluation functions with resource management.

    It handles:
    - Coercing input messages (and ground truths if applicable) to Pydantic `Message` objects
      if the decorated function is type-hinted to receive them. This part currently targets
      parameters named 'messages' and 'ground_truth'.
    - Validating that the output conforms to `EvaluateResult` (for pointwise) or `List[EvaluateResult]` (for batch).
    - Managing declared resources (LLMs, databases, etc.) with automatic setup and cleanup

    Args:
        _func: The user's reward/evaluation function. Optional for decorator usage with args.
        mode: Specifies the operational mode. Defaults to "pointwise".
              - "pointwise": Function processes one rollout. Expected output: `EvaluateResult`.
              - "batch": Function processes a batch of rollouts. Expected output: `List[EvaluateResult]`.
        id: Optional identifier for the reward function, used for deployment
        requirements: Optional string content for requirements.txt for deployment
        resources: Optional dictionary of resource types to resource instances.
                  Example: {"llms": [llm_resource]}
                  Resources are automatically setup before evaluation and cleaned up after.
        concurrency: Optional number of concurrent requests to the reward function. This will only take effect if the function is async or there are async resources binded to the reward function (e.g. LLM resource).
        timeout: Optional timeout for the reward function. This will only take effect if the function is async or there are async resources binded to the reward function (e.g. LLM resource).

    Returns:
        A decorator if `_func` is None, or the decorated function.
    """

    def decorator(func: F) -> F:
        sig = inspect.signature(func)
        params = sig.parameters

        # Validate that the function accepts **kwargs
        has_var_keyword = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())

        if not has_var_keyword:
            raise ValueError(
                f"Function '{func.__name__}' must accept **kwargs parameter. Please add '**kwargs' to the function signature."
            )

        # Setup resources once when the decorator is applied
        resource_managers = {}
        if resources:
            for resource_type, resource_list in resources.items():
                managers = []
                for resource in resource_list:
                    resource.setup()
                    managers.append(resource)
                resource_managers[resource_type] = managers

        # Detect if the user supplied function is a coroutine (async def)
        _is_async_function = inspect.iscoroutinefunction(func)

        def _is_list_of_message_annotation(annotation: Any) -> bool:
            origin = get_origin(annotation)
            args = get_args(annotation)
            # Direct List[Message]
            if origin in (list, List) and args and args[0] == Message:
                return True
            # Optional[List[Message]] or Union[List[Message], None]
            if origin is Union and args:
                # Filter out NoneType
                non_none = [a for a in args if a is not type(None)]  # noqa: E721
                if len(non_none) == 1:
                    inner = non_none[0]
                    inner_origin = get_origin(inner)
                    inner_args = get_args(inner)
                    return (inner_origin in (list, List)) and bool(inner_args) and (inner_args[0] == Message)
            return False

        def _prepare_final_args(*args: Any, **kwargs: Any):
            """Prepare final positional and keyword arguments for the user function call.
            This includes Pydantic coercion and resource injection. Returns a tuple of
            (call_args, call_kwargs).
            """
            # Bind arguments to handle *args and **kwargs correctly for the wrapped function
            bound_args = sig.bind_partial(*args, **kwargs)
            bound_args.apply_defaults()

            # Create a mutable copy of arguments to modify
            final_func_args = dict(bound_args.arguments)

            def _coerce_to_list_message(data_list: Any, arg_name_for_error: str) -> List[Message]:
                if not isinstance(data_list, list):
                    raise TypeError(f"Expected a list for '{arg_name_for_error}', got {type(data_list)}")
                typed_list = []
                for i, item_data in enumerate(data_list):
                    if isinstance(item_data, Message):
                        typed_list.append(item_data)
                    elif isinstance(item_data, dict):
                        typed_list.append(Message.model_validate(item_data))
                    else:
                        raise TypeError(f"Unexpected type for item {i} in '{arg_name_for_error}': {type(item_data)}")
                return typed_list

            # 1. Conditional Pydantic conversion for 'messages' (pointwise) or 'rollouts_messages' (batch)
            if mode == "pointwise" and "messages" in params and "messages" in final_func_args:
                messages_param_annotation = params["messages"].annotation
                if _is_list_of_message_annotation(messages_param_annotation):
                    try:
                        final_func_args["messages"] = _coerce_to_list_message(final_func_args["messages"], "messages")
                    except Exception as err:
                        raise ValueError(f"Input 'messages' failed Pydantic validation: {err}") from None

            elif mode == "batch" and "rollouts_messages" in params and "rollouts_messages" in final_func_args:
                param_annotation = params["rollouts_messages"].annotation
                inner = get_args(param_annotation)[0] if get_args(param_annotation) else None
                if get_origin(param_annotation) == list and inner and get_origin(inner) == list:
                    if get_args(inner) and get_args(inner)[0] == Message:
                        try:
                            coerced_rollouts = []
                            for i, rollout_data in enumerate(final_func_args["rollouts_messages"]):
                                coerced_rollouts.append(
                                    _coerce_to_list_message(rollout_data, f"rollouts_messages[{i}]")
                                )
                            final_func_args["rollouts_messages"] = coerced_rollouts
                        except Exception as err:
                            raise ValueError(f"Input 'rollouts_messages' failed Pydantic validation: {err}") from None

            # Ground truth coercion (if needed)
            if "ground_truth" in params and "ground_truth" in final_func_args:
                gt_ann = params["ground_truth"].annotation
                if _is_list_of_message_annotation(gt_ann):
                    if final_func_args["ground_truth"] is not None:
                        gt_val = final_func_args["ground_truth"]
                        try:
                            if isinstance(gt_val, list):
                                final_func_args["ground_truth"] = _coerce_to_list_message(gt_val, "ground_truth")
                            elif isinstance(gt_val, dict):
                                final_func_args["ground_truth"] = _coerce_to_list_message([gt_val], "ground_truth")
                            elif isinstance(gt_val, str):
                                final_func_args["ground_truth"] = _coerce_to_list_message(
                                    [{"role": "system", "content": gt_val}], "ground_truth"
                                )
                        except Exception as err:
                            raise ValueError(
                                f"Input 'ground_truth' failed Pydantic validation for List[Message]: {err}"
                            ) from None

            # Inject resource clients into kwargs (resources are already setup)
            if resource_managers:
                final_func_args["resources"] = {
                    resource_type: [manager.get_client() for manager in managers]
                    for resource_type, managers in resource_managers.items()
                }

            # Call the author's function using the (potentially modified) arguments dictionary.
            # final_func_args should contain all parameters expected by func, correctly mapped.
            # Reconstruct args and kwargs for the call to func
            call_args: List[Any] = []
            call_kwargs: Dict[str, Any] = {}
            for (
                p_name,
                p_obj,
            ) in params.items():  # params from inspect.signature(func).parameters
                if p_obj.kind == inspect.Parameter.VAR_POSITIONAL:
                    # If original func had *pos_args, final_func_args might contain it as a tuple
                    call_args.extend(final_func_args.get(p_name, ()))
                elif p_obj.kind == inspect.Parameter.VAR_KEYWORD:  # **kwargs
                    # If original func had **kw_args, final_func_args contains the dict of these
                    call_kwargs.update(final_func_args.get(p_name, {}))
                elif p_name in final_func_args:  # Named parameters
                    if p_obj.kind == inspect.Parameter.POSITIONAL_ONLY:
                        call_args.append(final_func_args[p_name])
                    else:  # POSITIONAL_OR_KEYWORD, KEYWORD_ONLY
                        call_kwargs[p_name] = final_func_args[p_name]

            return call_args, call_kwargs

        def _validate_output(result: Any):
            if mode == "pointwise":
                if isinstance(result, EvaluateResult):
                    return result
                return _single_res_adapter.validate_python(result)
            elif mode == "batch":
                if isinstance(result, list) and all(isinstance(item, EvaluateResult) for item in result):
                    return result
                return _list_res_adapter.validate_python(result)
            else:
                raise ValueError(f"Internal error: Invalid mode '{mode}' in wrapper.")

        if _is_async_function:

            @wraps(func)
            async def async_wrapper(
                *args: Any,
                **kwargs: Any,
            ) -> Union[EvaluateResult, List[EvaluateResult]]:
                call_args, call_kwargs = _prepare_final_args(*args, **kwargs)
                result = await func(*call_args, **call_kwargs)  # type: ignore[misc]
                try:
                    return _validate_output(result)
                except ValidationError as err:
                    raise ValueError(
                        f"Return value from function '{func.__name__}' failed Pydantic validation for mode '{mode}':\n{err}"
                    ) from None

            wrapper_fn = async_wrapper

        else:

            @wraps(func)
            def sync_wrapper(
                *args: Any,
                **kwargs: Any,
            ) -> Union[EvaluateResult, List[EvaluateResult]]:
                call_args, call_kwargs = _prepare_final_args(*args, **kwargs)
                result = func(*call_args, **call_kwargs)
                try:
                    return _validate_output(result)
                except ValidationError as err:
                    raise ValueError(
                        f"Return value from function '{func.__name__}' failed Pydantic validation for mode '{mode}':\n{err}"
                    ) from None

            wrapper_fn = sync_wrapper

        # Set attributes for introspection and deployment
        wrapper_fn._reward_function_id = id  # type: ignore[attr-defined]
        wrapper_fn._reward_function_requirements = requirements  # type: ignore[attr-defined]
        wrapper_fn._reward_function_mode = mode  # type: ignore[attr-defined]
        wrapper_fn._reward_function_resources = resources  # type: ignore[attr-defined]
        wrapper_fn._reward_function_timeout = timeout  # type: ignore[attr-defined]
        wrapper_fn._reward_function_concurrency = concurrency  # type: ignore[attr-defined]

        return cast(F, wrapper_fn)

    if _func is None:  # Decorator called with arguments, e.g., @reward_function(mode="batch")
        return decorator
    else:  # Decorator called without arguments, e.g., @reward_function (defaults to pointwise)
        return decorator(_func)
