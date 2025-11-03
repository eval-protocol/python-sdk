import importlib
import importlib.util
import inspect
import logging
import os
import warnings
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

import requests

from .models import EvaluateResult, MetricResult
from .typed_interface import reward_function

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Callable[..., EvaluateResult])


class RewardFunction:
    """
    A wrapper for reward functions that allows them to be run locally or remotely.

    The RewardFunction class wraps a reward function (either a local function or a remote endpoint)
    and provides a unified interface for calling it. It supports:

    - Local functions (mode="local")
    - Remote endpoints (mode="remote")
    - Fireworks-hosted models (mode="fireworks_hosted")

    Args:
        func: The local function to use (for mode="local")
        func_path: A string path to a function (e.g., "module.submodule:function_name")
        mode: The mode of operation ("local", "remote", or "fireworks_hosted")
        endpoint: The URL of the remote endpoint (for mode="remote")
        model_id: The ID of the Fireworks-hosted model (for mode="fireworks_hosted")
        **kwargs: Additional keyword arguments to pass to the function
    """

    def __init__(
        self,
        func: Optional[Callable] = None,
        func_path: Optional[str] = None,
        mode: str = "local",
        endpoint: Optional[str] = None,
        name: Optional[str] = None,
        model_id: Optional[str] = None,
        **kwargs,
    ):
        self.mode = mode
        self.func = func
        self.func_path = func_path
        self.endpoint = endpoint
        self.name = name
        self.model_id = model_id
        self.kwargs = kwargs

        if mode == "local":
            if func is None and func_path is None:
                raise ValueError("Either 'func' or 'func_path' must be provided for local mode")
            if func_path and func is None:
                self.func = self._load_function_from_path(func_path)
        elif mode == "remote":
            if endpoint is None and name is None:
                raise ValueError("Either 'endpoint' or 'name' must be provided for remote mode")
            if name and endpoint is None:
                self.endpoint = f"https://api.fireworks.ai/v1/reward/{name}"
        elif mode == "fireworks_hosted":
            if model_id is None:
                raise ValueError("'model_id' must be provided for fireworks_hosted mode")
            self.endpoint = f"https://api.fireworks.ai/v1/models/{model_id}/reward"
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _load_function_from_path(self, func_path: str) -> Callable:
        """
        Load a function from a path string.
        The path string should be in the format 'module.submodule:function_name' or 'module.submodule.function_name'.
        """
        # Check for the colon format first (preferred)
        if ":" in func_path:
            module_path, func_name = func_path.split(":", 1)

            try:
                module = importlib.import_module(module_path)
                func = getattr(module, func_name)
                return func
            except (ImportError, AttributeError) as e:
                raise ImportError(f"Failed to load function from path {func_path}: {str(e)}")

        # Try dot notation format: module.path.function_name
        # This assumes the last component is the function name
        parts = func_path.split(".")
        if len(parts) < 2:
            raise ValueError(
                f"Invalid func_path format: {func_path}, expected 'module.path:function_name' or 'module.path.function_name'"
            )

        module_path = ".".join(parts[:-1])
        func_name = parts[-1]

        try:
            module = importlib.import_module(module_path)
            func = getattr(module, func_name)
            return func
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to load function from path {func_path}: {str(e)}")

    def __call__(
        self,
        messages: List[Dict[str, str]],
        ground_truth: Optional[Union[str, List[Dict[str, str]]]] = None,
        **kwargs,
    ) -> EvaluateResult:
        """
        Call the reward function with the provided messages.

        Args:
            messages: List of conversation messages, each with 'role' and 'content' keys
            ground_truth: Ground truth data, which can be a string (e.g., an expected answer)
                          or a list of original conversation messages (for context).
                          If None and context is expected as a list, defaults to messages[:-1].
            **kwargs: Additional keyword arguments to pass to the function

        Returns:
            EvaluateResult object with score and metrics
        """
        if ground_truth is None:
            # Default to messages[:-1] if ground_truth is not provided and expected as context (list)
            # This maintains previous behavior of original_messages defaulting.
            # If ground_truth is meant to be a string and is None, it should be handled by the specific reward function.
            ground_truth = messages[:-1] if messages else []

        combined_kwargs = {**self.kwargs, **kwargs}

        if self.mode == "local":
            if self.func is None:
                raise ValueError("No function provided for local mode")

            try:
                result = self.func(
                    messages=messages,
                    ground_truth=ground_truth,
                    **combined_kwargs,
                )

                if isinstance(result, EvaluateResult):
                    return result
                elif isinstance(result, tuple) and len(result) == 2:
                    # Handle legacy (score, components) tuple format
                    warnings.warn(
                        "Tuple return format is deprecated. Use EvaluateResult instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    score, components = result
                    metrics = {
                        k: MetricResult(score=v, reason=f"{k} score", is_score_valid=True)
                        for k, v in components.items()
                    }
                    return EvaluateResult(score=score, metrics=metrics)
                elif isinstance(result, dict) and "score" in result:
                    # Handle dictionary return format
                    warnings.warn(
                        "Dictionary return format is deprecated. Use EvaluateResult instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    metrics = {}
                    if "metrics" in result:
                        for k, v in result["metrics"].items():
                            if isinstance(v, dict):
                                metrics[k] = MetricResult(
                                    score=v.get("score", 0.0),
                                    reason=v.get("reason", f"{k} score"),
                                    is_score_valid=v.get("is_score_valid", True),
                                )
                            else:
                                metrics[k] = MetricResult(
                                    score=float(v),
                                    reason=f"{k} score",
                                    is_score_valid=True,
                                )
                    return EvaluateResult(
                        score=result["score"],
                        reason=result.get("reason"),
                        metrics=metrics,
                    )
                else:
                    raise TypeError(
                        f"Invalid return type from reward function: {type(result)}. "
                        f"Expected EvaluateResult or (float, Dict[str, float]) tuple."
                    )

            except Exception as e:
                logger.error(f"Error calling local reward function: {str(e)}")
                raise

        elif self.mode in ["remote", "fireworks_hosted"]:
            if self.endpoint is None:
                raise ValueError(f"No endpoint provided for {self.mode} mode")

            payload = {
                "messages": messages,
                "ground_truth": ground_truth,
                **combined_kwargs,
            }

            api_key = os.environ.get("FIREWORKS_API_KEY")
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}" if api_key else "",
            }

            try:
                response = requests.post(self.endpoint, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()

                if isinstance(result, dict) and "score" in result:
                    metrics = {}
                    if "metrics" in result:
                        for k, v in result["metrics"].items():
                            if isinstance(v, dict):
                                metrics[k] = MetricResult(
                                    score=v.get("score", 0.0),
                                    reason=v.get("reason", f"{k} score"),
                                    is_score_valid=v.get("success", True),
                                )
                            else:
                                metrics[k] = MetricResult(
                                    score=float(v),
                                    reason=f"{k} score",
                                    is_score_valid=True,
                                )

                    return EvaluateResult(
                        score=result["score"],
                        reason=result.get("reason"),
                        metrics=metrics,
                    )
                else:
                    raise ValueError(f"Invalid response from remote endpoint: {result}")

            except Exception as e:
                logger.error(f"Error calling remote endpoint: {str(e)}")
                raise

        raise ValueError(f"Invalid mode: {self.mode}")

    def get_trl_adapter(self) -> Callable:
        """
        Create an adapter function for use with TRL library.

        The TRL library expects a function that takes batch inputs and returns a batch of reward values.
        This adapter handles:
        1. Batch of messages (List[List[Dict]]) and original messages (List[List[Dict]])
        2. Batch of texts (List[str]) for simpler cases

        Returns:
            A callable function compatible with TRL's expected signature for reward functions.
        """

        def adapter(prompts: List[List[Dict]], completions: Optional[List[str]] = None, **kwargs) -> List[float]:
            """
            Adapter function compatible with TRL's expected reward function signature.
            TRL typically expects: (prompts: List[str], completions: List[str], **kwargs: Any) -> List[float]
            This adapter handles the conversion from reward-kit's Message format.

            Args:
                prompts: A batch of prompt message lists as expected by this RewardFunction instance.
                         Typically List[List[Dict[str, str]]], e.g.,
                         [[{'role':'system',...}, {'role':'user',...}], ...]
                completions: A batch of generated completion strings by the model.
                             Optional. If None, it's assumed that the `prompts` argument
                             already contains the full conversation history including the assistant's response.
                **kwargs: Additional keyword arguments passed by TRL. These often include
                          other columns from the HuggingFace dataset being used for training
                          (e.g., 'solution', 'reference_answer'). These are expected to be
                          lists of the same length as `prompts`.

            Returns:
                A list of float reward scores for the batch, one score per sample.
            """
            results = []
            batch_size = len(prompts)

            # If completions is None, assume prompts already contain complete conversations
            if completions is None:
                completions = [""] * batch_size

            # Make sure completions has the right length after the None check
            if batch_size != len(completions):
                logger.warning(
                    f"Batch size mismatch between prompts ({batch_size}) and "
                    f"completions ({len(completions)}). Using min length."
                )
                batch_size = min(batch_size, len(completions))

            # Extract potential ground truth solutions if available
            # TRL passes columns from the dataset that weren't removed.
            # We expect 'solution' based on our grpo_example.py setup.
            solutions = kwargs.get("solution", [None] * batch_size)
            if not isinstance(solutions, list) or len(solutions) != batch_size:
                logger.warning(
                    f"Expected 'solution' kwarg to be a list of size {batch_size}, but got {type(solutions)}. Ground truth might not be passed correctly."
                )
                solutions = [None] * batch_size

            for i in range(batch_size):
                completion_input = completions[i]
                actual_completion_str = ""

                if isinstance(completion_input, list):
                    if completion_input:
                        first_element = completion_input[0]
                        if (
                            isinstance(first_element, dict)
                            and "content" in first_element
                            and isinstance(first_element.get("role"), str)
                            and first_element.get("role") == "assistant"
                        ):
                            # Expected structure: completions[i] = [{'role': 'assistant', 'content': 'str_content'}]
                            actual_completion_str = str(first_element.get("content", ""))
                            logger.debug(
                                f"Adapter: completions[{i}] is a list with an assistant message dict. Extracted content."
                            )
                        else:
                            logger.warning(
                                f"Adapter: completions[{i}] is a list, but its first element "
                                f"is not the expected assistant message dict or is malformed: {first_element}. "
                                f"Using str(first_element) as content."
                            )
                            actual_completion_str = str(first_element)
                    else:
                        logger.warning(f"Adapter: completions[{i}] is an empty list. Using empty string for content.")
                        actual_completion_str = ""
                elif isinstance(completion_input, str):
                    actual_completion_str = completion_input
                else:
                    # Fallback for other types (e.g. a direct dict, though less likely given warnings)
                    logger.warning(
                        f"Adapter: completions[{i}] is of unexpected type: {type(completion_input)}. "
                        f"Attempting to stringify for content: {completion_input}"
                    )
                    actual_completion_str = str(completion_input)

                messages = prompts[i] + [{"role": "assistant", "content": actual_completion_str}]

                # Prepare kwargs for the underlying reward function call for this specific sample
                call_kwargs: Dict[str, Any] = {}  # Initialize with Any type for values
                current_solution = solutions[i]

                debug_solution_val_str = str(current_solution) if current_solution is not None else "None"
                logger.debug(
                    f"Adapter loop i={i}, type(current_solution)={type(current_solution)}, value='{debug_solution_val_str[:100]}...'"
                )

                processed_solution_val: Optional[str] = None
                if current_solution is not None:
                    if isinstance(current_solution, list):
                        logger.warning(
                            f"Sample {i} solution is a list, attempting to use first element: {current_solution}"
                        )
                        if current_solution:
                            processed_solution_val = str(current_solution[0])
                        # If current_solution is an empty list, processed_solution_val remains None
                    else:
                        processed_solution_val = str(current_solution)

                if processed_solution_val is not None:
                    call_kwargs["solution"] = processed_solution_val

                # Add any other necessary kwargs (if they were extracted from the main **kwargs)
                # For now, only "solution" is dynamically handled from TRL kwargs.

                try:
                    # Call the underlying RewardFunction instance (__call__)
                    # Pass the constructed messages and the extracted kwargs for this sample
                    result = self(
                        messages=messages,
                        ground_truth=(
                            call_kwargs.pop("solution") if "solution" in call_kwargs else None
                        ),  # Pass solution as ground_truth if available
                        **call_kwargs,
                    )
                    # Handle both RewardOutput and EvaluateResult
                    score = result.score
                    results.append(score)
                except Exception as e:
                    logger.error(f"Error processing sample {i} in TRL adapter: {str(e)}")
                    # Append a default low score (e.g., 0.0) on error
                    results.append(0.0)

            return results

        return adapter


# The legacy_reward_function decorator has been removed as it is no longer needed.
# Use the reward_function decorator from typed_interface instead.
#
# For deployment functionality, use the RewardFunction class or the deployment
# methods from the evaluation module directly.


# The alias below is removed to ensure that `from .typed_interface import reward_function`
# is the one used throughout the library, thus avoiding the deprecation warning
# when using the @reward_function decorator.
