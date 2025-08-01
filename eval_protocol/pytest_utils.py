import asyncio
import inspect
from typing import Any, Callable, Dict, List, Optional

import pytest
from openai import OpenAI

from .auth import get_fireworks_api_base, get_fireworks_api_key
from .common_utils import load_jsonl
from .models import CompletionParams, EvaluateResult, EvaluationRow, InputMetadata, Message


def evaluate(
    rows: List[EvaluationRow], reward_fn: Callable[..., EvaluateResult], **kwargs: Any
) -> List[EvaluationRow]:
    """Apply a reward function to each row and attach the result."""
    evaluated: List[EvaluationRow] = []
    for row in rows:
        result = reward_fn(messages=row.messages, ground_truth=row.ground_truth, **kwargs)
        row.evaluation_result = result
        evaluated.append(row)
    return evaluated


def _aggregate(scores: List[float], method: str) -> float:
    if not scores:
        return 0.0
    if method == "mean":
        return sum(scores) / len(scores)
    if method == "max":
        return max(scores)
    if method == "min":
        return min(scores)
    raise ValueError(f"Unknown aggregation method: {method}")


"""
Parameter types
"""
ModelParam = str  # gpt-4o, gpt-4o-mini, accounts/fireworks/models/llama-3.1-8b-instruct
DatasetPathParam = str
InputParam = Dict[str, Any]
InputMessagesParam = List[Message]

Dataset = List[EvaluationRow]
"""
Test function types
"""
TestFunction = Callable[..., Dataset]


def default_rollout_processor(row: EvaluationRow, model: ModelParam, input_params: InputParam) -> List[EvaluationRow]:
    """Generate a single response from a Fireworks model."""

    api_key = get_fireworks_api_key()
    api_base = get_fireworks_api_base()
    client = OpenAI(api_key=api_key, base_url=f"{api_base}/inference/v1")

    messages_payload = [{"role": m.role, "content": m.content} for m in row.messages]

    response = client.chat.completions.create(model=model, messages=messages_payload, **input_params)
    assistant_content = response.choices[0].message.content or ""
    messages = list(row.messages) + [Message(role="assistant", content=assistant_content)]
    processed = EvaluationRow(
        messages=messages,
        ground_truth=row.ground_truth,
        input_metadata=InputMetadata(completion_params=CompletionParams(model=model)),
    )
    return [processed]


def evaluation_test(
    *,
    model: List[ModelParam],
    input_messages: Optional[List[InputMessagesParam]] = None,
    input_dataset: Optional[List[DatasetPathParam]] = None,
    dataset_adapter: Optional[Callable[[List[Dict[str, Any]]], Dataset]] = lambda x: x,
    input_params: Optional[List[InputParam]] = None,
    rollout_processor: Callable[
        [EvaluationRow, ModelParam, InputParam], List[EvaluationRow]
    ] = default_rollout_processor,
    aggregation_method: str = "mean",
    threshold_of_success: Optional[float] = None,
    num_runs: int = 1,
    max_dataset_rows: Optional[int] = None,
) -> Callable[
    [TestFunction],
    TestFunction,
]:
    """Decorator to create pytest-based evaluation tests.

    Args:
        model: Model identifiers to query.
        input_messages: Messages to send to the model. This is useful if you
            don't have a dataset but can hard-code the messages.
        input_dataset: Paths to JSONL datasets. This is useful if you have a
            dataset already. Provide a dataset_adapter to convert the input dataset
            to a list of EvaluationRows if you have a custom dataset format.
        dataset_adapter: Function to convert the input dataset to a list of
            EvaluationRows. This is useful if you have a custom dataset format.
        input_params: Generation parameters for the model.
        rollout_processor: Function used to perform the rollout.
        aggregation_method: How to aggregate scores across rows.
        threshold_of_success: If set, fail the test if the aggregated score is
            below this threshold.
        num_runs: Number of times to repeat the evaluation.
        max_dataset_rows: Limit dataset to the first N rows.

    Usage:
    With an input dataset and input params, the test function will be called with the following arguments:

    ```python
    @evaluation_test(
        model=["gpt-4o", "gpt-4o-mini"],
        input_dataset=["data/test.jsonl"],
        input_params=[{"temperature": 0.5}],
        rollout_processor=default_rollout_processor,
        aggregation_method="mean",
    )
    def test_func(dataset_path: str, model_name: str, input_params: Dict[str, Any]):
        pass
    ```

    Without an input dataset and input params, the test function will be called with the following arguments:

    ```python
    @evaluation_test(
        model=["gpt-4o", "gpt-4o-mini"],
    )
    def test_func(model_name: str):
        pass
    ```

    With model and input_messages, the test function will be called with the following arguments:

    ```python
    @evaluation_test(
        model=["gpt-4o", "gpt-4o-mini"],
        input_messages=[{"role": "user", "content": "Hello, how are you?"}],
    )
    def test_func(model_name: str, input_messages: List[List[Message]]):
        pass
    ```
    """

    def decorator(
        test_func: TestFunction,
    ):
        # Check if the function is async
        is_async = inspect.iscoroutinefunction(test_func)

        def execute_with_params(
            test_func: TestFunction,
            model: str,
            input_dataset: List[EvaluationRow] | None = None,
            input_params: InputParam | None = None,
            input_messages: InputMessagesParam | None = None,
        ):
            kwargs = {}
            if input_dataset is not None:
                kwargs["input_dataset"] = list(input_dataset)
            if input_params is not None:
                kwargs["input_params"] = input_params
            if model is not None:
                kwargs["model"] = model
            if input_messages is not None:
                kwargs["input_messages"] = input_messages
            if is_async:
                # Handle async functions with proper event loop management
                try:
                    loop = asyncio.get_event_loop()
                    if not loop.is_closed():
                        # Use existing loop
                        task = loop.create_task(test_func(**kwargs))
                        results = loop.run_until_complete(task)
                    else:
                        # Loop is closed, create a new one
                        results = asyncio.run(test_func(**kwargs))
                except RuntimeError:
                    # No event loop or other issues, create a new one
                    results = asyncio.run(test_func(**kwargs))
            else:
                results = test_func(**kwargs)
            return results

        # Calculate all possible combinations of parameters
        def generate_combinations(model: List[ModelParam]):
            combinations = []

            # Always include models
            model_list = model

            # Handle optional parameters with defaults
            datasets: List[Optional[DatasetPathParam]] = input_dataset if input_dataset is not None else [None]  # type: ignore
            params: List[Optional[InputParam]] = input_params if input_params is not None else [None]  # type: ignore
            messages: List[Optional[InputMessagesParam]] = input_messages if input_messages is not None else [None]  # type: ignore

            # Generate all combinations
            for m in model_list:
                for ds in datasets:
                    for ip in params:
                        for im in messages:
                            # Skip combinations that don't make sense
                            # If we have a dataset, we should have params for rollout
                            if ds is not None and ip is None:
                                continue
                            # If we have messages but no dataset, that's fine
                            # If we have no dataset and no messages, that's also fine
                            combinations.append((m, ds, ip, im))

            return combinations

        combinations = generate_combinations(model)

        # Create parameter tuples for pytest.mark.parametrize
        param_tuples = []
        for combo in combinations:
            model_name, dataset, params, messages = combo
            param_tuple = [model_name]
            if input_dataset is not None:
                param_tuple.append(dataset)
            if input_params is not None:
                param_tuple.append(params)
            if input_messages is not None:
                param_tuple.append(messages)
            param_tuples.append(tuple(param_tuple))

        # Determine the parameter names for the test function
        test_param_names = ["model"]
        if input_dataset is not None:
            test_param_names.append("dataset_path")
        if input_params is not None:
            test_param_names.append("input_params")
        if input_messages is not None:
            test_param_names.append("input_messages")

        # Create wrapper function with exact signature that pytest expects
        def create_wrapper_with_signature():
            # Create the function body that will be used
            def wrapper_body(**kwargs):
                model_name = kwargs["model"]

                # Handle dataset loading if dataset_path is provided
                input_dataset = None
                if "dataset_path" in kwargs and kwargs["dataset_path"] is not None:
                    data = load_jsonl(kwargs["dataset_path"])
                    if max_dataset_rows is not None:
                        data = data[:max_dataset_rows]
                    data = dataset_adapter(data)
                    input_dataset: List[EvaluationRow] = []
                    for row in data:
                        processed = rollout_processor(
                            row, model=model_name, input_params=kwargs.get("input_params") or {}
                        )
                        input_dataset.extend(processed)

                all_results: List[EvaluationRow] = []
                for _ in range(num_runs):
                    # Each run reuses the same processed rows
                    results = execute_with_params(
                        test_func,
                        model=model_name,
                        input_dataset=input_dataset,
                        input_params=kwargs.get("input_params") if "input_params" in kwargs else None,
                        input_messages=kwargs.get("input_messages") if "input_messages" in kwargs else None,
                    )
                    if results is None:
                        raise ValueError(
                            f"Test function {test_func.__name__} did not return an EvaluationRow instance. You must return an EvaluationRow instance from your test function decorated with @evaluation_test."
                        )
                    if not isinstance(results, list):
                        raise ValueError(
                            f"Test function {test_func.__name__} did not return a list of EvaluationRow instances. You must return a list of EvaluationRow instances from your test function decorated with @evaluation_test."
                        )
                    if not results:
                        raise ValueError(
                            f"Test function {test_func.__name__} returned an empty list. You must return a non-empty list of EvaluationRow instances from your test function decorated with @evaluation_test."
                        )
                    if not all(isinstance(r, EvaluationRow) for r in results):
                        raise ValueError(
                            f"Test function {test_func.__name__} returned a list containing non-EvaluationRow instances. You must return a list of EvaluationRow instances from your test function decorated with @evaluation_test."
                        )
                    all_results.extend(results)

                scores = [r.evaluation_result.score for r in all_results if r.evaluation_result]
                agg_score = _aggregate(scores, aggregation_method)
                if threshold_of_success is not None:
                    assert (
                        agg_score >= threshold_of_success
                    ), f"Aggregated score {agg_score:.3f} below threshold {threshold_of_success}"

            # Create a function with the exact signature pytest expects without using exec
            from functools import wraps

            @wraps(test_func)
            def wrapper(**kwargs):
                return wrapper_body(**kwargs)

            parameters = [
                inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in test_param_names
            ]
            wrapper.__signature__ = inspect.Signature(parameters)

            return wrapper

        wrapper = create_wrapper_with_signature()
        wrapper = pytest.mark.parametrize(test_param_names, param_tuples)(wrapper)

        return wrapper

    return decorator
