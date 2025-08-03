import inspect
from typing import Any, Callable, Dict, List, Optional

import pytest

from eval_protocol.models import EvaluationRow
from eval_protocol.pytest.default_no_op_rollout_process import default_no_op_rollout_processor
from eval_protocol.pytest.types import (
    Dataset,
    DatasetPathParam,
    EvaluationTestMode,
    InputMessagesParam,
    InputParam,
    ModelParam,
    RolloutProcessor,
    RolloutProcessorConfig,
    TestFunction,
)
from eval_protocol.pytest.utils import aggregate, create_dynamically_parameterized_wrapper, execute_function

from ..common_utils import load_jsonl


def evaluation_test(
    *,
    model: List[ModelParam],
    input_messages: Optional[List[InputMessagesParam]] = None,
    input_dataset: Optional[List[DatasetPathParam]] = None,
    dataset_adapter: Optional[Callable[[List[Dict[str, Any]]], Dataset]] = lambda x: x,
    input_params: Optional[List[InputParam]] = None,
    rollout_processor: RolloutProcessor = default_no_op_rollout_processor,
    aggregation_method: str = "mean",
    threshold_of_success: Optional[float] = None,
    num_runs: int = 1,
    max_dataset_rows: Optional[int] = None,
    mcp_config_path: Optional[str] = None,
    mode: EvaluationTestMode = "batch",
) -> Callable[
    [TestFunction],
    TestFunction,
]:
    """Decorator to create pytest-based evaluation tests.

    Args:
        model: Model identifiers to query.
        input_messages: Messages to send to the model. This is useful if you
            don't have a dataset but can hard-code the messages. Will be passed as
            "input_dataset" to the test function.
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
        mode: Evaluation mode. "batch" (default) expects test function to handle
            full dataset. "pointwise" applies test function to each row. If your evaluation requires
            the full rollout of all rows to compute the score, use

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
        sig = inspect.signature(test_func)

        # For pointwise/rowwise mode, we expect a different signature
        if mode == "pointwise":
            # Pointwise mode: function should accept messages and other row-level params
            if "row" not in sig.parameters:
                raise ValueError(f"In pointwise mode, your eval function must have a parameter named 'row'")

            # validate that "Row" is of type EvaluationRow
            if sig.parameters["row"].annotation is not EvaluationRow:
                raise ValueError(f"In pointwise mode, the 'row' parameter must be of type EvaluationRow")

            # validate that the function has a return type of EvaluationRow
            if sig.return_annotation is not EvaluationRow:
                raise ValueError("In pointwise mode, your eval function must return an EvaluationRow instance")
        else:
            # Batch mode: function should accept input_dataset and model
            if "rows" not in sig.parameters:
                raise ValueError("In batch mode, your eval function must have a parameter named 'rows'")

            # validate that "Rows" is of type List[EvaluationRow]
            if sig.parameters["rows"].annotation is not List[EvaluationRow]:
                raise ValueError(f"In batch mode, the 'rows' parameter must be of type List[EvaluationRow]")

            # validate that the function has a return type of List[EvaluationRow]
            if sig.return_annotation is not List[EvaluationRow]:
                raise ValueError("In batch mode, your eval function must return a list of EvaluationRow instances")

        def execute_with_params(
            test_func: TestFunction,
            model: str,
            row: EvaluationRow | None = None,
            input_dataset: List[EvaluationRow] | None = None,
            input_params: InputParam | None = None,
        ):
            kwargs = {}
            if input_dataset is not None:
                kwargs["rows"] = input_dataset
            if input_params is not None:
                kwargs["input_params"] = input_params
            if model is not None:
                kwargs["model"] = model
            if row is not None:
                kwargs["row"] = row
            return execute_function(test_func, **kwargs)

        # Calculate all possible combinations of parameters
        def generate_combinations():
            combinations = []

            # Handle optional parameters with defaults
            datasets: List[Optional[DatasetPathParam]] = input_dataset if input_dataset is not None else [None]  # type: ignore
            params: List[Optional[InputParam]] = input_params if input_params is not None else [None]  # type: ignore
            messages: List[Optional[InputMessagesParam]] = input_messages if input_messages is not None else [None]  # type: ignore

            # Generate all combinations
            for m in model:
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

        combinations = generate_combinations()

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

        # For batch mode, use the original parameter names
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

                # Handle dataset loading
                if "dataset_path" in kwargs and kwargs["dataset_path"] is not None:
                    data = load_jsonl(kwargs["dataset_path"])
                    if max_dataset_rows is not None:
                        data = data[:max_dataset_rows]
                    data = dataset_adapter(data)
                elif "input_messages" in kwargs and kwargs["input_messages"] is not None:
                    data: List[EvaluationRow] = [EvaluationRow(messages=kwargs["input_messages"])]
                else:
                    raise ValueError("No input dataset or input messages provided")

                input_dataset: List[EvaluationRow] = []
                config = RolloutProcessorConfig(
                    model=model_name,
                    input_params=kwargs.get("input_params") or {},
                    mcp_config_path=mcp_config_path or "",
                    initial_messages=kwargs.get("input_messages") if "input_messages" in kwargs else [],
                )
                for row in data:
                    processed: List[EvaluationRow] = execute_function(rollout_processor, row=row, config=config)
                    input_dataset.extend(processed)

                all_results: List[EvaluationRow] = []
                for _ in range(num_runs):
                    if mode == "pointwise":
                        # Pointwise mode: apply the evaluator function to each row
                        for row in input_dataset:
                            result = execute_with_params(
                                test_func,
                                model=model_name,
                                row=row,
                                input_params=kwargs.get("input_params") if "input_params" in kwargs else None,
                            )
                            if result is None or not isinstance(result, EvaluationRow):
                                raise ValueError(
                                    f"Test function {test_func.__name__} did not return an EvaluationRow instance. You must return an EvaluationRow instance from your test function decorated with @evaluation_test."
                                )
                            all_results.append(result)
                    else:
                        # Batch mode: call the test function with the full dataset
                        results = execute_with_params(
                            test_func,
                            model=model_name,
                            input_dataset=input_dataset,
                            input_params=kwargs.get("input_params") if "input_params" in kwargs else None,
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
                agg_score = aggregate(scores, aggregation_method)
                if threshold_of_success is not None:
                    assert (
                        agg_score >= threshold_of_success
                    ), f"Aggregated score {agg_score:.3f} below threshold {threshold_of_success}"

            return create_dynamically_parameterized_wrapper(test_func, wrapper_body, test_param_names)

        wrapper = create_wrapper_with_signature()
        wrapper = pytest.mark.parametrize(test_param_names, param_tuples)(wrapper)

        return wrapper

    return decorator
