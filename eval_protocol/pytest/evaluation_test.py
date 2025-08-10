import copy
import inspect
import os
import statistics
from typing import Any, Callable, Dict, List, Optional, Union

import pytest

from eval_protocol.dataset_logger import default_logger
from eval_protocol.models import CompletionParams, EvalMetadata, EvaluationRow, EvaluationThreshold, InputMetadata
from eval_protocol.pytest.default_dataset_adapter import default_dataset_adapter
from eval_protocol.pytest.default_no_op_rollout_process import default_no_op_rollout_processor
from eval_protocol.pytest.types import (
    Dataset,
    DatasetPathParam,
    EvaluationInputParam,
    EvaluationTestMode,
    InputMessagesParam,
    ModelParam,
    RolloutInputParam,
    RolloutProcessor,
    RolloutProcessorConfig,
    TestFunction,
)
from eval_protocol.pytest.utils import (
    AggregationMethod,
    aggregate,
    create_dynamically_parameterized_wrapper,
    execute_function,
)

from ..common_utils import load_jsonl


def evaluation_test(
    *,
    model: List[ModelParam],
    input_messages: Optional[List[InputMessagesParam]] = None,
    input_dataset: Optional[List[DatasetPathParam]] = None,
    dataset_adapter: Callable[[List[Dict[str, Any]]], Dataset] = default_dataset_adapter,
    rollout_input_params: Optional[List[RolloutInputParam]] = None,
    rollout_processor: RolloutProcessor = default_no_op_rollout_processor,
    evaluation_test_kwargs: Optional[List[EvaluationInputParam]] = None,
    aggregation_method: AggregationMethod = "mean",
    threshold: Optional[EvaluationThreshold] = None,
    num_runs: int = 1,
    max_dataset_rows: Optional[int] = None,
    mcp_config_path: Optional[str] = None,
    max_concurrent_rollouts: int = 8,
    server_script_path: Optional[str] = None,
    steps: int = 30,
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
        rollout_input_params: Generation parameters for the rollout.
        rollout_processor: Function used to perform the rollout.
        evaluation_test_kwargs: Kwargs for the evaluation function.
        aggregation_method: How to aggregate scores across rows.
        threshold: Threshold configuration for test success.
            Success rate must be above success, and if set, standard deviation must be below standard_deviation.
        num_runs: Number of times to repeat the evaluation.
        max_dataset_rows: Limit dataset to the first N rows.
        mcp_config_path: Path to MCP config file that follows MCPMultiClientConfiguration schema
        max_concurrent_rollouts: Maximum number of concurrent rollouts to run in parallel.
        server_script_path: Path to the MCP server script to run (default: "examples/tau2_mcp/server.py").
        steps: Number of rollout steps to execute (default: 30).
        mode: Evaluation mode. "batch" (default) expects test function to handle
            full dataset. "pointwise" applies test function to each row. If your evaluation requires
            the full rollout of all rows to compute the score, use
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
            row: EvaluationRow | None = None,
            input_dataset: List[EvaluationRow] | None = None,
            evaluation_test_kwargs: Optional[EvaluationInputParam] = None,
        ):
            kwargs = {}
            if input_dataset is not None:
                kwargs["rows"] = input_dataset
            if row is not None:
                kwargs["row"] = row
            if evaluation_test_kwargs is not None:
                if "row" in evaluation_test_kwargs:
                    raise ValueError("'row' is a reserved parameter for the evaluation function")
                if "rows" in evaluation_test_kwargs:
                    raise ValueError("'rows' is a reserved parameter for the evaluation function")
                kwargs.update(evaluation_test_kwargs)
            return execute_function(test_func, **kwargs)

        # Calculate all possible combinations of parameters
        def generate_combinations():
            combinations = []

            # Handle optional parameters with defaults
            datasets: List[Optional[DatasetPathParam]] = input_dataset if input_dataset is not None else [None]  # type: ignore
            params: List[Optional[RolloutInputParam]] = rollout_input_params if rollout_input_params is not None else [None]  # type: ignore
            messages: List[Optional[InputMessagesParam]] = input_messages if input_messages is not None else [None]  # type: ignore
            kwargs: List[Optional[EvaluationInputParam]] = evaluation_test_kwargs if evaluation_test_kwargs is not None else [None]  # type: ignore

            # Generate all combinations
            for m in model:
                for ds in datasets:
                    for ip in params:
                        for im in messages:
                            for etk in kwargs:
                                # if no dataset and no messages, raise an error
                                if ds is None and im is None:
                                    raise ValueError(
                                        "No dataset or messages provided. Please provide at least one of input_dataset or input_messages."
                                    )
                                combinations.append((m, ds, ip, im, etk))

            return combinations

        combinations = generate_combinations()
        if len(combinations) == 0:
            raise ValueError(
                "No combinations of parameters were found. Please provide at least a model and one of input_dataset or input_messages."
            )

        # Create parameter tuples for pytest.mark.parametrize
        param_tuples = []
        for combo in combinations:
            model_name, dataset, params, messages, etk = combo
            param_tuple = [model_name]
            if input_dataset is not None:
                param_tuple.append(dataset)
            if rollout_input_params is not None:
                param_tuple.append(params)
            if input_messages is not None:
                param_tuple.append(messages)
            if evaluation_test_kwargs is not None:
                param_tuple.append(etk)
            param_tuples.append(tuple(param_tuple))

        # For batch mode, use the original parameter names
        test_param_names = ["model"]
        if input_dataset is not None:
            test_param_names.append("dataset_path")
        if rollout_input_params is not None:
            test_param_names.append("input_params")
        if input_messages is not None:
            test_param_names.append("input_messages")
        if evaluation_test_kwargs is not None:
            test_param_names.append("evaluation_test_kwargs")

        # Create wrapper function with exact signature that pytest expects
        def create_wrapper_with_signature() -> Callable:
            # Create the function body that will be used
            def wrapper_body(**kwargs):
                model_name = kwargs["model"]
                eval_metadata = None
                all_results: List[List[EvaluationRow]] = [[] for _ in range(num_runs)]

                try:
                    # Handle dataset loading
                    data: List[EvaluationRow] = []
                    if "dataset_path" in kwargs and kwargs["dataset_path"] is not None:
                        data_jsonl = load_jsonl(kwargs["dataset_path"])
                        if max_dataset_rows is not None:
                            data_jsonl = data_jsonl[:max_dataset_rows]
                        data = dataset_adapter(data_jsonl)
                    elif "input_messages" in kwargs and kwargs["input_messages"] is not None:
                        data: List[EvaluationRow] = [EvaluationRow(messages=kwargs["input_messages"])]
                    else:
                        raise ValueError("No input dataset or input messages provided")

                    input_params = kwargs.get("input_params") or {}

                    # Create eval metadata with test function info and current commit hash
                    eval_metadata = EvalMetadata(
                        name=test_func.__name__,
                        description=test_func.__doc__,
                        status="running",
                        num_runs=num_runs,
                        aggregation_method=aggregation_method,
                        threshold=threshold,
                        passed=None,
                    )

                    # Populate completion_params in input_metadata for all rows and initialize eval_metadata BEFORE rollouts
                    completion_params = CompletionParams(
                        model=model_name,
                        temperature=input_params.get("temperature"),
                        max_tokens=input_params.get("max_tokens"),
                        max_tool_calls=input_params.get("max_tool_calls"),
                    )

                    for row in data:
                        if row.input_metadata is None:
                            row.input_metadata = InputMetadata()
                        row.input_metadata.completion_params = completion_params
                        # Add mode to session_data
                        if row.input_metadata.session_data is None:
                            row.input_metadata.session_data = {}
                        row.input_metadata.session_data["mode"] = mode
                        # Initialize eval_metadata for each row
                        row.eval_metadata = eval_metadata

                        # has to be done in the pytest main process since it's
                        # used to determine whether this eval has stopped
                        row.pid = os.getpid()
                        default_logger.log(row)

                    # Now run the rollout processor with metadata-initialized data
                    config = RolloutProcessorConfig(
                        model=model_name,
                        input_params=input_params,
                        mcp_config_path=mcp_config_path or "",
                        max_concurrent_rollouts=max_concurrent_rollouts,
                        server_script_path=server_script_path,
                        steps=steps,
                    )

                    for i in range(num_runs):
                        # Regenerate outputs each run by deep-copying the pristine dataset
                        # so model responses are not reused across runs.
                        fresh_rows = [copy.deepcopy(r) for r in data]
                        input_dataset = execute_function(rollout_processor, rows=fresh_rows, config=config)

                        if mode == "pointwise":
                            # Pointwise mode: apply the evaluator function to each row
                            for row in input_dataset:
                                result = execute_with_params(
                                    test_func,
                                    row=row,
                                    evaluation_test_kwargs=kwargs.get("evaluation_test_kwargs") or {},
                                )
                                if result is None or not isinstance(result, EvaluationRow):
                                    raise ValueError(
                                        f"Test function {test_func.__name__} did not return an EvaluationRow instance. You must return an EvaluationRow instance from your test function decorated with @evaluation_test."
                                    )
                                all_results[i].append(result)
                        else:
                            # Batch mode: call the test function with the full dataset
                            results = execute_with_params(
                                test_func,
                                input_dataset=input_dataset,
                                evaluation_test_kwargs=kwargs.get("evaluation_test_kwargs") or {},
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
                            all_results[i] = results

                    scores = [
                        sum([r.evaluation_result.score for r in result if r.evaluation_result]) / len(result)
                        for result in all_results
                    ]
                    agg_score = aggregate(scores, aggregation_method)
                    score_std = statistics.stdev(scores) if len(scores) > 1 else 0.0

                    # Determine if the evaluation passed based on threshold
                    passed = None

                    if threshold is not None:
                        success_passed, std_passed = True, True

                        success_passed = agg_score >= threshold["success"]

                        if threshold["standard_deviation"] is not None:
                            std_passed = score_std <= threshold["standard_deviation"]

                        passed = success_passed and std_passed

                    # Update eval metadata status and passed field for all results
                    for result in all_results:
                        for r in result:
                            if r.eval_metadata is not None:
                                r.eval_metadata.status = "finished"
                                r.eval_metadata.passed = passed
                        default_logger.log(r)

                    # Check threshold after logging
                    if threshold is not None and not passed:
                        assert (
                            agg_score >= threshold["success"]
                        ), f"Aggregated score {agg_score:.3f} below threshold {threshold["success"]}"
                        if threshold["standard_deviation"] is not None:
                            assert (
                                score_std <= threshold["standard_deviation"]
                            ), f"Standard deviation {score_std:.3f} above threshold {threshold["standard_deviation"]}"

                except Exception as e:
                    # Update eval metadata status to error and log it
                    if eval_metadata is not None:
                        eval_metadata.status = "error"
                        eval_metadata.passed = False

                        # Create a minimal result row to log the error if we don't have any results yet
                        if not data:
                            error_row = EvaluationRow(messages=[], eval_metadata=eval_metadata, evaluation_result=None)
                            default_logger.log(error_row)
                        else:
                            # Update existing results with error status
                            for r in data:
                                if r.eval_metadata is not None:
                                    r.eval_metadata.status = "error"
                                    r.eval_metadata.passed = False
                                default_logger.log(r)

                    # Re-raise the exception to maintain pytest behavior
                    raise

            return create_dynamically_parameterized_wrapper(test_func, wrapper_body, test_param_names)

        # Create the pytest wrapper
        pytest_wrapper = create_wrapper_with_signature()
        pytest_wrapper = pytest.mark.parametrize(test_param_names, param_tuples)(pytest_wrapper)

        def create_dual_mode_wrapper() -> Callable:
            """
            Creates a wrapper that supports both pytest parameterized execution and direct function calls.

            This wrapper enables the decorated evaluation test function to be used in two ways:
            1. As a pytest test (via pytest.mark.parametrize) with full parameterization
            2. As a direct function call with EvaluationRow data for programmatic use

            The wrapper automatically detects the calling pattern and routes to the appropriate
            execution path, ensuring consistent behavior regardless of how the function is invoked.

            Returns:
                A callable that can handle both pytest test execution and direct function calls
            """
            import asyncio

            # Check if the test function is async
            is_async = asyncio.iscoroutinefunction(test_func)

            if is_async:

                async def dual_mode_wrapper(*args, **kwargs):
                    # Check if this is a direct call with the expected signature
                    if mode == "pointwise":
                        # For pointwise mode, check if called with a single row argument
                        if len(args) == 1 and isinstance(args[0], EvaluationRow) and not kwargs:
                            return await test_func(row=args[0])
                    else:
                        # For batch mode, check if called with rows argument
                        if (
                            len(args) == 1
                            and isinstance(args[0], list)
                            and all(isinstance(r, EvaluationRow) for r in args[0])
                            and not kwargs
                        ):
                            return await test_func(rows=args[0])
                        # Also check if called with keyword argument 'rows'
                        if (
                            len(args) == 0
                            and "rows" in kwargs
                            and isinstance(kwargs["rows"], list)
                            and all(isinstance(r, EvaluationRow) for r in kwargs["rows"])
                        ):
                            return await test_func(**kwargs)

                    # If not a direct call, use the pytest wrapper
                    return pytest_wrapper(*args, **kwargs)

            else:

                def dual_mode_wrapper(*args, **kwargs):
                    # Check if this is a direct call with the expected signature
                    if mode == "pointwise":
                        # For pointwise mode, check if called with a single row argument
                        if len(args) == 1 and isinstance(args[0], EvaluationRow) and not kwargs:
                            return test_func(row=args[0])

                        if len(args) == 0 and "row" in kwargs and isinstance(kwargs["row"], EvaluationRow):
                            return test_func(**kwargs)
                    else:
                        # For batch mode, check if called with rows argument
                        if (
                            len(args) == 1
                            and isinstance(args[0], list)
                            and all(isinstance(r, EvaluationRow) for r in args[0])
                            and not kwargs
                        ):
                            return test_func(rows=args[0])
                        # Also check if called with keyword argument 'rows'
                        if (
                            len(args) == 0
                            and "rows" in kwargs
                            and isinstance(kwargs["rows"], list)
                            and all(isinstance(r, EvaluationRow) for r in kwargs["rows"])
                        ):
                            return test_func(**kwargs)

                    # If not a direct call, use the pytest wrapper
                    return pytest_wrapper(*args, **kwargs)

            # Copy all attributes from the pytest wrapper to our dual mode wrapper
            import functools

            functools.update_wrapper(dual_mode_wrapper, pytest_wrapper)

            return dual_mode_wrapper

        # Create the dual mode wrapper
        dual_mode_wrapper = create_dual_mode_wrapper()

        return dual_mode_wrapper

    return decorator
