import asyncio
import copy
import inspect
import json
import math
import os
import pathlib
import re
import statistics
import time
from dataclasses import replace
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import pytest

from eval_protocol.dataset_logger import default_logger
from eval_protocol.dataset_logger.dataset_logger import DatasetLogger
from eval_protocol.human_id import generate_id
from eval_protocol.models import (
    CompletionParams,
    EvalMetadata,
    EvaluationRow,
    EvaluationThreshold,
    InputMetadata,
    Message,
)
from eval_protocol.pytest.default_dataset_adapter import default_dataset_adapter
from eval_protocol.pytest.default_no_op_rollout_process import default_no_op_rollout_processor
from eval_protocol.pytest.types import (
    Dataset,
    DatasetPathParam,
    EvaluationInputParam,
    EvaluationTestMode,
    InputMessagesParam,
    ModelParam,
    RolloutProcessor,
    RolloutProcessorConfig,
    RolloutProcessorInputParam,
    TestFunction,
)
from eval_protocol.pytest.utils import (
    AggregationMethod,
    aggregate,
    create_dynamically_parameterized_wrapper,
    execute_function,
    log_eval_status_and_rows,
)
from eval_protocol.stats.confidence_intervals import compute_fixed_set_mu_ci

from ..common_utils import load_jsonl


def evaluation_test(  # noqa: C901
    *,
    completion_params: List[CompletionParams],
    input_messages: Optional[List[InputMessagesParam]] = None,
    input_dataset: Optional[List[DatasetPathParam]] = None,
    dataset_adapter: Callable[[List[Dict[str, Any]]], Dataset] = default_dataset_adapter,
    rollout_processor: RolloutProcessor = default_no_op_rollout_processor,
    evaluation_test_kwargs: Optional[List[EvaluationInputParam]] = None,
    rollout_processor_kwargs: Optional[RolloutProcessorInputParam] = None,
    aggregation_method: AggregationMethod = "mean",
    passed_threshold: Optional[Union[EvaluationThreshold, float, dict]] = None,
    num_runs: int = 1,
    max_dataset_rows: Optional[int] = None,
    mcp_config_path: Optional[str] = None,
    max_concurrent_rollouts: int = 8,
    server_script_path: Optional[str] = None,
    steps: int = 30,
    mode: EvaluationTestMode = "batch",
    combine_datasets: bool = True,
    logger: Optional[DatasetLogger] = None,
) -> Callable[
    [TestFunction],
    TestFunction,
]:
    """Decorator to create pytest-based evaluation tests.

    Here are some key concepts to understand the terminology in EP:

    - "invocation" is a single execution of a test function. An invocation can
        generate 1 or more experiments. Grouping by invocation might be useful to
        aggregate eval scores across multiple invocations when you want to aggregate
        scores across multiple datasets.
    - "experiment" is a group of runs with for a combination of parameters. A single
        experiment will have multiple runs if num_runs > 1.
        1. If your evaluation_test has combinations of parameters, it will generate
        multiple experiments per combination of parameters.
        2. A new execution of a test function will generate a new experiment.
    - "run" is a group of rollouts. For multiple num_runs > 1, there will be
        multiple "run_id"s.
    - "rollout" is the execution/process that produces a "trajectory". You
        "execute" multiple rollouts to generate a dataset of trajectories.
    - "trajectory" is the result produced by a rollout — a list of OpenAI Chat
        Completion messages (e.g. the "messages" field in EvaluationRow).
    - "row" both the input and output of an evaluation. For example, in
        tau-bench, a row is a task within the dataset that can be identified as
        "airline_task_0" or "airline_task_1" etc. The "row_id" can be populated from
        the dataset itself to identify a particular task you want to evaluate.  If
        not provided, EP will generate a "row_id" for each row whenever you call the
        evaluation test.
    - "dataset" is a collection of rows (e.g. List[EvauluationRow])
    - "eval" is a rubric implemented in the body of an @evaluation_test
        decorated test. It simply produces a score from 0 to 1 and attached it
        to the row as the "evaluation_result" field.

    "invocation", "experiment", "run", "rollout", and "row" each have a unique ID
    which can be used to easily group and identify your dataset by.

    Args:
        input_messages: Messages to send to the model. This is useful if you
            don't have a dataset but can hard-code the messages. Will be passed as
            "input_dataset" to the test function.
        input_dataset: Paths to JSONL datasets. This is useful if you have a
            dataset already. Provide a dataset_adapter to convert the input dataset
            to a list of EvaluationRows if you have a custom dataset format.
        dataset_adapter: Function to convert the input dataset to a list of
            EvaluationRows. This is useful if you have a custom dataset format.
        completion_params: Generation parameters for the rollout.
        rollout_processor: Function used to perform the rollout.
        evaluation_test_kwargs: Kwargs for the evaluation function.
        rollout_processor_kwargs: Kwargs for the rollout processor.
        aggregation_method: How to aggregate scores across rows.
        passed_threshold: Threshold configuration for test success. Must be a float or EvaluationThreshold object.
            Success rate must be above success, and if set, standard deviation must be below standard_deviation.
        num_runs: Number of times to repeat the rollout and evaluations.
        max_dataset_rows: Limit dataset to the first N rows.
        mcp_config_path: Path to MCP config file that follows MCPMultiClientConfiguration schema
        max_concurrent_rollouts: Maximum number of concurrent rollouts to run in parallel.
        server_script_path: Path to the MCP server script to run (default: "examples/tau2_mcp/server.py").
        steps: Number of rollout steps to execute (default: 30).
        mode: Evaluation mode. "batch" (default) expects test function to handle
            full dataset. "pointwise" applies test function to each row. If your evaluation requires
            the full rollout of all rows to compute the score, use
        logger: DatasetLogger to use for logging. If not provided, a default logger will be used.
    """

    active_logger: DatasetLogger = logger if logger else default_logger

    def decorator(
        test_func: TestFunction,
    ):
        if passed_threshold is not None:
            if isinstance(passed_threshold, float):
                threshold = EvaluationThreshold(success=passed_threshold)
            else:
                threshold = EvaluationThreshold(**passed_threshold)
        else:
            threshold = None

        sig = inspect.signature(test_func)

        # For pointwise/rowwise mode, we expect a different signature
        if mode == "pointwise":
            # Pointwise mode: function should accept messages and other row-level params
            if "row" not in sig.parameters:
                raise ValueError("In pointwise mode, your eval function must have a parameter named 'row'")

            # validate that "Row" is of type EvaluationRow
            if sig.parameters["row"].annotation is not EvaluationRow:
                raise ValueError("In pointwise mode, the 'row' parameter must be of type EvaluationRow")

            # validate that the function has a return type of EvaluationRow
            if sig.return_annotation is not EvaluationRow:
                raise ValueError("In pointwise mode, your eval function must return an EvaluationRow instance")
        else:
            # Batch mode: function should accept input_dataset and model
            if "rows" not in sig.parameters:
                raise ValueError("In batch mode, your eval function must have a parameter named 'rows'")

            # validate that "Rows" is of type List[EvaluationRow]
            if sig.parameters["rows"].annotation is not List[EvaluationRow]:
                raise ValueError("In batch mode, the 'rows' parameter must be of type List[EvaluationRow")

            # validate that the function has a return type of List[EvaluationRow]
            if sig.return_annotation is not List[EvaluationRow]:
                raise ValueError("In batch mode, your eval function must return a list of EvaluationRow instances")

        async def execute_with_params(
            test_func: TestFunction,
            processed_row: EvaluationRow | None = None,
            processed_dataset: List[EvaluationRow] | None = None,
            evaluation_test_kwargs: Optional[EvaluationInputParam] = None,
        ):
            kwargs = {}
            if processed_dataset is not None:
                kwargs["rows"] = processed_dataset
            if processed_row is not None:
                kwargs["row"] = processed_row
            if evaluation_test_kwargs is not None:
                if "row" in evaluation_test_kwargs:
                    raise ValueError("'row' is a reserved parameter for the evaluation function")
                if "rows" in evaluation_test_kwargs:
                    raise ValueError("'rows' is a reserved parameter for the evaluation function")
                kwargs.update(evaluation_test_kwargs)

            # Handle both sync and async test functions
            if asyncio.iscoroutinefunction(test_func):
                return await test_func(**kwargs)
            else:
                return test_func(**kwargs)

        # Calculate all possible combinations of parameters
        def _parse_ep_max_rows(default_value: int | None) -> int | None:
            """Read EP_MAX_DATASET_ROWS env override as int or None."""
            raw = os.getenv("EP_MAX_DATASET_ROWS")
            if raw is None:
                return default_value
            s = raw.strip().lower()
            if s == "none":
                return None
            try:
                return int(s)
            except ValueError:
                return default_value

        def _deep_update_dict(base: dict, override: dict) -> dict:
            """Recursively update nested dictionaries in-place and return base."""
            for key, value in override.items():
                if isinstance(value, dict) and isinstance(base.get(key), dict):
                    _deep_update_dict(base[key], value)
                else:
                    base[key] = value
            return base

        def generate_combinations():
            combinations = []

            # Handle optional parameters with defaults
            # Optionally combine multiple dataset paths into one logical dataset,
            # or parameterize to run one dataset per test invocation.
            if input_dataset is not None:
                if combine_datasets:
                    datasets: List[Optional[List[DatasetPathParam]]] = [input_dataset]  # type: ignore
                else:
                    # Fan out: one dataset path per parameterization
                    if isinstance(input_dataset, list):  # type: ignore
                        datasets = [[p] for p in input_dataset]  # type: ignore
                    else:
                        datasets = [[input_dataset]]  # type: ignore
            else:
                datasets = [None]
            cps: List[Optional[CompletionParams]] = completion_params if completion_params is not None else [None]  # type: ignore
            # Apply EP_MAX_DATASET_ROWS to input_messages, but do NOT parameterize over
            # each row. Instead, pass the entire sliced list through in a single test run
            # so summaries aggregate all rows together (AIME-style behavior).
            if input_messages is not None and isinstance(input_messages, list):
                effective_max_rows = _parse_ep_max_rows(max_dataset_rows)
                if effective_max_rows is not None:
                    sliced_messages = input_messages[:effective_max_rows]  # type: ignore
                else:
                    sliced_messages = input_messages  # type: ignore
                # Wrap as a single parameter payload
                messages = [sliced_messages]  # type: ignore
            else:
                messages = [None]  # type: ignore
            kwargs: List[Optional[EvaluationInputParam]] = evaluation_test_kwargs if evaluation_test_kwargs is not None else [None]  # type: ignore

            # Generate all combinations
            for ds in datasets:
                for cp in cps:
                    for im in messages:
                        for etk in kwargs:
                            # if no dataset and no messages, raise an error
                            if ds is None and im is None:
                                raise ValueError(
                                    "No dataset or messages provided. Please provide at least one of input_dataset or input_messages."
                                )
                            combinations.append((ds, cp, im, etk))

            return combinations

        async def rollout_processor_with_retry(
            rollout_processor: RolloutProcessor,
            fresh_dataset: List[EvaluationRow],
            config: RolloutProcessorConfig,
            max_retry: int,
        ):
            """
            Wrapper around rollout_processor that handles retry logic internally.
            Uses async queue pattern to yield results immediately as they become available.
            Yields both successful and failed results, leaving it up to the user to handle them in test_func.
            """

            try:
                queue = asyncio.Queue()
                retry_counts = {r.execution_metadata.rollout_id: 0 for r in fresh_dataset}
                failed_permanently = []

                async def retry_handler(failed_row: EvaluationRow):
                    rollout_id = failed_row.execution_metadata.rollout_id
                    current_attempts = retry_counts.get(rollout_id, 0)

                    if current_attempts >= max_retry:
                        assert (
                            failed_row.rollout_status and failed_row.rollout_status.status == "error"
                        ), f"Rollout {failed_row.execution_metadata.rollout_id} did not fail with error status"
                        failed_permanently.append(failed_row)
                        await queue.put(failed_row)  # put failed row on queue
                        return

                    retry_counts[rollout_id] = current_attempts + 1

                    # add kwargs start_server=False to config so we don't start new MCP server
                    retry_config = replace(config, kwargs={**(config.kwargs or {}), "start_server": False})

                    retry_tasks = rollout_processor([failed_row], retry_config)

                    try:
                        retry_result = await retry_tasks[0]
                        retry_result.rollout_status.status = "finished"
                        await queue.put(retry_result)
                    except Exception as e:
                        failed_row.rollout_status.status = "error"
                        failed_row.rollout_status.termination_reason = str(e)
                        asyncio.create_task(retry_handler(failed_row))  # retry failed, spawn another retry

                async def initial_processor():
                    """Process initial batch and spawn retries for failures"""
                    base_tasks = rollout_processor(fresh_dataset, config)
                    pending = set(base_tasks)

                    while pending:
                        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

                        for task in done:
                            task_index = base_tasks.index(task)

                            try:
                                result = await task
                                result.rollout_status.status = "finished"
                                await queue.put(result)
                            except Exception as e:
                                failed_row = fresh_dataset[task_index]
                                failed_row.rollout_status.status = "error"
                                failed_row.rollout_status.termination_reason = str(e)
                                asyncio.create_task(retry_handler(failed_row))  # rollout errored, spawn retry task

                processor_task = asyncio.create_task(initial_processor())

                # yield results as they become available
                completed_count = 0
                total_expected = len(fresh_dataset)

                while completed_count < total_expected:
                    finished_row = await queue.get()

                    # only permanent failure rows are put on the queue, so we can check for them here
                    if finished_row.rollout_status and finished_row.rollout_status.status == "error":
                        if os.getenv("EP_FAIL_ON_PERMANENT_FAILURE", "true") != "false":
                            raise RuntimeError(
                                f"Rollout {finished_row.execution_metadata.rollout_id} failed after {max_retry} retries. Errors: {finished_row.rollout_status.termination_reason}"
                            )

                    completed_count += 1
                    yield finished_row

                await processor_task  # explicitly wait for task completion and catch any exceptions

            finally:
                # processor clean up after themselves if they have a cleanup method
                if hasattr(rollout_processor, "cleanup"):
                    rollout_processor.cleanup()

        combinations = generate_combinations()
        if len(combinations) == 0:
            raise ValueError(
                "No combinations of parameters were found. Please provide at least a model and one of input_dataset or input_messages."
            )

        # Create parameter tuples for pytest.mark.parametrize
        param_tuples = []
        for combo in combinations:
            dataset, cp, messages, etk = combo
            param_tuple = []
            if input_dataset is not None:
                param_tuple.append(dataset)
            if completion_params is not None:
                param_tuple.append(cp)
            if input_messages is not None:
                param_tuple.append(messages)
            if evaluation_test_kwargs is not None:
                param_tuple.append(etk)
            param_tuples.append(tuple(param_tuple))

        # For batch mode, use the original parameter names
        test_param_names = []
        if input_dataset is not None:
            test_param_names.append("dataset_path")
        if completion_params is not None:
            test_param_names.append("completion_params")
        if input_messages is not None:
            test_param_names.append("input_messages")
        if evaluation_test_kwargs is not None:
            test_param_names.append("evaluation_test_kwargs")

        # Create wrapper function with exact signature that pytest expects
        def create_wrapper_with_signature() -> Callable:
            # Create the function body that will be used
            invocation_id = generate_id()

            async def wrapper_body(**kwargs):
                eval_metadata = None
                all_results: List[List[EvaluationRow]] = [[] for _ in range(num_runs)]

                experiment_id = generate_id()

                def _log_eval_error(
                    status: Literal["finished", "error"], rows: Optional[List[EvaluationRow]] | None, passed: bool
                ) -> None:
                    log_eval_status_and_rows(eval_metadata, rows, status, passed, active_logger)

                try:
                    # Handle dataset loading
                    data: List[EvaluationRow] = []
                    if "dataset_path" in kwargs and kwargs["dataset_path"] is not None:
                        ds_arg = kwargs["dataset_path"]
                        # Support either a single path or a list of paths; if a list is provided,
                        # concatenate the rows from each file in order.
                        if isinstance(ds_arg, list):
                            data_jsonl = []
                            for p in ds_arg:
                                data_jsonl.extend(load_jsonl(p))
                        else:
                            data_jsonl = load_jsonl(ds_arg)
                        # Apply env override for max rows if present
                        effective_max_rows = _parse_ep_max_rows(max_dataset_rows)
                        if effective_max_rows is not None:
                            data_jsonl = data_jsonl[:effective_max_rows]
                        data = dataset_adapter(data_jsonl)
                    elif "input_messages" in kwargs and kwargs["input_messages"] is not None:
                        # Support either a single row (List[Message]) or many rows (List[List[Message]])
                        im = kwargs["input_messages"]
                        if isinstance(im, list) and len(im) > 0 and isinstance(im[0], Message):
                            # Single row of Message objects
                            data = [EvaluationRow(messages=im)]
                        else:
                            # Multiple rows: list of List[Message]
                            data = [EvaluationRow(messages=m) for m in im]
                    else:
                        raise ValueError("No input dataset or input messages provided")

                    if "completion_params" not in kwargs or not kwargs["completion_params"]:
                        raise ValueError(
                            "No completion parameters provided. Please provide a completion parameters object."
                        )
                    completion_params = kwargs["completion_params"]
                    if "model" not in completion_params or not completion_params["model"]:
                        raise ValueError(
                            "No model provided. Please provide a model in the completion parameters object."
                        )

                    # Optional global overrides via environment for ad-hoc experimentation
                    # EP_INPUT_PARAMS_JSON can contain a JSON object that will be deep-merged
                    # into input_params (e.g., '{"temperature":0,"extra_body":{"reasoning":{"effort":"low"}}}').
                    try:
                        import json as _json

                        _env_override = os.getenv("EP_INPUT_PARAMS_JSON")
                        if _env_override:
                            override_obj = _json.loads(_env_override)
                            if isinstance(override_obj, dict):
                                completion_params = _deep_update_dict(dict(completion_params), override_obj)
                    except Exception:
                        pass

                    # Create eval metadata with test function info and current commit hash
                    eval_metadata = EvalMetadata(
                        name=test_func.__name__,
                        description=test_func.__doc__,
                        status="running",
                        num_runs=num_runs,
                        aggregation_method=aggregation_method,
                        passed_threshold=threshold,
                        passed=None,
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
                        row.execution_metadata.experiment_id = experiment_id
                        row.execution_metadata.invocation_id = invocation_id

                        # has to be done in the pytest main process since it's
                        # used to determine whether this eval has stopped
                        row.pid = os.getpid()

                    # Prepare rollout processor config once; we will generate fresh outputs per run
                    config = RolloutProcessorConfig(
                        completion_params=completion_params,
                        mcp_config_path=mcp_config_path or "",
                        max_concurrent_rollouts=max_concurrent_rollouts,
                        server_script_path=server_script_path,
                        steps=steps,
                        logger=active_logger,
                        kwargs=rollout_processor_kwargs or {},
                    )

                    max_retry = int(os.getenv("EP_MAX_RETRY", "0"))

                    for i in range(num_runs):
                        # Regenerate outputs each run by deep-copying the pristine dataset
                        # so model responses are not reused across runs.
                        run_id = generate_id()
                        fresh_dataset = [r.model_copy(deep=True) for r in data]

                        # apply new run_id to fresh_dataset
                        for row in fresh_dataset:
                            row.execution_metadata.run_id = run_id

                        # generate new rollout_id for each row
                        for row in fresh_dataset:
                            row.execution_metadata.rollout_id = generate_id()

                        # log the fresh_dataset
                        for row in fresh_dataset:
                            active_logger.log(row)

                        if mode == "pointwise":
                            # Pointwise mode, rollouts will return as they complete so we can pipeline evaluation_test execution
                            semaphore = asyncio.Semaphore(max_concurrent_rollouts)
                            tasks = []

                            async def _execute_with_semaphore(row):
                                async with semaphore:
                                    # NOTE: we will still evaluate errored rows (give users control over this)
                                    # i.e., they can choose to give EvaluateResult.score = 0 for errored rows in their test_func
                                    result = await execute_with_params(
                                        test_func,
                                        processed_row=row,
                                        evaluation_test_kwargs=kwargs.get("evaluation_test_kwargs") or {},
                                    )
                                    if result is None or not isinstance(result, EvaluationRow):
                                        raise ValueError(
                                            f"Test function {test_func.__name__} did not return an EvaluationRow instance. You must return an EvaluationRow instance from your test function decorated with @evaluation_test."
                                        )
                                    return result

                            # Use wrapper that handles retry logic internally
                            async for row in rollout_processor_with_retry(
                                rollout_processor, fresh_dataset, config, max_retry
                            ):
                                tasks.append(asyncio.create_task(_execute_with_semaphore(row)))

                            all_results[i] = await asyncio.gather(*tasks)

                        else:
                            # Batch mode: collect all results first, then evaluate (no pipelining)
                            input_dataset = []
                            async for row in rollout_processor_with_retry(
                                rollout_processor, fresh_dataset, config, max_retry
                            ):
                                input_dataset.append(row)
                            # NOTE: we will still evaluate errored rows (give users control over this)
                            # i.e., they can choose to give EvaluateResult.score = 0 for errored rows in their test_func
                            results = await execute_with_params(
                                test_func,
                                processed_dataset=input_dataset,
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

                    # Compute 95% confidence interval for the fixed-set mean μ (by-question, using repeats)
                    ci_low: float | None = None
                    ci_high: float | None = None
                    if aggregation_method == "mean":
                        try:
                            result_ci = compute_fixed_set_mu_ci([item for sublist in all_results for item in sublist])
                            mu_ci_low, mu_ci_high = result_ci[1], result_ci[2]
                            if mu_ci_low is not None and mu_ci_high is not None:
                                ci_low = float(mu_ci_low)
                                ci_high = float(mu_ci_high)
                                # Keep agg_score as-is (mean over scores). For equal repeats per question these match.
                        except Exception:
                            ci_low = None
                            ci_high = None

                    # Determine if the evaluation passed based on threshold
                    passed = None

                    if threshold is not None:
                        success_passed, std_passed = True, True

                        success_passed = agg_score >= threshold.success

                        if threshold.standard_deviation is not None:
                            std_passed = score_std <= threshold.standard_deviation

                        passed = success_passed and std_passed

                    # Update eval metadata status and passed field for all results
                    for result in all_results:
                        for r in result:
                            if r.eval_metadata is not None:
                                r.eval_metadata.status = "finished"  # TODO: might not be needed
                                r.eval_metadata.passed = passed
                            active_logger.log(r)

                    # Optional: print and/or persist a summary artifact for CI
                    try:
                        should_print = os.getenv("EP_PRINT_SUMMARY") == "1"
                        summary_path = os.getenv("EP_SUMMARY_JSON")
                        suite_name = test_func.__name__
                        model_used = config.completion_params["model"]
                        total_rows = len([item for sublist in all_results for item in sublist])
                        summary_obj = {
                            "suite": suite_name,
                            "model": model_used,
                            "agg_score": float(agg_score) if agg_score is not None else None,
                            "num_runs": num_runs,
                            "rows": total_rows,
                        }
                        if ci_low is not None and ci_high is not None:
                            summary_obj["agg_ci_low"] = ci_low
                            summary_obj["agg_ci_high"] = ci_high

                        # Aggregate per-metric mean and 95% CI when available
                        metrics_summary: Dict[str, Dict[str, float]] = {}
                        from collections import defaultdict

                        metric_scores: Dict[str, list] = defaultdict(list)
                        for r in [item for sublist in all_results for item in sublist]:
                            if r.evaluation_result and r.evaluation_result.metrics:
                                for m_name, m_res in r.evaluation_result.metrics.items():
                                    if m_res is not None and getattr(m_res, "score", None) is not None:
                                        metric_scores[m_name].append(m_res.score)
                        for m_name, vals in metric_scores.items():
                            if len(vals) == 0:
                                continue
                            m_mean = sum(vals) / len(vals)
                            m_low = None
                            m_high = None
                            if len(vals) >= 2:
                                try:
                                    m_std = statistics.stdev(vals)
                                    m_se = m_std / math.sqrt(len(vals))
                                    m_margin = 1.96 * m_se
                                    m_low = max(0.0, m_mean - m_margin)
                                    m_high = min(1.0, m_mean + m_margin)
                                except Exception:
                                    m_low = None
                                    m_high = None
                            entry: Dict[str, float] = {"mean": float(m_mean)}
                            if m_low is not None and m_high is not None:
                                entry["ci_low"] = float(m_low)
                                entry["ci_high"] = float(m_high)
                            metrics_summary[m_name] = entry
                        if metrics_summary:
                            summary_obj["metrics_agg"] = metrics_summary
                        if should_print:
                            if ci_low is not None and ci_high is not None:
                                print(
                                    f"EP Summary | suite={suite_name} model={model_used} agg={summary_obj['agg_score']:.3f} ci95=[{ci_low:.3f},{ci_high:.3f}] runs={num_runs} rows={total_rows}"
                                )
                            else:
                                print(
                                    f"EP Summary | suite={suite_name} model={model_used} agg={summary_obj['agg_score']:.3f} runs={num_runs} rows={total_rows}"
                                )
                            # As per project convention, avoid printing per-metric CI lines to reduce noise
                        if summary_path:

                            def _sanitize_filename(text: str) -> str:
                                safe = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip())
                                return safe[:120]

                            def _extract_effort_tag(params: dict) -> str | None:
                                try:
                                    if not isinstance(params, dict):
                                        return None
                                    # Common locations
                                    if "extra_body" in params and isinstance(params["extra_body"], dict):
                                        eb = params["extra_body"]
                                        if isinstance(eb.get("reasoning"), dict) and "effort" in eb["reasoning"]:
                                            return str(eb["reasoning"]["effort"]).lower()
                                        if "reasoning_effort" in eb:
                                            return str(eb["reasoning_effort"]).lower()
                                    if (
                                        "reasoning" in params
                                        and isinstance(params["reasoning"], dict)
                                        and "effort" in params["reasoning"]
                                    ):
                                        return str(params["reasoning"]["effort"]).lower()
                                except Exception:
                                    return None
                                return None

                            model_slug = _sanitize_filename(model_used)
                            effort_tag = _extract_effort_tag(completion_params) or ""
                            effort_suffix = f"__effort-{_sanitize_filename(effort_tag)}" if effort_tag else ""
                            base_name = f"{suite_name}__{model_slug}{effort_suffix}__{mode}__runs{num_runs}.json"

                            p = pathlib.Path(summary_path)
                            summary_obj["timestamp"] = int(time.time())

                            # When a directory is provided (or a path without .json), write per-combination files inside it
                            if p.suffix.lower() != ".json" or summary_path.endswith("/") or p.is_dir():
                                out_dir = p
                                out_dir.mkdir(parents=True, exist_ok=True)
                                out_file = out_dir / base_name
                            else:
                                # A file path was provided
                                # If multiple parameterizations exist, write side-by-side files with suffixes based on base name
                                parent = p.parent
                                parent.mkdir(parents=True, exist_ok=True)
                                # If we detected an effort tag, fan out to separate files; otherwise write to the exact file
                                if effort_tag:
                                    out_file = parent / f"{p.stem}__{_sanitize_filename(effort_tag)}{p.suffix}"
                                else:
                                    out_file = p

                            with open(out_file, "w", encoding="utf-8") as f:
                                json.dump(summary_obj, f)
                    except Exception:
                        # Do not fail evaluation if summary writing fails
                        pass

                    # # Write all rows from active_logger.read() to a JSONL file in the same directory as the summary
                    # try:
                    #     if active_logger is not None:
                    #         rows = active_logger.read()
                    #         # Write to a .jsonl file alongside the summary file
                    #         jsonl_path = "logs.jsonl"
                    #         import json

                    #         with open(jsonl_path, "w", encoding="utf-8") as f_jsonl:
                    #             for row in rows:
                    #                 json.dump(row.model_dump(exclude_none=True, mode="json"), f_jsonl)
                    #                 f_jsonl.write("\n")
                    # except Exception as e:
                    #     # Do not fail evaluation if log writing fails
                    #     print(e)
                    #     pass

                    # Check threshold after logging
                    if threshold is not None and not passed:
                        assert (
                            agg_score >= threshold.success
                        ), f"Aggregated score {agg_score:.3f} below threshold {threshold.success}"
                        if threshold.standard_deviation is not None:
                            assert (
                                score_std <= threshold.standard_deviation
                            ), f"Standard deviation {score_std:.3f} above threshold {threshold.standard_deviation}"

                except AssertionError:
                    _log_eval_error("finished", data if "data" in locals() else None, passed=False)
                    raise
                except Exception:
                    _log_eval_error("error", data if "data" in locals() else None, passed=False)
                    raise

            return create_dynamically_parameterized_wrapper(test_func, wrapper_body, test_param_names)

        # Create the pytest wrapper
        pytest_wrapper = create_wrapper_with_signature()
        pytest_wrapper = pytest.mark.parametrize(test_param_names, param_tuples)(pytest_wrapper)
        pytest_wrapper = pytest.mark.asyncio(pytest_wrapper)

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

            async def call_test_func(**call_kwargs):
                """Helper to call test_func with proper async/sync handling"""
                if is_async:
                    return await test_func(**call_kwargs)
                else:
                    return test_func(**call_kwargs)

            async def dual_mode_wrapper(*args, **kwargs):
                # Check if this is a direct call with the expected signature
                if mode == "pointwise":
                    # For pointwise mode, check if called with a single row argument
                    if len(args) == 1 and isinstance(args[0], EvaluationRow) and not kwargs:
                        return await call_test_func(row=args[0])
                else:
                    # For batch mode, check if called with rows argument
                    if (
                        len(args) == 1
                        and isinstance(args[0], list)
                        and all(isinstance(r, EvaluationRow) for r in args[0])
                        and not kwargs
                    ):
                        return await call_test_func(rows=args[0])
                    # Also check if called with keyword argument 'rows'
                    if (
                        len(args) == 0
                        and "rows" in kwargs
                        and isinstance(kwargs["rows"], list)
                        and all(isinstance(r, EvaluationRow) for r in kwargs["rows"])
                    ):
                        return await call_test_func(**kwargs)

                # If not a direct call, use the pytest wrapper
                return await pytest_wrapper(*args, **kwargs)

            # Copy all attributes from the pytest wrapper to our dual mode wrapper
            import functools

            functools.update_wrapper(dual_mode_wrapper, pytest_wrapper)

            return dual_mode_wrapper

        # Create the dual mode wrapper
        dual_mode_wrapper = create_dual_mode_wrapper()

        # Attach metadata so non-pytest runners (e.g., export_benchmark) can reconstruct runs
        try:
            dual_mode_wrapper.__ep_original_test_func = test_func  # type: ignore[attr-defined]
            dual_mode_wrapper.__ep_config = {
                "input_messages": input_messages,
                "input_dataset": input_dataset,
                "dataset_adapter": dataset_adapter,
                "rollout_input_params": completion_params,
                "rollout_processor": rollout_processor,
                "evaluation_test_kwargs": evaluation_test_kwargs,
                "rollout_processor_kwargs": rollout_processor_kwargs,
                "aggregation_method": aggregation_method,
                "passed_threshold": passed_threshold,
                "num_runs": num_runs,
                "max_dataset_rows": max_dataset_rows,
                "mcp_config_path": mcp_config_path,
                "max_concurrent_rollouts": max_concurrent_rollouts,
                "server_script_path": server_script_path,
                "steps": steps,
                "mode": mode,
                "combine_datasets": combine_datasets,
            }  # type: ignore[attr-defined]

            # Provide a direct runner method to avoid external imports
            def __ep_run_direct(
                *,
                model_override: str | None = None,
                num_runs_override: int | None = None,
                rollout_input_params_override: Dict[str, Any] | None = None,
            ):
                cfg = dual_mode_wrapper.__ep_config  # type: ignore[attr-defined]
                models = cfg.get("model") or []
                _model = model_override or (models[0] if models else None)
                if not _model:
                    raise ValueError("No model provided for direct run")
                rip = rollout_input_params_override
                if rip is None:
                    rip_list = cfg.get("rollout_input_params")
                    rip = rip_list[0] if isinstance(rip_list, list) and rip_list else {}
                return run_evaluation_test_direct(
                    test_func=dual_mode_wrapper.__ep_original_test_func,  # type: ignore[attr-defined]
                    input_messages=cfg.get("input_messages"),
                    input_dataset=cfg.get("input_dataset"),
                    dataset_adapter=cfg.get("dataset_adapter"),
                    completion_params=rip,
                    rollout_processor=cfg.get("rollout_processor"),
                    aggregation_method=cfg.get("aggregation_method"),
                    passed_threshold=cfg.get("passed_threshold"),
                    num_runs=(num_runs_override if num_runs_override is not None else cfg.get("num_runs")),
                    max_dataset_rows=cfg.get("max_dataset_rows"),
                    mcp_config_path=cfg.get("mcp_config_path"),
                    max_concurrent_rollouts=cfg.get("max_concurrent_rollouts"),
                    server_script_path=cfg.get("server_script_path"),
                    steps=cfg.get("steps"),
                    mode=cfg.get("mode"),
                    combine_datasets=cfg.get("combine_datasets"),
                )

            dual_mode_wrapper.__ep_run_direct = __ep_run_direct  # type: ignore[attr-defined]
        except Exception:
            # Best-effort; never fail pytest setup due to metadata attachment
            pass

        return dual_mode_wrapper

    return decorator


def run_evaluation_test_direct(
    *,
    test_func: TestFunction,
    input_messages: Optional[List[InputMessagesParam]] = None,
    input_dataset: Optional[List[DatasetPathParam]] = None,
    dataset_adapter: Callable[[List[Dict[str, Any]]], Dataset] = default_dataset_adapter,
    completion_params: Optional[CompletionParams] = None,
    rollout_processor: RolloutProcessor = default_no_op_rollout_processor,
    rollout_processor_kwargs: Optional[RolloutProcessorInputParam] = None,
    aggregation_method: AggregationMethod = "mean",
    passed_threshold: Optional[Union[EvaluationThreshold, float]] = None,
    num_runs: int = 1,
    max_dataset_rows: Optional[int] = None,
    mcp_config_path: Optional[str] = None,
    max_concurrent_rollouts: int = 8,
    server_script_path: Optional[str] = None,
    steps: int = 30,
    mode: EvaluationTestMode = "batch",
    combine_datasets: bool = True,
) -> Dict[str, Any]:
    """
    Programmatic runner that executes the same pipeline as @evaluation_test without pytest.
    Honors EP_* env overrides and emits the same summary/JSON artifact.
    Returns a dict with keys: summary, results.
    """

    if passed_threshold is not None and not isinstance(passed_threshold, EvaluationThreshold):
        passed_threshold = EvaluationThreshold(success=passed_threshold)

    def _parse_ep_max_rows(default_value: int | None) -> int | None:
        raw = os.getenv("EP_MAX_DATASET_ROWS")
        if raw is None:
            return default_value
        s = raw.strip().lower()
        if s == "none":
            return None
        try:
            return int(s)
        except ValueError:
            return default_value

    def _deep_update_dict(base: dict, override: dict) -> dict:
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                _deep_update_dict(base[key], value)
            else:
                base[key] = value
        return base

    # Build dataset/messages
    data: List[EvaluationRow] = []
    if input_dataset is not None:
        # Concatenate rows across multiple paths/URLs
        data_jsonl: List[Dict[str, Any]] = []
        for p in input_dataset:
            data_jsonl.extend(load_jsonl(p))
        effective_max_rows = _parse_ep_max_rows(max_dataset_rows)
        if effective_max_rows is not None:
            data_jsonl = data_jsonl[:effective_max_rows]
        data = dataset_adapter(data_jsonl)
    elif input_messages is not None:
        effective_max_rows = _parse_ep_max_rows(max_dataset_rows)
        msgs = input_messages
        if effective_max_rows is not None and isinstance(msgs, list):
            msgs = msgs[:effective_max_rows]  # type: ignore
        if isinstance(msgs, list) and msgs and isinstance(msgs[0], Message):
            data = [EvaluationRow(messages=msgs)]  # type: ignore[arg-type]
        else:
            data = [EvaluationRow(messages=m) for m in msgs]  # type: ignore
    else:
        raise ValueError("No input dataset or input messages provided")

    # Build input params and apply env JSON override
    completion_params: Dict[str, Any] = completion_params or {}
    try:
        import json as _json

        _env_override = os.getenv("EP_INPUT_PARAMS_JSON")
        if _env_override:
            override_obj = _json.loads(_env_override)
            if isinstance(override_obj, dict):
                completion_params = _deep_update_dict(dict(completion_params), override_obj)
    except Exception:
        pass

    # Prepare metadata
    eval_metadata = EvalMetadata(
        name=test_func.__name__,
        description=test_func.__doc__,
        status="running",
        num_runs=num_runs,
        aggregation_method=aggregation_method,
        passed_threshold=passed_threshold,
        passed=None,
    )

    for row in data:
        if row.input_metadata is None:
            row.input_metadata = InputMetadata()
        row.input_metadata.completion_params = completion_params
        if row.input_metadata.session_data is None:
            row.input_metadata.session_data = {}
        row.input_metadata.session_data["mode"] = mode
        row.eval_metadata = eval_metadata
        row.pid = os.getpid()
        default_logger.log(row)

    config = RolloutProcessorConfig(
        completion_params=completion_params,
        mcp_config_path=mcp_config_path or "",
        max_concurrent_rollouts=max_concurrent_rollouts,
        server_script_path=server_script_path,
        steps=steps,
        kwargs=rollout_processor_kwargs or {},
    )

    all_results: List[EvaluationRow] = []
    try:
        for _ in range(num_runs):
            fresh_rows = [copy.deepcopy(r) for r in data]
            processed_rows = execute_function(rollout_processor, rows=fresh_rows, config=config)
            if mode == "pointwise":
                for row in processed_rows:
                    result = execute_function(test_func, row=row)
                    if result is None or not isinstance(result, EvaluationRow):
                        raise ValueError(
                            f"Test function {test_func.__name__} did not return an EvaluationRow instance."
                        )
                    all_results.append(result)
            else:
                results = execute_function(test_func, rows=processed_rows)
                if results is None or not isinstance(results, list) or not results:
                    raise ValueError(
                        f"Test function {test_func.__name__} did not return a non-empty list of EvaluationRow instances."
                    )
                if not all(isinstance(r, EvaluationRow) for r in results):
                    raise ValueError(
                        f"Test function {test_func.__name__} returned a list containing non-EvaluationRow instances."
                    )
                all_results.extend(results)

        scores = [r.evaluation_result.score for r in all_results if r.evaluation_result]
        agg_score = aggregate(scores, aggregation_method)

        ci_low: float | None = None
        ci_high: float | None = None
        if aggregation_method == "mean":
            try:
                result_ci = compute_fixed_set_mu_ci(all_results)
                mu_ci_low, mu_ci_high = result_ci[1], result_ci[2]
                if mu_ci_low is not None and mu_ci_high is not None:
                    ci_low = float(mu_ci_low)
                    ci_high = float(mu_ci_high)
            except Exception:
                ci_low = None
                ci_high = None

        passed = None
        if passed_threshold is not None:
            passed = agg_score >= passed_threshold.success
        for r in all_results:
            if r.eval_metadata is not None:
                r.eval_metadata.status = "finished"
                r.eval_metadata.passed = passed
            default_logger.log(r)

        # Summary/JSON artifact (same EP_* env behavior)
        summary_obj: Dict[str, Any] = {}
        try:
            should_print = os.getenv("EP_PRINT_SUMMARY") == "1"
            summary_path = os.getenv("EP_SUMMARY_JSON")
            suite_name = test_func.__name__
            total_rows = len(all_results)
            summary_obj = {
                "suite": suite_name,
                "model": config.completion_params["model"],
                "agg_score": float(agg_score) if agg_score is not None else None,
                "num_runs": num_runs,
                "rows": total_rows,
            }
            if ci_low is not None and ci_high is not None:
                summary_obj["agg_ci_low"] = ci_low
                summary_obj["agg_ci_high"] = ci_high
            if should_print:
                if ci_low is not None and ci_high is not None:
                    print(
                        f"EP Summary | suite={suite_name} model={config.completion_params['model']} agg={summary_obj['agg_score']:.3f} ci95=[{ci_low:.3f},{ci_high:.3f}] runs={num_runs} rows={total_rows}"
                    )
                else:
                    print(
                        f"EP Summary | suite={suite_name} model={config.completion_params['model']} agg={summary_obj['agg_score']:.3f} runs={num_runs} rows={total_rows}"
                    )
            if summary_path:
                import json as _json
                import pathlib as _pathlib
                import re as _re
                import time as _time

                def _sanitize_filename(text: str) -> str:
                    safe = _re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip())
                    return safe[:120]

                def _extract_effort_tag(params: dict) -> str | None:
                    try:
                        if not isinstance(params, dict):
                            return None
                        if "extra_body" in params and isinstance(params["extra_body"], dict):
                            eb = params["extra_body"]
                            if isinstance(eb.get("reasoning"), dict) and "effort" in eb["reasoning"]:
                                return str(eb["reasoning"]["effort"]).lower()
                            if "reasoning_effort" in eb:
                                return str(eb["reasoning_effort"]).lower()
                        if (
                            "reasoning" in params
                            and isinstance(params["reasoning"], dict)
                            and "effort" in params["reasoning"]
                        ):
                            return str(params["reasoning"]["effort"]).lower()
                    except Exception:
                        return None
                    return None

                model_slug = _sanitize_filename(config.completion_params["model"])
                effort_tag = _extract_effort_tag(completion_params) or ""
                effort_suffix = f"__effort-{_sanitize_filename(effort_tag)}" if effort_tag else ""
                base_name = f"{suite_name}__{model_slug}{effort_suffix}__{mode}__runs{num_runs}.json"

                p = _pathlib.Path(summary_path)
                summary_obj["timestamp"] = int(_time.time())
                if p.suffix.lower() != ".json" or str(summary_path).endswith("/") or p.is_dir():
                    out_dir = p
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_file = out_dir / base_name
                else:
                    parent = p.parent
                    parent.mkdir(parents=True, exist_ok=True)
                    if effort_tag:
                        out_file = parent / f"{p.stem}__{_sanitize_filename(effort_tag)}{p.suffix}"
                    else:
                        out_file = p
                with open(out_file, "w", encoding="utf-8") as f:
                    _json.dump(summary_obj, f)
        except Exception:
            pass

        if passed_threshold is not None and not passed:
            assert (
                agg_score >= passed_threshold.success
            ), f"Aggregated score {agg_score:.3f} below threshold {passed_threshold}"

        return {"summary": summary_obj, "results": all_results}
    except Exception:
        # Mark errors on rows
        if eval_metadata is not None:
            eval_metadata.status = "error"
            eval_metadata.passed = False
            for r in data or []:
                if r.eval_metadata is not None:
                    r.eval_metadata.status = "error"
                    r.eval_metadata.passed = False
                default_logger.log(r)
        raise
