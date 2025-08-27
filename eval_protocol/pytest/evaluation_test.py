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
from collections import defaultdict
import hashlib
import ast
from mcp.types import Completion
import pytest

from eval_protocol.dataset_logger import default_logger
from eval_protocol.dataset_logger.dataset_logger import DatasetLogger
from eval_protocol.human_id import generate_id, num_combinations
from eval_protocol.models import (
    CompletionParams,
    ErrorInfo,
    EvalMetadata,
    EvaluationRow,
    EvaluationThreshold,
    InputMetadata,
    Message,
    Status,
)
from eval_protocol.pytest.default_dataset_adapter import default_dataset_adapter
from eval_protocol.pytest.default_mcp_gym_rollout_processor import MCPGymRolloutProcessor
from eval_protocol.pytest.default_no_op_rollout_processor import NoOpRolloutProcessor
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.types import (
    Dataset,
    DatasetPathParam,
    EvaluationInputParam,
    EvaluationTestMode,
    InputMessagesParam,
    ModelParam,
    RolloutProcessorConfig,
    RolloutProcessorInputParam,
    TestFunction,
)
from eval_protocol.pytest.utils import (
    AggregationMethod,
    aggregate,
    create_dynamically_parameterized_wrapper,
    deep_update_dict,
    extract_effort_tag,
    generate_parameter_combinations,
    log_eval_status_and_rows,
    parse_ep_max_rows,
    parse_ep_max_concurrent_rollouts,
    parse_ep_num_runs,
    parse_ep_completion_params,
    parse_ep_passed_threshold,
    rollout_processor_with_retry,
    sanitize_filename,
)
from eval_protocol.pytest.exception_config import ExceptionHandlerConfig
from eval_protocol.stats.confidence_intervals import compute_fixed_set_mu_ci
from eval_protocol.types.types import TerminationReason

from ..common_utils import load_jsonl


def postprocess(
    all_results: List[List[EvaluationRow]],
    aggregation_method: AggregationMethod,
    threshold: Optional[EvaluationThreshold],
    active_logger: DatasetLogger,
    mode: EvaluationTestMode,
    completion_params: CompletionParams,
    test_func_name: str,
    num_runs: int,
):
    scores = [
        sum([r.evaluation_result.score for r in result if r.evaluation_result]) / len(result) for result in all_results
    ]
    agg_score = aggregate(scores, aggregation_method)

    # Compute 95% confidence interval for the fixed-set mean μ (by-question, using repeats)
    ci_low: float | None = None
    ci_high: float | None = None
    if aggregation_method == "mean":
        try:
            result_ci = compute_fixed_set_mu_ci([item for sublist in all_results for item in sublist])
            _, mu_ci_low, mu_ci_high, se = result_ci
            if mu_ci_low is not None and mu_ci_high is not None and se is not None:
                ci_low = float(mu_ci_low)
                ci_high = float(mu_ci_high)
                standard_error = float(se)
                # Keep agg_score as-is (mean over scores). For equal repeats per question these match.
        except Exception:
            ci_low = None
            ci_high = None
            standard_error = None

    # Determine if the evaluation passed based on threshold
    passed = None

    if threshold is not None:
        success_passed, standard_error_passed = True, True

        success_passed = agg_score >= threshold.success

        if threshold.standard_error is not None and standard_error is not None:
            standard_error_passed = standard_error <= threshold.standard_error

        passed = success_passed and standard_error_passed

    # Update eval metadata passed field for all results
    for result in all_results:
        for r in result:
            if r.eval_metadata is not None:
                r.eval_metadata.passed = passed
            if r.evaluation_result is not None:
                r.evaluation_result.agg_score = agg_score
                r.evaluation_result.standard_error = standard_error
            active_logger.log(r)

    # Optional: print and/or persist a summary artifact for CI
    try:
        should_print = os.getenv("EP_PRINT_SUMMARY") == "1"
        summary_path = os.getenv("EP_SUMMARY_JSON")
        suite_name = test_func_name
        model_used = completion_params["model"]
        total_rows = len([item for sublist in all_results for item in sublist])
        summary_obj = {
            "suite": suite_name,
            "model": model_used,
            "agg_score": float(agg_score) if agg_score is not None else None,
            "num_runs": num_runs,
            "rows": total_rows,
        }
        if ci_low is not None and ci_high is not None and standard_error is not None:
            summary_obj["agg_ci_low"] = ci_low
            summary_obj["agg_ci_high"] = ci_high
            summary_obj["standard_error"] = standard_error

        # Aggregate per-metric mean and 95% CI when available
        metrics_summary: Dict[str, Dict[str, float]] = {}

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
            if ci_low is not None and ci_high is not None and standard_error is not None:
                print(
                    f"EP Summary | suite={suite_name} model={model_used} agg={summary_obj['agg_score']:.3f} se={summary_obj['standard_error']:.3f} ci95=[{ci_low:.3f},{ci_high:.3f}] runs={num_runs} rows={total_rows}"
                )
            else:
                print(
                    f"EP Summary | suite={suite_name} model={model_used} agg={summary_obj['agg_score']:.3f} runs={num_runs} rows={total_rows}"
                )
            # As per project convention, avoid printing per-metric CI lines to reduce noise
        if summary_path:
            model_slug = sanitize_filename(model_used)
            effort_tag = extract_effort_tag(completion_params) or ""
            effort_suffix = f"__effort-{sanitize_filename(effort_tag)}" if effort_tag else ""
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
                    out_file = parent / f"{p.stem}__{sanitize_filename(effort_tag)}{p.suffix}"
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
        assert agg_score >= threshold.success, f"Aggregated score {agg_score:.3f} below threshold {threshold.success}"
        if threshold.standard_error is not None and standard_error is not None:
            assert standard_error <= threshold.standard_error, (
                f"Standard error {standard_error:.3f} above threshold {threshold.standard_error}"
            )


def evaluation_test(  # noqa: C901
    *,
    completion_params: List[CompletionParams],
    input_messages: Optional[List[InputMessagesParam]] = None,
    input_dataset: Optional[List[DatasetPathParam]] = None,
    input_rows: Optional[List[EvaluationRow]] = None,
    dataset_adapter: Callable[[List[Dict[str, Any]]], Dataset] = default_dataset_adapter,
    rollout_processor: RolloutProcessor = NoOpRolloutProcessor(),
    evaluation_test_kwargs: Optional[List[EvaluationInputParam]] = None,
    rollout_processor_kwargs: Optional[RolloutProcessorInputParam] = None,
    aggregation_method: AggregationMethod = "mean",
    passed_threshold: Optional[Union[EvaluationThreshold, float, dict]] = None,
    num_runs: int = 1,
    max_dataset_rows: Optional[int] = None,
    mcp_config_path: Optional[str] = None,
    max_concurrent_rollouts: int = 8,
    max_concurrent_evaluations: int = 64,
    server_script_path: Optional[str] = None,
    steps: int = 30,
    mode: EvaluationTestMode = "pointwise",
    combine_datasets: bool = True,
    logger: Optional[DatasetLogger] = None,
    exception_handler_config: Optional[ExceptionHandlerConfig] = None,
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
        input_rows: Pre-constructed EvaluationRow objects to use directly. This is useful
            when you want to provide EvaluationRow objects with custom metadata, input_messages,
            or other fields already populated. Will be passed as "input_dataset" to the test function.
        dataset_adapter: Function to convert the input dataset to a list of
            EvaluationRows. This is useful if you have a custom dataset format.
        completion_params: Generation parameters for the rollout.
        rollout_processor: Function used to perform the rollout.
        evaluation_test_kwargs: Kwargs for the evaluation function.
        rollout_processor_kwargs: Kwargs for the rollout processor.
        aggregation_method: How to aggregate scores across rows.
        passed_threshold: Threshold configuration for test success. Must be a float or EvaluationThreshold object.
            Success rate must be above success, and if set, standard error must be below standard_error.
            Success rate +/- one standard_error is equivalent to 68% confidence interval.
        num_runs: Number of times to repeat the rollout and evaluations.
        max_dataset_rows: Limit dataset to the first N rows.
        mcp_config_path: Path to MCP config file that follows MCPMultiClientConfiguration schema
        max_concurrent_rollouts: Maximum number of concurrent rollouts to run in parallel.
        max_concurrent_evaluations: Maximum number of concurrent evaluations to run in parallel.
        server_script_path: Path to the MCP server script to run (default: "examples/tau2_mcp/server.py").
        steps: Number of rollout steps to execute (default: 30).
        mode: Evaluation mode. "pointwise" (default) applies test function to each row (rollout result).
            "groupwise" applies test function to a group of rollout results from the same original row (for use cases such as dpo/grpo).
            "all" applies test function to the whole dataset.
        logger: DatasetLogger to use for logging. If not provided, a default logger will be used.
        exception_handler_config: Configuration for exception handling and backoff retry logic.
            If not provided, a default configuration will be used with common retryable exceptions.
    """

    active_logger: DatasetLogger = logger if logger else default_logger

    # Optional global overrides via environment for ad-hoc experimentation
    # EP_INPUT_PARAMS_JSON can contain a JSON object that will be deep-merged
    # into input_params (e.g., '{"temperature":0,"extra_body":{"reasoning":{"effort":"low"}}}').
    num_runs = parse_ep_num_runs(num_runs)
    max_concurrent_rollouts = parse_ep_max_concurrent_rollouts(max_concurrent_rollouts)
    max_dataset_rows = parse_ep_max_rows(max_dataset_rows)
    completion_params = parse_ep_completion_params(completion_params)
    original_completion_params = completion_params
    passed_threshold = parse_ep_passed_threshold(passed_threshold)

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
        if not completion_params:
            raise ValueError("completion_params is required")

        # For pointwise/groupwise mode, we expect a different signature
        # we expect single row to be passed in as the original row
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

            # additional check for groupwise evaluation
        elif mode == "groupwise":
            if "rows" not in sig.parameters:
                raise ValueError("In groupwise mode, your eval function must have a parameter named 'rows'")

            # validate that "Rows" is of type List[EvaluationRow]
            if sig.parameters["rows"].annotation is not List[EvaluationRow]:
                raise ValueError("In groupwise mode, the 'rows' parameter must be of type List[EvaluationRow")

            # validate that the function has a return type of List[EvaluationRow]
            if sig.return_annotation is not List[EvaluationRow]:
                raise ValueError("In groupwise mode, your eval function must return a list of EvaluationRow instances")
            if len(completion_params) < 2:
                raise ValueError("In groupwise mode, you must provide at least 2 completion parameters")
        else:
            # all mode: function should accept input_dataset and model
            if "rows" not in sig.parameters:
                raise ValueError("In all mode, your eval function must have a parameter named 'rows'")

            # validate that "Rows" is of type List[EvaluationRow]
            if sig.parameters["rows"].annotation is not List[EvaluationRow]:
                raise ValueError("In all mode, the 'rows' parameter must be of type List[EvaluationRow")

            # validate that the function has a return type of List[EvaluationRow]
            if sig.return_annotation is not List[EvaluationRow]:
                raise ValueError("In all mode, your eval function must return a list of EvaluationRow instances")

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
        if mode == "groupwise":
            combinations = generate_parameter_combinations(
                input_dataset,
                completion_params,
                input_messages,
                input_rows,
                evaluation_test_kwargs,
                max_dataset_rows,
                combine_datasets,
            )
        else:
            combinations = generate_parameter_combinations(
                input_dataset,
                completion_params,
                input_messages,
                input_rows,
                evaluation_test_kwargs,
                max_dataset_rows,
                combine_datasets,
            )
        if len(combinations) == 0:
            raise ValueError(
                "No combinations of parameters were found. Please provide at least a model and one of input_dataset, input_messages, or input_rows."
            )

        # Create parameter tuples for pytest.mark.parametrize
        param_tuples = []
        for combo in combinations:
            dataset, cp, messages, rows, etk = combo
            param_tuple = []
            if input_dataset is not None:
                param_tuple.append(dataset)
            if completion_params is not None:
                param_tuple.append(cp)
            if input_messages is not None:
                param_tuple.append(messages)
            if input_rows is not None:
                param_tuple.append(rows)
            if evaluation_test_kwargs is not None:
                param_tuple.append(etk)
            param_tuples.append(tuple(param_tuple))

        # For all mode, preserve the original parameter names
        test_param_names = []
        if input_dataset is not None:
            test_param_names.append("dataset_path")
        if completion_params is not None:
            test_param_names.append("completion_params")
        if input_messages is not None:
            test_param_names.append("input_messages")
        if input_rows is not None:
            test_param_names.append("input_rows")
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

                def _log_eval_error(status: Status, rows: Optional[List[EvaluationRow]] | None, passed: bool) -> None:
                    log_eval_status_and_rows(eval_metadata, rows, status, passed, active_logger)

                try:
                    # Handle dataset loading
                    data: List[EvaluationRow] = []
                    # Track all rows processed in the current run for error logging
                    processed_rows_in_run: List[EvaluationRow] = []
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
                        # Apply override for max rows if present
                        if max_dataset_rows is not None:
                            data_jsonl = data_jsonl[:max_dataset_rows]
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
                    elif "input_rows" in kwargs and kwargs["input_rows"] is not None:
                        # Use pre-constructed EvaluationRow objects directly
                        data = kwargs["input_rows"]
                    else:
                        raise ValueError("No input dataset, input messages, or input rows provided")

                    for row in data:
                        # generate a stable row_id for each row
                        if row.input_metadata.row_id is None:
                            # Generate a stable, deterministic row_id using the row's hash and num_combinations
                            index = hash(row)
                            max_index = num_combinations() - 1
                            # Ensure index is a non-negative integer within [0, max_index]
                            index = abs(index) % (max_index + 1)
                            row.input_metadata.row_id = generate_id(seed=0, index=index)

                    completion_params = kwargs["completion_params"]
                    if completion_params and ("model" not in completion_params or not completion_params["model"]):
                        raise ValueError(
                            "No model provided. Please provide a model in the completion parameters object."
                        )

                    # Create eval metadata with test function info and current commit hash
                    eval_metadata = EvalMetadata(
                        name=test_func.__name__,
                        description=test_func.__doc__,
                        status=Status.eval_running(),
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
                        exception_handler_config=exception_handler_config,
                    )

                    async def execute_run(i: int, config: RolloutProcessorConfig):
                        nonlocal all_results

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
                            processed_rows_in_run.append(row)

                        # prepare parallel eval helper function
                        semaphore = asyncio.Semaphore(max_concurrent_evaluations)

                        async def _execute_eval_with_semaphore(**inner_kwargs):
                            async with semaphore:
                                # NOTE: we will still evaluate errored rows (give users control over this)
                                # i.e., they can choose to give EvaluateResult.score = 0 for errored rows in their test_func
                                if "row" in inner_kwargs:
                                    result = await execute_with_params(
                                        test_func,
                                        processed_row=inner_kwargs["row"],
                                        evaluation_test_kwargs=kwargs.get("evaluation_test_kwargs") or {},
                                    )
                                    if result is None or not isinstance(result, EvaluationRow):
                                        raise ValueError(
                                            f"Test function {test_func.__name__} did not return an EvaluationRow instance. You must return an EvaluationRow instance from your test function decorated with @evaluation_test."
                                        )
                                    return result
                                if "rows" in inner_kwargs:
                                    results = await execute_with_params(
                                        test_func,
                                        processed_dataset=inner_kwargs["rows"],
                                        evaluation_test_kwargs=kwargs.get("evaluation_test_kwargs") or {},
                                    )
                                    if results is None or not isinstance(results, list):
                                        raise ValueError(
                                            f"Test function {test_func.__name__} did not return a list of EvaluationRow instances. You must return a list of EvaluationRow instances from your test function decorated with @evaluation_test."
                                        )
                                    return results

                        if mode == "pointwise":
                            # Pointwise mode, rollouts will return as they complete so we can pipeline evaluation_test execution
                            tasks = []
                            # Use wrapper that handles retry logic internally
                            async for row in rollout_processor_with_retry(rollout_processor, fresh_dataset, config):
                                tasks.append(asyncio.create_task(_execute_eval_with_semaphore(row=row)))

                            results = await asyncio.gather(*tasks)

                            all_results[i] = results
                        elif mode == "groupwise":
                            # rollout all the completion_params for the same row at once, and then send the output to the test_func
                            row_groups = defaultdict(list)  # key: row_id, value: list of rollout_result
                            tasks: List[asyncio.Task[List[EvaluationRow]]] = []
                            # completion_groups = []
                            for idx, cp in enumerate(original_completion_params):
                                config = RolloutProcessorConfig(
                                    completion_params=cp,
                                    mcp_config_path=mcp_config_path or "",
                                    max_concurrent_rollouts=max_concurrent_rollouts,
                                    server_script_path=server_script_path,
                                    steps=steps,
                                    logger=active_logger,
                                    kwargs=rollout_processor_kwargs or {},
                                )
                                lst = []

                                async def _collect_result(config, lst):
                                    result = []
                                    async for row in rollout_processor_with_retry(rollout_processor, lst, config):
                                        result.append(row)
                                    return result

                                for ori_row in fresh_dataset:
                                    copied_row = ori_row.model_copy(deep=True)
                                    # overwrite the rollout_id to the index of the completion_params
                                    copied_row.execution_metadata.rollout_id = (
                                        str(ori_row.execution_metadata.rollout_id) + "_" + str(idx)
                                    )
                                    copied_row.input_metadata.completion_params = cp
                                    lst.append(copied_row)
                                tasks.append(asyncio.create_task(_collect_result(config, lst)))
                            rollout_results = await asyncio.gather(*tasks)
                            for result in rollout_results:
                                for row in result:
                                    row_groups[row.input_metadata.row_id].append(row)
                            tasks = []
                            for row_id, rows in row_groups.items():
                                tasks.append(asyncio.create_task(_execute_eval_with_semaphore(rows=rows)))
                            results = []
                            for task in tasks:
                                res = await task
                                results.extend(res)
                            all_results[i] = results
                        else:
                            # Batch mode: collect all results first, then evaluate (no pipelining)
                            input_dataset = []
                            async for row in rollout_processor_with_retry(rollout_processor, fresh_dataset, config):
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

                        for r in results:
                            if r.eval_metadata is not None:
                                if r.rollout_status.is_error():
                                    r.eval_metadata.status = Status.error(
                                        r.rollout_status.message, r.rollout_status.details
                                    )
                                else:
                                    r.eval_metadata.status = Status.eval_finished()
                            active_logger.log(r)

                    # if rollout_processor is McpGymRolloutProcessor, we execute runs sequentially since McpGym does not support concurrent runs
                    # else, we execute runs in parallel
                    if isinstance(rollout_processor, MCPGymRolloutProcessor):
                        # For MCPGymRolloutProcessor, create and execute tasks one at a time to avoid port conflicts
                        for i in range(num_runs):
                            task = asyncio.create_task(execute_run(i, config))
                            await task
                    else:
                        # For other processors, create all tasks at once and run in parallel
                        tasks = []
                        for i in range(num_runs):
                            tasks.append(asyncio.create_task(execute_run(i, config)))
                        await asyncio.gather(*tasks)

                    # for groupwise mode, the result contains eval otuput from multiple completion_params, we need to differentiate them
                    # rollout_id is used to differentiate the result from different completion_params
                    if mode == "groupwise":
                        results_by_group = [
                            [[] for _ in range(num_runs)] for _ in range(len(original_completion_params))
                        ]
                        for i_run, result in enumerate(all_results):
                            for r in result:
                                completion_param_idx = int(r.execution_metadata.rollout_id.split("_")[1])
                                results_by_group[completion_param_idx][i_run].append(r)
                        for rollout_id, result in enumerate(results_by_group):
                            postprocess(
                                result,
                                aggregation_method,
                                threshold,
                                active_logger,
                                mode,
                                original_completion_params[rollout_id],
                                test_func.__name__,
                                num_runs,
                            )
                    else:
                        postprocess(
                            all_results,
                            aggregation_method,
                            threshold,
                            active_logger,
                            mode,
                            completion_params,
                            test_func.__name__,
                            num_runs,
                        )

                except AssertionError:
                    _log_eval_error(
                        Status.eval_finished(),
                        processed_rows_in_run if "processed_rows_in_run" in locals() else None,
                        passed=False,
                    )
                    raise
                except Exception as e:
                    _log_eval_error(
                        Status.error(str(e)),
                        processed_rows_in_run if "processed_rows_in_run" in locals() else None,
                        passed=False,
                    )
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

            dual_mode_wrapper._origin_func = test_func
            dual_mode_wrapper._metainfo = {
                "mode": mode,
                "max_rollout_concurrency": max_concurrent_rollouts,
                "max_evaluation_concurrency": max_concurrent_evaluations,
            }

            # Copy all attributes from the pytest wrapper to our dual mode wrapper
            import functools

            functools.update_wrapper(dual_mode_wrapper, pytest_wrapper)

            return dual_mode_wrapper

        # Create the dual mode wrapper
        dual_mode_wrapper = create_dual_mode_wrapper()

        return dual_mode_wrapper

    return decorator
