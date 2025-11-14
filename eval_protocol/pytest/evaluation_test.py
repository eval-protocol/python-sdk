import asyncio
import inspect
import os
import time
from collections import defaultdict
from typing import Any, Callable
from typing_extensions import Unpack
from collections.abc import Sequence

import pytest

from eval_protocol.data_loader.models import EvaluationDataLoader
from eval_protocol.dataset_logger import default_logger
from eval_protocol.dataset_logger.dataset_logger import DatasetLogger
from eval_protocol.human_id import generate_id, num_combinations
from eval_protocol.models import (
    CompletionParams,
    EvalMetadata,
    EvaluationRow,
    EvaluationThreshold,
    EvaluationThresholdDict,
    EvaluateResult,
    Status,
)
from eval_protocol.pytest.dual_mode_wrapper import create_dual_mode_wrapper
from eval_protocol.pytest.evaluation_test_postprocess import postprocess
from eval_protocol.pytest.execution import execute_pytest, execute_pytest_with_exception_handling
from eval_protocol.pytest.generate_parameter_combinations import (
    ParameterizedTestKwargs,
    generate_parameter_combinations,
)
from eval_protocol.pytest.parameterize import pytest_parametrize, create_dynamically_parameterized_wrapper
from eval_protocol.pytest.validate_signature import validate_signature
from eval_protocol.pytest.default_dataset_adapter import default_dataset_adapter
from eval_protocol.pytest.default_mcp_gym_rollout_processor import MCPGymRolloutProcessor
from eval_protocol.pytest.default_no_op_rollout_processor import NoOpRolloutProcessor
from eval_protocol.pytest.exception_config import ExceptionHandlerConfig
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.types import (
    Dataset,
    DatasetPathParam,
    EvaluationInputParam,
    EvaluationTestMode,
    InputMessagesParam,
    RolloutProcessorConfig,
    RolloutProcessorInputParam,
    TestFunction,
)


from eval_protocol.pytest.evaluation_test_utils import (
    AggregationMethod,
    add_cost_metrics,
    log_eval_status_and_rows,
    parse_ep_completion_params,
    parse_ep_completion_params_overwrite,
    parse_ep_max_concurrent_rollouts,
    parse_ep_max_rows,
    parse_ep_num_runs,
    parse_ep_passed_threshold,
    parse_ep_dataloaders,
    rollout_processor_with_retry,
    run_tasks_with_eval_progress,
    run_tasks_with_run_progress,
)
from eval_protocol.utils.show_results_url import store_local_ui_results_url, generate_invocation_filter_url
from eval_protocol.log_utils.init import init_external_logging_from_env
from eval_protocol.log_utils.rollout_context import rollout_logging_context
from eval_protocol.utils.browser_utils import is_logs_server_running, open_browser_tab

from ..common_utils import load_jsonl


def evaluation_test(
    *,
    completion_params: Sequence[CompletionParams | None] | None = None,
    input_messages: Sequence[list[InputMessagesParam] | None] | None = None,
    input_dataset: Sequence[DatasetPathParam] | None = None,
    input_rows: Sequence[list[EvaluationRow]] | None = None,
    data_loaders: Sequence[EvaluationDataLoader] | EvaluationDataLoader | None = None,
    dataset_adapter: Callable[[list[dict[str, Any]]], Dataset] = default_dataset_adapter,
    rollout_processor: RolloutProcessor | None = None,
    evaluation_test_kwargs: Sequence[EvaluationInputParam | None] | None = None,
    rollout_processor_kwargs: RolloutProcessorInputParam | None = None,
    aggregation_method: AggregationMethod = "mean",
    passed_threshold: EvaluationThreshold | float | EvaluationThresholdDict | None = None,
    disable_browser_open: bool = False,
    num_runs: int = 1,
    filtered_row_ids: Sequence[str] | None = None,
    max_dataset_rows: int | None = None,
    mcp_config_path: str | None = None,
    max_concurrent_rollouts: int = 8,
    max_concurrent_evaluations: int = 64,
    server_script_path: str | None = None,
    steps: int = 30,
    mode: EvaluationTestMode = "pointwise",
    combine_datasets: bool = True,
    preprocess_fn: Callable[[list[EvaluationRow]], list[EvaluationRow]] | None = None,
    logger: DatasetLogger | None = None,
    exception_handler_config: ExceptionHandlerConfig | None = None,
) -> Callable[[TestFunction], TestFunction]:
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
        input_loaders: Data loaders to use to load the input dataset.
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
        filtered_row_ids: List of row_ids to filter for the evaluation. If provided, only the rows with the given row_ids will be evaluated.
        max_dataset_rows: Limit dataset to the first N rows.
        mcp_config_path: Path to MCP config file that follows MCPMultiClientConfiguration schema
        max_concurrent_rollouts: Maximum number of concurrent rollouts to run in parallel.
        max_concurrent_evaluations: Maximum number of concurrent evaluations to run in parallel.
        server_script_path: Path to the MCP server script to run (default: "examples/tau2_mcp/server.py").
        steps: Number of rollout steps to execute (default: 30).
        mode: Evaluation mode. "pointwise" (default) applies test function to each row (rollout result).
            "groupwise" applies test function to a group of rollout results from the same original row (for use cases such as dpo/grpo).
            "all" applies test function to the whole dataset.
        preprocess_fn: Optional preprocessing function that takes a list of EvaluationRow objects
            and returns a modified list. Useful for transformations like splitting multi-turn conversations,
            filtering data, or other preprocessing steps before rollout execution.
        logger: DatasetLogger to use for logging. If not provided, a default logger will be used.
        exception_handler_config: Configuration for exception handling and backoff retry logic.
            If not provided, a default configuration will be used with common retryable exceptions.
    """
    # Default to [None] when completion_params is not provided
    # This allows evaluation-only tests (e.g., using NoOpRolloutProcessor)
    # to work without requiring model generation parameters
    if completion_params is None:
        completion_params_provided = False
        completion_params = [None]
    else:
        completion_params_provided = True

    # Override rollout processor if flag is set
    if os.environ.get("EP_USE_NO_OP_ROLLOUT_PROCESSOR") == "1":
        rollout_processor = NoOpRolloutProcessor()
    elif rollout_processor is None:
        rollout_processor = NoOpRolloutProcessor()

    active_logger: DatasetLogger = logger if logger else default_logger

    if data_loaders is not None and (
        input_dataset is not None or input_messages is not None or input_rows is not None
    ):
        raise ValueError("data_loaders cannot be combined with input_dataset, input_messages, or input_rows.")

    # Optional global overrides via environment for ad-hoc experimentation
    # EP_INPUT_PARAMS_JSON can contain a JSON object that will be deep-merged
    # into input_params (e.g., '{"temperature":0,"extra_body":{"reasoning":{"effort":"low"}}}').
    num_runs = parse_ep_num_runs(num_runs)
    max_concurrent_rollouts = parse_ep_max_concurrent_rollouts(max_concurrent_rollouts)
    max_dataset_rows = parse_ep_max_rows(max_dataset_rows)
    completion_params = parse_ep_completion_params(completion_params)
    completion_params = parse_ep_completion_params_overwrite(completion_params)
    original_completion_params = completion_params
    passed_threshold = parse_ep_passed_threshold(passed_threshold)
    data_loaders = parse_ep_dataloaders(data_loaders)
    custom_invocation_id = os.environ.get("EP_INVOCATION_ID", None)

    # ignore other data input params when dataloader is provided
    if data_loaders:
        input_dataset = None
        input_messages = None
        input_rows = None

    def decorator(
        test_func: TestFunction,
    ) -> TestFunction:
        sig = inspect.signature(test_func)
        validate_signature(sig, mode, completion_params)

        # Calculate all possible combinations of parameters
        combinations = generate_parameter_combinations(
            input_dataset,
            completion_params,
            input_messages,
            input_rows,
            evaluation_test_kwargs,
            max_dataset_rows,
            combine_datasets,
            data_loaders,
        )
        if len(combinations) == 0:
            raise ValueError(
                "No combinations of parameters were found. Please provide at least a model and one of input_dataset, input_messages, or input_rows."
            )

        # Create parameter tuples for pytest.mark.parametrize
        pytest_parametrize_args = pytest_parametrize(
            combinations,
            test_func,
            input_dataset,
            completion_params,
            completion_params_provided,
            input_messages,
            input_rows,
            data_loaders,
            evaluation_test_kwargs,
        )

        # Create wrapper function with exact signature that pytest expects
        def create_wrapper_with_signature() -> Callable[[], None]:
            # Create the function body that will be used
            if custom_invocation_id:
                invocation_id = custom_invocation_id
            else:
                invocation_id = generate_id()

            # Track whether we've opened browser for this invocation
            browser_opened_for_invocation = False

            async def wrapper_body(**kwargs: Unpack[ParameterizedTestKwargs]) -> None:
                nonlocal browser_opened_for_invocation

                # Initialize external logging sinks (Fireworks/ES) from env (idempotent)
                init_external_logging_from_env()

                # Store URL for viewing results (after all postprocessing is complete)
                store_local_ui_results_url(invocation_id)

                # Auto-open browser if server is running and not disabled (only once per invocation)
                if (
                    not browser_opened_for_invocation
                    and not disable_browser_open
                    and os.environ.get("EP_DISABLE_AUTO_BROWSER") is None
                ):
                    is_running, port = is_logs_server_running()
                    if is_running:
                        # Generate URL for table view with invocation filter
                        base_url = f"http://localhost:{port}" if port else "http://localhost:8000"
                        table_url = generate_invocation_filter_url(invocation_id, f"{base_url}/table")
                        open_browser_tab(table_url)
                        browser_opened_for_invocation = True

                eval_metadata = None

                all_results: list[list[EvaluationRow]] = [[] for _ in range(num_runs)]

                experiment_id = generate_id()
                experiment_start_time = time.perf_counter()

                def _log_eval_error(status: Status, rows: list[EvaluationRow] | None, passed: bool) -> None:
                    log_eval_status_and_rows(eval_metadata, rows, status, passed, active_logger)

                try:
                    # Handle dataset loading
                    data: list[EvaluationRow] = []
                    # Track all rows processed in the current run for error logging
                    processed_rows_in_run: list[EvaluationRow] = []
                    if "data_loaders" in kwargs and kwargs["data_loaders"] is not None:
                        data_loaders = kwargs["data_loaders"]
                        data_loaders_list = (
                            [data_loaders] if isinstance(data_loaders, EvaluationDataLoader) else data_loaders
                        )
                        for data_loader in data_loaders_list:
                            results = data_loader.load()
                            for result in results:
                                data.extend(result.rows)
                        # Apply max_dataset_rows limit to data from data loaders
                        if max_dataset_rows is not None:
                            data = data[:max_dataset_rows]
                    elif "dataset_path" in kwargs and kwargs["dataset_path"] is not None:
                        ds_arg: list[str] = kwargs["dataset_path"]
                        # Support either a single path or a list of paths; if a list is provided,
                        # concatenate the rows from each file in order.
                        data_jsonl: list[dict[str, object]] = []
                        for p in ds_arg:
                            data_jsonl.extend(load_jsonl(p))
                        # Apply override for max rows if present
                        if max_dataset_rows is not None:
                            data_jsonl = data_jsonl[:max_dataset_rows]
                        data = dataset_adapter(data_jsonl)
                    elif "input_messages" in kwargs and kwargs["input_messages"] is not None:
                        # Support either a single row (List[Message]) or many rows (List[List[Message]])
                        im = kwargs["input_messages"]
                        data = [EvaluationRow(messages=dataset_messages) for dataset_messages in im]
                    elif "input_rows" in kwargs and kwargs["input_rows"] is not None:
                        # Deep copy pre-constructed EvaluationRow objects
                        data = [row.model_copy(deep=True) for row in kwargs["input_rows"]]
                    else:
                        raise ValueError("No input dataset, input messages, or input rows provided")

                    if filtered_row_ids is not None:
                        data = [row for row in data if row.input_metadata.row_id in filtered_row_ids]

                    """
                    data_loaders handles preprocess_fn internally so we want
                    to specially handle data_loaders here so we don't double
                    apply preprocess_fn.
                    """
                    if preprocess_fn:
                        if not data_loaders:
                            data = preprocess_fn(data)
                        else:
                            raise ValueError(
                                "preprocess_fn should not be used with data_loaders. Pass preprocess_fn to data_loaders instead."
                            )

                    for row in data:
                        # generate a stable row_id for each row
                        if row.input_metadata.row_id is None:
                            # Generate a stable, deterministic row_id using the row's hash and num_combinations
                            index = hash(row)
                            max_index = num_combinations() - 1
                            # Ensure index is a non-negative integer within [0, max_index]
                            index = abs(index) % (max_index + 1)
                            row.input_metadata.row_id = generate_id(seed=0, index=index)

                    completion_params = kwargs["completion_params"] if "completion_params" in kwargs else None
                    # Create eval metadata with test function info and current commit hash
                    eval_metadata = EvalMetadata(
                        name=test_func.__name__,
                        description=test_func.__doc__,
                        status=Status.eval_running(),
                        num_runs=num_runs,
                        aggregation_method=aggregation_method,
                        passed_threshold=passed_threshold,
                        passed=None,
                    )
                    for row in data:
                        row.input_metadata.completion_params = (
                            completion_params if completion_params is not None else {}
                        )
                        # Add mode to session_data
                        if row.input_metadata.session_data is None:
                            row.input_metadata.session_data = {}
                        row.input_metadata.session_data["mode"] = mode
                        # Initialize eval_metadata for each row
                        row.eval_metadata = eval_metadata.model_copy(deep=True)
                        row.execution_metadata.experiment_id = experiment_id
                        row.execution_metadata.invocation_id = invocation_id

                        # has to be done in the pytest main process since it's
                        # used to determine whether this eval has stopped
                        row.pid = os.getpid()

                    # Create shared semaphore for unified concurrency control across all runs and rollouts
                    shared_semaphore = asyncio.Semaphore(max_concurrent_rollouts)

                    # Prepare rollout processor config once; we will generate fresh outputs per run
                    config = RolloutProcessorConfig(
                        completion_params=completion_params if completion_params is not None else {},
                        mcp_config_path=mcp_config_path or "",
                        server_script_path=server_script_path,
                        steps=steps,
                        logger=active_logger,
                        semaphore=shared_semaphore,
                        kwargs=rollout_processor_kwargs or {},
                        exception_handler_config=exception_handler_config,
                    )

                    rollout_processor.setup()

                    async def execute_run(run_idx: int, config: RolloutProcessorConfig):
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

                        async def _execute_pointwise_eval_with_semaphore(
                            row: EvaluationRow,
                        ) -> EvaluationRow:
                            async with semaphore:
                                evaluation_test_kwargs = kwargs.get("evaluation_test_kwargs") or {}
                                async with rollout_logging_context(
                                    row.execution_metadata.rollout_id or "",
                                    experiment_id=experiment_id,
                                    run_id=run_id,
                                ):
                                    result = await execute_pytest_with_exception_handling(
                                        test_func=test_func,
                                        evaluation_test_kwargs=evaluation_test_kwargs,
                                        processed_row=row,
                                    )
                                if not isinstance(result, EvaluationRow):
                                    raise ValueError(
                                        f"Test function {test_func.__name__} did not return an EvaluationRow instance. You must return an EvaluationRow instance from your test function decorated with @evaluation_test."
                                    )
                                return result

                        async def _execute_groupwise_eval_with_semaphore(
                            rows: list[EvaluationRow],
                        ) -> list[EvaluationRow]:
                            async with semaphore:
                                evaluation_test_kwargs = kwargs.get("evaluation_test_kwargs") or {}
                                primary_rollout_id = rows[0].execution_metadata.rollout_id if rows else None
                                group_rollout_ids = [
                                    r.execution_metadata.rollout_id for r in rows if r.execution_metadata.rollout_id
                                ]
                                async with rollout_logging_context(
                                    primary_rollout_id or "",
                                    experiment_id=experiment_id,
                                    run_id=run_id,
                                    rollout_ids=group_rollout_ids or None,
                                ):
                                    results = await execute_pytest_with_exception_handling(
                                        test_func=test_func,
                                        evaluation_test_kwargs=evaluation_test_kwargs,
                                        processed_dataset=rows,
                                    )
                                if not isinstance(results, list):
                                    raise ValueError(
                                        f"Test function {test_func.__name__} did not return a list of EvaluationRow instances. You must return a list of EvaluationRow instances from your test function decorated with @evaluation_test."
                                    )
                                return results

                        if mode == "pointwise":
                            # Pointwise mode, rollouts will return as they complete so we can pipeline evaluation_test execution
                            pointwise_tasks: list[asyncio.Task[EvaluationRow]] = []
                            # Use wrapper that handles retry logic internally
                            async for row in rollout_processor_with_retry(
                                rollout_processor, fresh_dataset, config, run_idx
                            ):
                                pointwise_tasks.append(
                                    asyncio.create_task(_execute_pointwise_eval_with_semaphore(row=row))
                                )

                            # Run evaluation tasks with progress bar
                            results = await run_tasks_with_eval_progress(pointwise_tasks, run_idx)

                            all_results[run_idx] = results
                        elif mode == "groupwise":
                            # rollout all the completion_params for the same row at once, and then send the output to the test_func
                            row_groups = defaultdict(list)  # key: row_id, value: list of rollout_result
                            tasks: list[asyncio.Task[list[EvaluationRow]]] = []
                            # completion_groups = []
                            for idx, cp in enumerate(original_completion_params):
                                config = RolloutProcessorConfig(
                                    completion_params=cp if cp is not None else {},
                                    mcp_config_path=mcp_config_path or "",
                                    server_script_path=server_script_path,
                                    steps=steps,
                                    logger=active_logger,
                                    semaphore=shared_semaphore,
                                    kwargs=rollout_processor_kwargs or {},
                                )
                                lst = []

                                async def _collect_result(config, lst):
                                    result = []
                                    async for row in rollout_processor_with_retry(
                                        rollout_processor, lst, config, run_idx
                                    ):  # pyright: ignore[reportUnknownArgumentType]
                                        result.append(row)
                                    return result

                                for ori_row in fresh_dataset:
                                    copied_row = ori_row.model_copy(deep=True)
                                    # overwrite the rollout_id to the index of the completion_params
                                    copied_row.execution_metadata.rollout_id = (
                                        str(ori_row.execution_metadata.rollout_id) + "_" + str(idx)
                                    )
                                    copied_row.input_metadata.completion_params = cp if cp is not None else {}
                                    lst.append(copied_row)
                                tasks.append(asyncio.create_task(_collect_result(config, lst)))
                            rollout_results = await asyncio.gather(*tasks)
                            for result in rollout_results:
                                for row in result:
                                    row_groups[row.input_metadata.row_id].append(row)
                            tasks = []
                            for _, rows in row_groups.items():
                                tasks.append(asyncio.create_task(_execute_groupwise_eval_with_semaphore(rows=rows)))
                            results = []
                            for task in tasks:
                                res = await task
                                results.extend(res)
                            all_results[run_idx] = results
                        else:
                            # Batch mode: collect all results first, then evaluate (no pipelining)
                            input_dataset = []
                            async for row in rollout_processor_with_retry(
                                rollout_processor, fresh_dataset, config, run_idx
                            ):
                                input_dataset.append(row)
                            # NOTE: we will still evaluate errored rows (give users control over this)
                            # i.e., they can choose to give EvaluateResult.score = 0 for errored rows in their test_func
                            primary_rollout_id = (
                                input_dataset[0].execution_metadata.rollout_id if input_dataset else None
                            )
                            group_rollout_ids = [
                                r.execution_metadata.rollout_id
                                for r in input_dataset
                                if r.execution_metadata.rollout_id
                            ]
                            async with rollout_logging_context(
                                primary_rollout_id or "",
                                experiment_id=experiment_id,
                                run_id=run_id,
                                rollout_ids=group_rollout_ids or None,
                            ):
                                results = await execute_pytest_with_exception_handling(
                                    test_func=test_func,
                                    evaluation_test_kwargs=kwargs.get("evaluation_test_kwargs") or {},
                                    processed_dataset=input_dataset,
                                )
                            if (
                                results is None
                                or not isinstance(results, list)
                                or not all(isinstance(r, EvaluationRow) for r in results)
                            ):
                                raise ValueError(
                                    f"Test function {test_func.__name__} did not return a list of EvaluationRow instances. You must return a list of EvaluationRow instances from your test function decorated with @evaluation_test."
                                )
                            if not results:
                                raise ValueError(
                                    f"Test function {test_func.__name__} returned an empty list. You must return a non-empty list of EvaluationRow instances from your test function decorated with @evaluation_test."
                                )
                            all_results[run_idx] = results

                        for r in results:
                            add_cost_metrics(r)
                            if r.eval_metadata is not None:
                                if r.rollout_status.is_error():
                                    r.eval_metadata.status = Status.error(
                                        r.rollout_status.message, r.rollout_status.details
                                    )
                                elif not (
                                    r.eval_metadata.status and r.eval_metadata.status.code != Status.Code.RUNNING
                                ):
                                    # if the eval_metadata status code has not been set to something else, consider it as finished
                                    r.eval_metadata.status = Status.eval_finished()
                            # Optional debug print for assistant/tool sequence
                            if os.getenv("EP_DEBUG_SERIALIZATION", "0").strip() == "1":
                                try:
                                    preview = [
                                        {
                                            "role": m.role,
                                            "len": len(m.content or "") if isinstance(m.content, str) else None,
                                            "tool_calls": len(m.tool_calls or [])
                                            if hasattr(m, "tool_calls") and isinstance(m.tool_calls, list)
                                            else 0,
                                            "tool_call_id": getattr(m, "tool_call_id", None),
                                            "name": getattr(m, "name", None),
                                        }
                                        for m in r.messages
                                    ]
                                    print("[EP-Log] Row messages:", preview)
                                except Exception:
                                    pass
                            active_logger.log(r)

                    # if rollout_processor is McpGymRolloutProcessor, we execute runs sequentially since McpGym does not support concurrent runs
                    # else, we execute runs in parallel
                    if isinstance(rollout_processor, MCPGymRolloutProcessor):
                        # For MCPGymRolloutProcessor, create and execute tasks one at a time to avoid port conflicts
                        for run_idx in range(num_runs):
                            task = asyncio.create_task(execute_run(run_idx, config))
                            await task
                    else:
                        # For other processors, create all tasks at once and run in parallel
                        # Concurrency is now controlled by the shared semaphore in each rollout processor
                        await run_tasks_with_run_progress(execute_run, num_runs, config)

                    experiment_duration_seconds = time.perf_counter() - experiment_start_time

                    # for groupwise mode, the result contains eval otuput from multiple completion_params, we need to differentiate them
                    # rollout_id is used to differentiate the result from different completion_params
                    if mode == "groupwise":
                        results_by_group = [
                            [[] for _ in range(num_runs)] for _ in range(len(original_completion_params))
                        ]
                        for i_run, result in enumerate(all_results):
                            for r in result:
                                completion_param_idx = int(r.execution_metadata.rollout_id.split("_")[1])  # pyright: ignore[reportOptionalMemberAccess]
                                results_by_group[completion_param_idx][i_run].append(r)
                        for rollout_id, result in enumerate(results_by_group):
                            postprocess(
                                result,
                                aggregation_method,
                                passed_threshold,
                                active_logger,
                                mode,
                                original_completion_params[rollout_id],  # pyright: ignore[reportArgumentType]
                                test_func.__name__,
                                num_runs,
                                experiment_duration_seconds,
                            )
                    else:
                        postprocess(
                            all_results,
                            aggregation_method,
                            passed_threshold,
                            active_logger,
                            mode,
                            completion_params,  # pyright: ignore[reportArgumentType]
                            test_func.__name__,
                            num_runs,
                            experiment_duration_seconds,
                        )

                except AssertionError:
                    _log_eval_error(
                        Status.eval_finished(),
                        locals().get("processed_rows_in_run", None),
                        passed=False,
                    )
                    raise
                except Exception as e:
                    _log_eval_error(
                        Status.error(str(e)),
                        locals().get("processed_rows_in_run", None),
                        passed=False,
                    )
                    raise

            return create_dynamically_parameterized_wrapper(
                test_func,
                wrapper_body,
                pytest_parametrize_args["sig_parameters"],
            )

        # Create the pytest wrapper
        pytest_wrapper = create_wrapper_with_signature()
        pytest_wrapper = pytest.mark.parametrize(**pytest_parametrize_args["pytest_parametrize_kwargs"])(
            pytest_wrapper
        )
        pytest_wrapper = pytest.mark.asyncio(pytest_wrapper)

        ep_params: dict[str, Any] = {
            "rollout_processor": rollout_processor,
            "server_script_path": server_script_path,
            "mcp_config_path": mcp_config_path,
            "rollout_processor_kwargs": rollout_processor_kwargs,
            "mode": mode,
        }

        # Create the dual mode wrapper
        dual_mode_wrapper = create_dual_mode_wrapper(
            test_func, mode, max_concurrent_rollouts, max_concurrent_evaluations, pytest_wrapper
        )

        setattr(dual_mode_wrapper, "__ep_params__", ep_params)

        # Make this pytest discoverable regardless of pytest configuration. So
        # you can name your eval whatever you want, as long as it's decorated
        # with @evaluation_test.
        dual_mode_wrapper.__test__ = True

        return dual_mode_wrapper  # pyright: ignore[reportReturnType, reportUnknownVariableType]

    return decorator
