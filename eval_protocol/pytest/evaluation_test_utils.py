import asyncio
from collections.abc import Sequence
import os
import re
import sys
from dataclasses import replace
from typing import Any, Literal, Callable, AsyncGenerator, Optional

from litellm.cost_calculator import cost_per_token
from tqdm import tqdm

from eval_protocol.dataset_logger.dataset_logger import DatasetLogger
from eval_protocol.models import (
    CostMetrics,
    CompletionParams,
    EvalMetadata,
    EvaluationRow,
    EvaluationThreshold,
    EvaluationThresholdDict,
    Status,
)
from eval_protocol.data_loader import DynamicDataLoader
from eval_protocol.data_loader.models import EvaluationDataLoader
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.default_mcp_gym_rollout_processor import MCPGymRolloutProcessor
from eval_protocol.pytest.types import (
    RolloutProcessorConfig,
    ServerMode,
)
from eval_protocol.pytest.exception_config import get_default_exception_handler_config

import logging
import json
import random
import statistics


logger = logging.getLogger(__name__)

AggregationMethod = Literal["mean", "max", "min", "bootstrap"]


async def run_tasks_with_eval_progress(
    pointwise_tasks: list[asyncio.Task[EvaluationRow]], run_idx: int
) -> list[EvaluationRow]:
    """
    Run evaluation tasks with a progress bar and proper cancellation handling.

    Args:
        pointwise_tasks: List of asyncio tasks to execute
        run_idx: Run index for progress bar positioning and naming

    Returns:
        Results from all tasks
    """
    eval_position = run_idx + 2  # Position after rollout progress bar
    with tqdm(
        total=len(pointwise_tasks),
        desc=f"  Eval {run_idx + 1}",
        unit="eval",
        file=sys.__stderr__,
        leave=False,
        position=eval_position,
        dynamic_ncols=True,
        miniters=1,
        mininterval=0.1,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    ) as eval_pbar:

        async def task_with_progress(task: asyncio.Task[EvaluationRow]) -> EvaluationRow:
            try:
                result = await task
                return result
            finally:
                eval_pbar.update(1)

        wrapped_tasks = [task_with_progress(task) for task in pointwise_tasks]
        try:
            results = await asyncio.gather(*wrapped_tasks)
            return results
        except Exception:
            # Propagate cancellation to the real tasks and await them to quiesce
            for task in pointwise_tasks:
                task.cancel()
            await asyncio.gather(*pointwise_tasks, return_exceptions=True)
            raise


async def run_tasks_with_run_progress(
    execute_run_func: Callable[[int, RolloutProcessorConfig], Any], num_runs: int, config: RolloutProcessorConfig
) -> None:
    """
    Run tasks with a parallel runs progress bar, preserving original logic.

    Args:
        execute_run_func: The execute_run function to call
        num_runs: Number of runs to execute
        config: Configuration to pass to execute_run_func
    """
    with tqdm(
        total=num_runs,
        desc="Runs (Parallel)",
        unit="run",
        file=sys.__stderr__,
        position=0,
        leave=True,
        dynamic_ncols=True,
        miniters=1,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    ) as run_pbar:

        async def execute_run_with_progress(run_idx: int, config: RolloutProcessorConfig) -> Any:
            result = await execute_run_func(run_idx, config)
            run_pbar.update(1)
            return result

        tasks: list[asyncio.Task[Any]] = []
        for run_idx in range(num_runs):
            tasks.append(asyncio.create_task(execute_run_with_progress(run_idx, config)))
        try:
            await asyncio.gather(*tasks)
        except Exception:
            # Propagate cancellation to tasks and await them to quiesce
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise


def calculate_bootstrap_scores(all_scores: list[float], n_boot: int = 100, seed: int | None = None) -> float:
    """
    Calculate the mean of bootstrap sample means for a list of scores.

    Args:
        all_scores: List of individual scores from all rows.
        n_boot: Number of bootstrap resamples to draw (default 100).
        seed: Optional RNG seed for reproducibility.

    Returns:
        Mean bootstrap score (float). Returns 0.0 if all_scores is empty.
    """
    if not all_scores:
        return 0.0

    rng = random.Random(seed) if seed is not None else random
    k = len(all_scores)
    bootstrap_means = [statistics.fmean(rng.choices(all_scores, k=k)) for _ in range(n_boot)]
    return float(statistics.fmean(bootstrap_means))


def aggregate(scores: list[float], method: AggregationMethod) -> float:
    if not scores:
        return 0.0
    if method == "mean":
        return sum(scores) / len(scores)
    if method == "max":
        return max(scores)
    if method == "min":
        return min(scores)
    if method == "bootstrap":
        return calculate_bootstrap_scores(scores)


def log_eval_status_and_rows(
    eval_metadata: EvalMetadata | None,
    rows: list[EvaluationRow] | None,
    status: Status,
    passed: bool,
    logger: DatasetLogger,
) -> None:
    """Update eval status and emit rows to the given logger.

    If no rows are provided, emits a minimal placeholder row so downstream
    consumers still observe a terminal status.
    """
    if eval_metadata is None:
        return

    eval_metadata.status = status
    eval_metadata.passed = passed

    rows_to_log: list[EvaluationRow] = rows or []
    if not rows_to_log:
        error_row = EvaluationRow(messages=[], eval_metadata=eval_metadata, evaluation_result=None)
        logger.log(error_row)
    else:
        for r in rows_to_log:
            if r.eval_metadata is not None:
                r.eval_metadata.status = status
            logger.log(r)


def parse_ep_max_rows(default_value: int | None) -> int | None:
    """Read EP_MAX_DATASET_ROWS env override as int or None.

    Assumes the environment variable was already validated by plugin.py.
    """
    raw = os.getenv("EP_MAX_DATASET_ROWS")
    if raw is None:
        return default_value
    # plugin.py stores "None" as string for the "all" case
    return None if raw.lower() == "none" else int(raw)


def parse_ep_num_runs(default_value: int) -> int:
    """Read EP_NUM_RUNS env override as int.

    Assumes the environment variable was already validated by plugin.py.
    """
    raw = os.getenv("EP_NUM_RUNS")
    return int(raw) if raw is not None else default_value


def parse_ep_max_concurrent_rollouts(default_value: int) -> int:
    """Read EP_MAX_CONCURRENT_ROLLOUTS env override as int.

    Assumes the environment variable was already validated by plugin.py.
    """
    raw = os.getenv("EP_MAX_CONCURRENT_ROLLOUTS")
    return int(raw) if raw is not None else default_value


def parse_ep_completion_params(
    completion_params: Sequence[CompletionParams | None] | None,
) -> Sequence[CompletionParams | None]:
    """Apply EP_INPUT_PARAMS_JSON overrides to completion_params.

    Reads the environment variable set by plugin.py and applies deep merge to each completion param.
    """
    if completion_params is None:
        return []
    try:
        _env_override = os.getenv("EP_INPUT_PARAMS_JSON")
        if _env_override:
            override_obj = json.loads(_env_override)  # pyright: ignore[reportAny]
            if isinstance(override_obj, dict):
                # Apply override to each completion_params item
                return [deep_update_dict(dict(cp), override_obj) for cp in completion_params if cp is not None]  # pyright: ignore[reportUnknownArgumentType]
    except Exception:
        pass
    return completion_params


def parse_ep_completion_params_overwrite(
    completion_params: Sequence[CompletionParams | None] | None,
) -> Sequence[CompletionParams | None]:
    new_completion_params = os.getenv("EP_COMPLETION_PARAMS")
    if new_completion_params:
        try:
            new_completion_params_list = json.loads(new_completion_params)
            if isinstance(new_completion_params_list, list):
                return new_completion_params_list
        except Exception:
            pass
    return completion_params or []


def _rows_from_jsonl(path: str) -> list[EvaluationRow]:
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(EvaluationRow(**json.loads(line)))
    except Exception as e:
        print(f"❌ Failed to load rows from JSONL at {path}: {e}")
        return []

    return rows


def parse_ep_dataloaders(
    dataloaders: Sequence[EvaluationDataLoader] | EvaluationDataLoader | None,
) -> Sequence[EvaluationDataLoader] | EvaluationDataLoader | None:
    try:
        load_from_jsonl_path = os.getenv("EP_JSONL_PATH")
        if load_from_jsonl_path:
            return DynamicDataLoader(generators=[lambda path=load_from_jsonl_path: _rows_from_jsonl(path)])
    except Exception:
        pass
    return dataloaders or None


def parse_ep_passed_threshold(
    default_value: float | EvaluationThresholdDict | EvaluationThreshold | None,
) -> EvaluationThreshold | None:
    """Read EP_PASSED_THRESHOLD env override as float or dict.

    Assumes the environment variable was already validated by plugin.py.
    Supports both float values (e.g., "0.8") and JSON dict format (e.g., '{"success":0.8}').
    """
    raw = os.getenv("EP_PASSED_THRESHOLD")
    if raw is not None:
        try:
            success_value = float(raw)
            return EvaluationThreshold(success=success_value)
        except ValueError:
            raise ValueError(f"EP_PASSED_THRESHOLD env var exists but can't be parsed: {raw}")
    if isinstance(default_value, float):
        return EvaluationThreshold(success=default_value)
    if isinstance(default_value, dict):
        return EvaluationThreshold(**default_value)
    if isinstance(default_value, EvaluationThreshold):
        return default_value
    return None


def deep_update_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]
    """Recursively update nested dictionaries in-place and return base."""
    for key, value in override.items():  # pyright: ignore[reportAny]
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update_dict(base[key], value)  # pyright: ignore[reportAny, reportUnknownArgumentType]
        else:
            base[key] = value
    return base


def _set_rollout_status_to_finished(result: EvaluationRow) -> None:
    # Only set to finished if execution finished while not
    # updating status itself. In the case that the rollout
    # processor set the status to an error, we want to
    # preserve the error so we do nothing in this case.
    # test_remote_fireworks_propagate_status.py verifies this.
    if result.rollout_status.is_running():
        result.rollout_status = Status.rollout_finished()


async def rollout_processor_with_retry(
    rollout_processor: RolloutProcessor,
    fresh_dataset: list[EvaluationRow],
    config: RolloutProcessorConfig,
    run_idx: int = 0,
) -> AsyncGenerator[EvaluationRow, None]:
    """
    Wrapper around rollout_processor that handles retry logic using the Python backoff library.

    Provides configurable exception handling with automatic retry for specific exception types:
    - Retryable exceptions (e.g., ConnectionError, TimeoutError) are automatically retried with backoff
    - Fail-fast exceptions (e.g., ValueError, TypeError) are not retried and return immediately
    - Unknown exceptions can be configured to either re-raise or return as failed rows

    The backoff behavior (exponential/constant, delays, max attempts) is fully configurable
    through the ExceptionHandlerConfig in the RolloutProcessorConfig.

    Yields results as they complete, allowing for concurrent processing while handling
    retries transparently in the background.
    """

    # Use provided exception handler config or fall back to default
    # Environment variable overrides are automatically applied in __post_init__
    exception_config = config.exception_handler_config or get_default_exception_handler_config()

    try:
        # Create initial batch of tasks (preserves indexing for mock processors)
        try:
            base_tasks = rollout_processor(fresh_dataset, config)
        except Exception as e:
            print(f"❌ Rollout processor failed to initialize: {e}")
            raise e

        # Create a single backoff-decorated retry function that can be reused
        @exception_config.get_backoff_decorator()  # pyright: ignore[reportUntypedFunctionDecorator]
        async def execute_row_with_backoff_retry(row: EvaluationRow) -> EvaluationRow:
            """Execute rollout for a single row with backoff retry."""
            retry_config = replace(config, kwargs={**(config.kwargs or {}), "start_server": False})
            retry_tasks = rollout_processor([row], retry_config)
            return await retry_tasks[0]

        async def execute_row_with_backoff(task: asyncio.Task[EvaluationRow], row: EvaluationRow) -> EvaluationRow:
            """Execute a single row task with backoff retry."""

            try:
                # Try original task first
                result = await task  # pyright: ignore[reportUnknownVariableType]

                _set_rollout_status_to_finished(result)

                return result  # pyright: ignore[reportUnknownVariableType]
            except Exception as e:
                # NOTE: we perform these checks because we don't put the backoff decorator on initial batch call. we don't want to retry whole batch if anything fails.
                # Check if this exception should be retried
                is_retryable = any(isinstance(e, exc_type) for exc_type in exception_config.retryable_exceptions)
                giveup_func = exception_config.backoff_config.giveup_func
                should_giveup = giveup_func and giveup_func(e)

                if is_retryable and not should_giveup:
                    # Use shared backoff function for retryable exceptions
                    try:
                        result = await execute_row_with_backoff_retry(row)

                        _set_rollout_status_to_finished(result)

                        return result
                    except Exception as retry_error:
                        # Backoff gave up
                        logging.error(
                            f"❌ Rollout failed, (retried {exception_config.backoff_config.max_tries} times): {repr(retry_error)}"
                        )
                        row.rollout_status = Status.rollout_error(str(retry_error))
                        return row
                else:
                    # Non-retryable exception - fail immediately
                    logging.error(f"❌ Rollout failed (non-retryable error encountered): {repr(e)}")
                    row.rollout_status = Status.rollout_error(str(e))
                    return row

        async def execute_row_with_backoff_and_log(
            task: asyncio.Task[EvaluationRow], row: EvaluationRow
        ) -> EvaluationRow:
            """Execute a single row task with backoff retry and logging."""
            result = await execute_row_with_backoff(task, row)
            # Log the row after execution completes (success or failure)
            config.logger.log(result)
            return result

        # Process all tasks concurrently with backoff retry
        retry_tasks = [
            asyncio.create_task(execute_row_with_backoff_and_log(task, fresh_dataset[i]))
            for i, task in enumerate(base_tasks)
        ]

        position = run_idx + 1  # Position 0 is reserved for main run bar, so shift up by 1
        with tqdm(
            total=len(retry_tasks),
            desc=f"  Run {run_idx + 1}",
            unit="rollout",
            file=sys.__stderr__,
            leave=False,
            position=position,
            dynamic_ncols=True,
            miniters=1,
            mininterval=0.1,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ) as rollout_pbar:
            # Yield results as they complete
            for task in asyncio.as_completed(retry_tasks):
                result = await task
                rollout_pbar.update(1)
                yield result

    finally:
        rollout_processor.cleanup()


def sanitize_filename(text: str) -> str:
    """Sanitize text for use in filenames by replacing special characters with dashes."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip())
    return safe[:120]


def extract_effort_tag(params: dict[str, Any]) -> str | None:
    """
    Extract effort tag from completion parameters for use in file naming.

    Args:
        params: Completion parameters dictionary

    Returns:
        Effort tag string if found, None otherwise
    """
    try:
        if not isinstance(params, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
            return None  # pyright: ignore[reportUnreachable]
        # Common locations
        if "extra_body" in params and isinstance(params["extra_body"], dict):
            eb = params["extra_body"]  # pyright: ignore[reportUnknownVariableType]
            if isinstance(eb.get("reasoning"), dict) and "effort" in eb["reasoning"]:  # pyright: ignore[reportUnknownMemberType]
                return str(eb["reasoning"]["effort"]).lower()  # pyright: ignore[reportUnknownArgumentType]
            if "reasoning_effort" in eb:
                return str(eb["reasoning_effort"]).lower()  # pyright: ignore[reportUnknownArgumentType]
        if "reasoning" in params and isinstance(params["reasoning"], dict) and "effort" in params["reasoning"]:
            return str(params["reasoning"]["effort"]).lower()  # pyright: ignore[reportUnknownArgumentType]
    except Exception:
        return None
    return None


def add_cost_metrics(row: EvaluationRow) -> None:
    """Calculate and add cost metrics for an EvaluationRow based on its usage data."""
    # Can't calculate cost without usage stats or model info
    if not row.execution_metadata.usage or not row.input_metadata.completion_params:
        row.execution_metadata.cost_metrics = CostMetrics(
            input_cost=0.0,
            output_cost=0.0,
            total_cost_dollar=0.0,
        )
        return

    model = row.input_metadata.completion_params.get("model", "unknown")
    provider = row.input_metadata.completion_params.get("provider")

    # Pydantic AI mapping to LiteLLM format
    # TODO: make more generic for other frameworks too.
    provider_mapping = {
        "fireworks": "fireworks_ai",
        "together": "together_ai",
        "openai": "",  # No prefix needed
        "azure": "azure",
        "deepseek": "deepseek",
        "openrouter": "openrouter",
        "grok": "grok",
        "github": "github",
        "heroku": "heroku",
    }

    if provider and provider in provider_mapping:
        litellm_prefix = provider_mapping[provider]
        model_id = f"{litellm_prefix}/{model}" if litellm_prefix else model
    else:
        model_id = model

    usage = row.execution_metadata.usage

    input_tokens = usage.prompt_tokens or 0
    output_tokens = usage.completion_tokens or 0

    # Try to calculate costs, but gracefully handle unknown models
    try:
        input_cost, output_cost = cost_per_token(
            model=model_id, prompt_tokens=input_tokens, completion_tokens=output_tokens
        )
        total_cost = input_cost + output_cost
    except Exception as e:
        # Model not in LiteLLM's database - set costs to 0 and continue
        logger.debug(f"Could not calculate cost for model '{model_id}': {e}")
        input_cost = 0.0
        output_cost = 0.0
        total_cost = 0.0

    # Set all cost metrics on the row
    row.execution_metadata.cost_metrics = CostMetrics(
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost_dollar=total_cost,
    )


def build_rollout_processor_config(
    rollout_processor: RolloutProcessor,
    model: str,
    semaphore: asyncio.Semaphore,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    steps: int = 30,
    mcp_config_path: str = "",
    server_script_path: Optional[str] = None,
    rollout_processor_kwargs: Optional[dict[str, Any]] = None,
    start_server: bool = True,
    server_mode: Optional[ServerMode] = None,
) -> RolloutProcessorConfig:
    """Build rollout processor config with appropriate parameters for different processor types.

    Args:
        rollout_processor: The rollout processor instance
        model: Model name/path for completion_params
        semaphore: Semaphore for concurrency control
        temperature: Temperature for completion_params
        max_tokens: Max tokens for completion_params
        steps: Number of rollout steps
        mcp_config_path: Path to MCP config file
        server_script_path: Path to server script (required for MCPGymRolloutProcessor)
        rollout_processor_kwargs: Additional kwargs to pass to rollout processor
        start_server: Whether to start server (for MCPGymRolloutProcessor)
        server_mode: Optional server lifecycle mode ("per_run" or "shared") for MCPGymRolloutProcessor

    Returns:
        RolloutProcessorConfig: Configured rollout processor config
    """
    rollout_processor_kwargs = rollout_processor_kwargs or {}

    completion_params = {"model": model, "temperature": temperature, "max_tokens": max_tokens}

    if isinstance(rollout_processor, MCPGymRolloutProcessor):
        base_kwargs = {**(rollout_processor_kwargs or {}), "start_server": start_server}
        if server_mode is not None and "server_mode" not in base_kwargs:
            base_kwargs["server_mode"] = server_mode

        return RolloutProcessorConfig(
            completion_params=completion_params,
            mcp_config_path=mcp_config_path,
            steps=steps,
            semaphore=semaphore,
            server_script_path=server_script_path,
            kwargs=base_kwargs,
        )

    # RemoteRolloutProcessor, SingleTurnRolloutProcessor, AgentRolloutProcessor, etc.
    return RolloutProcessorConfig(
        completion_params=completion_params,
        mcp_config_path=mcp_config_path,
        steps=steps,
        semaphore=semaphore,
        server_script_path=None,
        kwargs=rollout_processor_kwargs,
    )
