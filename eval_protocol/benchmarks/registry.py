"""
Benchmark registry and export decorator.

This module provides a lightweight registry for benchmarks and a decorator
`@export_benchmark(name)` that can be stacked with `@evaluation_test`.

It registers a runnable handle that executes the exact same evaluation pipeline
as the pytest flow by calling `run_evaluation_test_direct` with the parameters
captured from the decorated function.

Usage in a suite module (stack under @evaluation_test):

    from eval_protocol.benchmarks.registry import export_benchmark

    @export_benchmark("aime25_low")
    @evaluation_test(...)
    def test_aime_pointwise(row: EvaluationRow) -> EvaluationRow:
        ...

Programmatic run:

    from eval_protocol.benchmarks.registry import get_benchmark_runner
    get_benchmark_runner("aime25_low")(model="fireworks_ai/...", print_summary=True, out="artifacts/aime.json")
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional


# Global registry: name -> callable runner
_BENCHMARK_REGISTRY: Dict[str, Callable[..., Any]] = {}


def list_benchmarks() -> List[str]:
    return sorted(_BENCHMARK_REGISTRY.keys())


def get_benchmark_runner(name: str) -> Callable[..., Any]:
    try:
        return _BENCHMARK_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Benchmark '{name}' not found. Available: {list_benchmarks()}") from exc


def export_benchmark(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to export a benchmark test into the global registry.

    This expects to be stacked with `@evaluation_test`, so the decorated function
    should carry `__ep_config` and `__ep_original_test_func` attributes that the
    decorator can read to construct a direct runner.

    The registered runner supports a subset of convenient overrides and maps them
    to the same EP_* environment variables used by the pytest plugin to ensure
    identical summaries and JSON artifact behavior.
    """

    def _decorator(test_wrapper: Callable[..., Any]) -> Callable[..., Any]:
        # Pull through metadata attached by evaluation_test
        ep_config: Dict[str, Any] = getattr(test_wrapper, "__ep_config", {})
        original_test_func: Optional[Callable[..., Any]] = getattr(
            test_wrapper, "__ep_original_test_func", None
        )

        def _runner(
            *,
            model: Optional[str] = None,
            print_summary: bool = False,
            out: Optional[str] = None,
            reasoning_effort: Optional[str] = None,
            max_rows: Optional[int | str] = None,
            num_runs: Optional[int] = None,
            input_params_override: Optional[Dict[str, Any]] = None,
            max_concurrency: Optional[int] = None,
        ) -> Any:
            # Map convenience flags to EP_* env used by the pytest flow
            if print_summary:
                os.environ["EP_PRINT_SUMMARY"] = "1"
            if out:
                os.environ["EP_SUMMARY_JSON"] = out
            # Merge reasoning effort and arbitrary overrides into EP_INPUT_PARAMS_JSON
            merged: Dict[str, Any] = {}
            if reasoning_effort:
                # Fireworks OpenAI-compatible endpoint expects extra_body.reasoning_effort, not nested reasoning dict
                merged.setdefault("extra_body", {})["reasoning_effort"] = str(reasoning_effort)
            if input_params_override:
                def _deep_update(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
                    for k, v in over.items():
                        if isinstance(v, dict) and isinstance(base.get(k), dict):
                            _deep_update(base[k], v)
                        else:
                            base[k] = v
                    return base
                merged = _deep_update(merged, dict(input_params_override))
            if merged:
                os.environ["EP_INPUT_PARAMS_JSON"] = json.dumps(merged)

            if max_rows is not None:
                if isinstance(max_rows, str) and max_rows.strip().lower() == "all":
                    os.environ["EP_MAX_DATASET_ROWS"] = "None"
                else:
                    os.environ["EP_MAX_DATASET_ROWS"] = str(max_rows)

            # Build effective parameters, preferring overrides
            models: List[str] = ep_config.get("model") or []
            model_to_use = model or (models[0] if models else None)
            if not model_to_use:
                raise ValueError(
                    f"No model provided and none captured from evaluation_test for benchmark '{name}'"
                )

            input_messages = ep_config.get("input_messages")
            input_dataset = ep_config.get("input_dataset")
            dataset_adapter = ep_config.get("dataset_adapter")
            rollout_input_params_list = ep_config.get("rollout_input_params")
            rollout_processor = ep_config.get("rollout_processor")
            aggregation_method = ep_config.get("aggregation_method")
            threshold = ep_config.get("threshold_of_success")
            default_num_runs = ep_config.get("num_runs")
            max_dataset_rows = ep_config.get("max_dataset_rows")
            mcp_config_path = ep_config.get("mcp_config_path")
            max_concurrent_rollouts = ep_config.get("max_concurrent_rollouts")
            if max_concurrency is not None:
                max_concurrent_rollouts = int(max_concurrency)
            server_script_path = ep_config.get("server_script_path")
            steps = ep_config.get("steps")
            mode = ep_config.get("mode")
            combine_datasets = ep_config.get("combine_datasets")

            # Choose the first rollout param set by default
            rollout_params = None
            if isinstance(rollout_input_params_list, list) and rollout_input_params_list:
                rollout_params = rollout_input_params_list[0]

            # Import runner lazily to avoid hard import dependencies and circulars
            import importlib

            _mod = importlib.import_module("eval_protocol.pytest.evaluation_test")
            run_evaluation_test_direct = getattr(_mod, "run_evaluation_test_direct")

            return run_evaluation_test_direct(
                test_func=original_test_func or test_wrapper,
                model=model_to_use,
                input_messages=input_messages,
                input_dataset=input_dataset,
                dataset_adapter=dataset_adapter,
                rollout_input_params=rollout_params,
                rollout_processor=rollout_processor,
                aggregation_method=aggregation_method,
                threshold_of_success=threshold,
                num_runs=(num_runs if num_runs is not None else default_num_runs),
                max_dataset_rows=max_dataset_rows,
                mcp_config_path=mcp_config_path,
                max_concurrent_rollouts=max_concurrent_rollouts,
                server_script_path=server_script_path,
                steps=steps,
                mode=mode,
            )

        # Register runner
        if name in _BENCHMARK_REGISTRY:
            # Overwrite with latest definition
            _BENCHMARK_REGISTRY[name] = _runner
        else:
            _BENCHMARK_REGISTRY[name] = _runner

        return test_wrapper

    return _decorator


