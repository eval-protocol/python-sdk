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
        original_test_func: Optional[Callable[..., Any]] = getattr(test_wrapper, "__ep_original_test_func", None)

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
                raise ValueError(f"No model provided and none captured from evaluation_test for benchmark '{name}'")

            input_messages = ep_config.get("input_messages")
            input_dataset = ep_config.get("input_dataset")
            dataset_adapter = ep_config.get("dataset_adapter")
            rollout_input_params_list = ep_config.get("rollout_input_params")
            rollout_processor = ep_config.get("rollout_processor")
            rollout_processor_kwargs = ep_config.get("rollout_processor_kwargs")
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
            # combine_datasets captured but not used here

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
                rollout_processor_kwargs=rollout_processor_kwargs,
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


def register_composite_benchmark(name: str, children: List[str]) -> None:
    """
    Register a composite benchmark that runs multiple exported benchmarks and aggregates results.

    The composite runner forwards common overrides to each child benchmark and aggregates
    a combined score as a rows-weighted mean of each child's aggregated score.

    Args:
        name: Name of the composite benchmark to register.
        children: List of child benchmark names previously registered via export_benchmark.
    """

    def _composite_runner(
        *,
        model: Optional[str] = None,
        print_summary: bool = False,
        out: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        max_rows: Optional[int | str] = None,
        num_runs: Optional[int] = None,
        input_params_override: Optional[Dict[str, Any]] = None,
        max_concurrency: Optional[int] = None,
    ) -> Dict[str, Any]:
        # Resolve child runners at call-time to ensure all suites are imported
        # Local import avoided to prevent circular import at module import time
        _get_benchmark_runner = get_benchmark_runner
        import pathlib as _pathlib
        import time as _time
        _json = json

        child_summaries: List[Dict[str, Any]] = []
        total_rows = 0
        weighted_sum = 0.0
        # For per-metric aggregation across children
        metric_weighted_sums: Dict[str, float] = {}
        metric_total_rows: Dict[str, int] = {}
        combined_rows: List[Any] = []

        # If 'out' is a file path, also compute a directory for child artifacts
        child_out_dir: Optional[str] = None
        if out:
            p = _pathlib.Path(out)
            if p.suffix.lower() == ".json" and not str(out).endswith("/"):
                # Use parent directory for child artifacts
                child_out_dir = str(p.parent)
            else:
                child_out_dir = out

        for child_name in children:
            runner = _get_benchmark_runner(child_name)
            result = runner(
                model=model,
                print_summary=print_summary,
                out=child_out_dir,
                reasoning_effort=reasoning_effort,
                max_rows=max_rows,
                num_runs=num_runs,
                input_params_override=input_params_override,
                max_concurrency=max_concurrency,
            )
            summary = (result or {}).get("summary") if isinstance(result, dict) else None
            if not summary:
                continue
            # Gather underlying rows to recompute CI across children
            try:
                rows_obj = result.get("results") if isinstance(result, dict) else None
                if isinstance(rows_obj, list):
                    combined_rows.extend(rows_obj)
            except Exception:
                pass
            child_summaries.append(summary)
            rows = int(summary.get("rows", 0) or 0)
            agg = summary.get("agg_score")
            if isinstance(agg, (int, float)) and rows > 0:
                total_rows += rows
                weighted_sum += float(agg) * rows
            # Combine per-metric means if available
            metrics_agg = summary.get("metrics_agg") or {}
            if isinstance(metrics_agg, dict):
                for m_name, m_vals in metrics_agg.items():
                    m_mean = m_vals.get("mean")
                    if isinstance(m_mean, (int, float)) and rows > 0:
                        metric_weighted_sums[m_name] = metric_weighted_sums.get(m_name, 0.0) + float(m_mean) * rows
                        metric_total_rows[m_name] = metric_total_rows.get(m_name, 0) + rows

        combined_agg = (weighted_sum / total_rows) if total_rows > 0 else None
        # Compute 95% CI for combined rows if available
        ci_low: Optional[float] = None
        ci_high: Optional[float] = None
        if combined_rows:
            try:
                from eval_protocol.stats.confidence_intervals import compute_fixed_set_mu_ci as _compute_ci

                r = _compute_ci(combined_rows)
                if r and len(r) >= 3 and r[1] is not None and r[2] is not None:
                    ci_low = float(r[1])
                    ci_high = float(r[2])
            except Exception:
                ci_low = None
                ci_high = None
        combined_metrics: Dict[str, Dict[str, float]] = {}
        for m_name, wsum in metric_weighted_sums.items():
            denom = metric_total_rows.get(m_name, 0)
            if denom > 0:
                combined_metrics[m_name] = {"mean": float(wsum / denom)}
        combined = {
            "suite": name,
            "model": model,
            "agg_score": float(combined_agg) if combined_agg is not None else None,
            "rows": total_rows,
            "children": child_summaries,
            "num_runs": num_runs,
            **({"metrics_agg": combined_metrics} if combined_metrics else {}),
            **({"agg_ci_low": ci_low, "agg_ci_high": ci_high} if (ci_low is not None and ci_high is not None) else {}),
        }

        # Optional print and persist
        # Respect either function arg or EP_PRINT_SUMMARY env
        _should_print = print_summary or (os.getenv("EP_PRINT_SUMMARY") == "1")
        if _should_print:
            try:
                if combined_agg is not None:
                    if ci_low is not None and ci_high is not None:
                        print(
                            f"EP Summary | suite={name} model={model} agg={combined['agg_score']:.3f} ci95=[{ci_low:.3f},{ci_high:.3f}] rows={total_rows}"
                        )
                    else:
                        print(
                            f"EP Summary | suite={name} model={model} agg={combined['agg_score']:.3f} rows={total_rows}"
                        )
                else:
                    print(
                        f"EP Summary | suite={name} model={model} agg=None rows={total_rows}"
                    )
            except Exception:
                pass

        if out:
            out_path = _pathlib.Path(out)
            if out_path.suffix.lower() == ".json" and not str(out).endswith("/"):
                # Write to the specified file
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    _json.dump({**combined, "timestamp": int(_time.time())}, f)
            else:
                # Treat as directory
                dir_path = out_path
                dir_path.mkdir(parents=True, exist_ok=True)
                safe_name = name.replace("/", "__")
                file_path = dir_path / f"{safe_name}__composite.json"
                with open(file_path, "w", encoding="utf-8") as f:
                    _json.dump({**combined, "timestamp": int(_time.time())}, f)

        return {"summary": combined}

    # Register (overwrite if exists)
    _BENCHMARK_REGISTRY[name] = _composite_runner
