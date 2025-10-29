from collections import defaultdict
import json
import math
import os
import pathlib
import statistics
import sys
import time
from eval_protocol.dataset_logger.dataset_logger import DatasetLogger
from eval_protocol.models import CompletionParams, EvaluationRow, EvaluationThreshold, Status
from eval_protocol.pytest.handle_persist_flow import handle_persist_flow
from eval_protocol.pytest.types import EvaluationTestMode
from eval_protocol.pytest.evaluation_test_utils import (
    AggregationMethod,
    aggregate,
    extract_effort_tag,
    sanitize_filename,
)
from eval_protocol.stats.confidence_intervals import compute_fixed_set_mu_ci


def postprocess(
    all_results: list[list[EvaluationRow]],
    aggregation_method: AggregationMethod,
    threshold: EvaluationThreshold | None,
    active_logger: DatasetLogger,
    mode: EvaluationTestMode,
    completion_params: CompletionParams,
    test_func_name: str,
    num_runs: int,
    experiment_duration_seconds: float,
):
    valid_results = [
        [r for r in result if r.evaluation_result and r.evaluation_result.is_score_valid] for result in all_results
    ]

    if aggregation_method == "bootstrap":
        scores = [r.evaluation_result.score for result in valid_results for r in result if r.evaluation_result]
    else:
        scores = [
            sum(r.evaluation_result.score for r in result if r.evaluation_result) / len(result)
            for result in valid_results
            if result
        ]
    agg_score = aggregate(scores, aggregation_method)

    # Calculate raw score (total score / total rows, including invalid scores)
    all_scores = [r.evaluation_result.score for sublist in all_results for r in sublist if r.evaluation_result]
    raw_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

    # Compute 95% confidence interval for the fixed-set mean μ (by-question, using repeats)
    ci_low: float | None = None
    ci_high: float | None = None
    standard_error: float | None = None
    if aggregation_method == "mean":
        try:
            result_ci = compute_fixed_set_mu_ci([item for sublist in valid_results for item in sublist])
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
    for results in all_results:
        for result in results:
            if result.eval_metadata is not None:
                result.eval_metadata.passed = passed
            if result.evaluation_result is not None:
                if result.evaluation_result.agg_score is None:
                    result.evaluation_result.agg_score = agg_score
                if result.evaluation_result.standard_error is None:
                    result.evaluation_result.standard_error = standard_error
                if result.evaluation_result.is_score_valid is False:
                    if result.eval_metadata is not None:
                        if not result.eval_metadata.status or not result.eval_metadata.status.is_error():
                            result.eval_metadata.status = Status.score_invalid()
            result.execution_metadata.experiment_duration_seconds = experiment_duration_seconds
            active_logger.log(result)

    # Optional: print and/or persist a summary artifact for CI
    try:
        should_print = os.getenv("EP_PRINT_SUMMARY") == "1"
        summary_path = os.getenv("EP_SUMMARY_JSON")
        suite_name = test_func_name
        model_used = completion_params["model"]  # pyright: ignore[reportAny]
        total_rows = len([item for sublist in all_results for item in sublist])
        summary_obj = {
            "suite": suite_name,
            "model": model_used,
            "agg_score": float(agg_score),
            "num_runs": num_runs,
            "rows": total_rows,
        }
        if ci_low is not None and ci_high is not None and standard_error is not None:
            summary_obj["agg_ci_low"] = ci_low
            summary_obj["agg_ci_high"] = ci_high
            summary_obj["standard_error"] = standard_error

        # Aggregate per-metric mean and 95% CI when available
        metrics_summary: dict[str, dict[str, float]] = {}

        metric_scores: dict[str, list[float]] = defaultdict(list)
        for r in [item for sublist in all_results for item in sublist]:
            if r.evaluation_result and r.evaluation_result.metrics:
                for m_name, m_res in r.evaluation_result.metrics.items():
                    if getattr(m_res, "score", None) is not None:
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
            entry: dict[str, float] = {"mean": float(m_mean)}
            if m_low is not None and m_high is not None:
                entry["ci_low"] = float(m_low)
                entry["ci_high"] = float(m_high)
            metrics_summary[m_name] = entry
        if metrics_summary:
            summary_obj["metrics_agg"] = metrics_summary
        if should_print:
            if ci_low is not None and ci_high is not None and standard_error is not None:
                print(
                    f"EP Summary | suite={suite_name} model={model_used} runs={num_runs} rows={total_rows}\n"
                    f"  agg_score={summary_obj['agg_score']:.3f} (valid scores only)\n"
                    f"  raw_score={raw_score:.3f} (invalid scores as 0)\n"
                    f"  se={summary_obj['standard_error']:.3f} ci95=[{ci_low:.3f},{ci_high:.3f}]",
                    file=sys.__stderr__,
                )
            else:
                print(
                    f"EP Summary | suite={suite_name} model={model_used} runs={num_runs} rows={total_rows}\n"
                    f"  agg_score={summary_obj['agg_score']:.3f} (valid scores only)\n"
                    f"  raw_score={raw_score:.3f} (invalid scores as 0)",
                    file=sys.__stderr__,
                )
            # As per project convention, avoid printing per-metric CI lines to reduce noise
        if summary_path:
            if not isinstance(model_used, str):
                raise ValueError(f"Model used is not a string: {model_used}")
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

    handle_persist_flow(all_results, test_func_name)

    # Check threshold after logging
    if threshold is not None and not passed:
        assert agg_score >= threshold.success, f"Aggregated score {agg_score:.3f} below threshold {threshold.success}"
        if threshold.standard_error is not None and standard_error is not None:
            assert standard_error <= threshold.standard_error, (
                f"Standard error {standard_error:.3f} above threshold {threshold.standard_error}"
            )
