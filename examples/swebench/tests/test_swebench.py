from typing import List
import yaml
from eval_protocol.data_loader.dynamic_data_loader import DynamicDataLoader
from eval_protocol.models import EvaluationRow, Message, EvaluateResult, MetricResult
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.remote_rollout_processor import RemoteRolloutProcessor
from eval_protocol.pytest.tracing_utils import default_fireworks_output_data_loader
import json
from pathlib import Path


def rows_from_indices(count: int) -> List[EvaluationRow]:
    out: List[EvaluationRow] = []
    for idx in range(count):
        out.append(
            EvaluationRow(
                messages=[],
                input_metadata={
                    "row_id": str(idx),
                    "instance_index": str(idx),
                },
            )
        )
    return out


def rows() -> List[EvaluationRow]:
    # Generate 10 rows by index; server maps index -> dataset instance via --slice
    return rows_from_indices(2)


# -------------------- Harness result attachment (UI pass/fail) --------------------
@evaluation_test(
    data_loaders=DynamicDataLoader(
        generators=[rows],
    ),
    rollout_processor=RemoteRolloutProcessor(
        remote_base_url="http://127.0.0.1:3000",
        model_base_url="https://tracing.fireworks.ai",
        timeout_seconds=1800,
        output_data_loader=default_fireworks_output_data_loader,
    ),
    completion_params=[{"model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b"}],
    max_concurrent_rollouts=3,
)
async def test_swebench_remote(row: EvaluationRow) -> EvaluationRow:
    """Evaluate SWE-bench instance by reading harness report or exit status."""

    # Get row_id
    try:
        row_id = str(row.input_metadata.row_id)
    except Exception:
        return row

    row_dir = Path.cwd() / f"row_{row_id}"

    # Find instance_id from preds.json
    preds_path = row_dir / "preds.json"
    instance_id = None
    if preds_path.exists():
        try:
            preds = json.loads(preds_path.read_text())
            instance_id = next(iter(preds.keys()), None)
        except Exception:
            pass

    if not instance_id:
        return row

    resolved: bool | None = None
    reason_text: str | None = None

    # Get model from completion_params and convert to safe directory name (matching SWE-bench convention)
    model_id = row.input_metadata.completion_params.get("model") if row.input_metadata.completion_params else None
    if not model_id:
        return row
    safe_model = model_id.replace("/", "__").replace(":", "-")

    # Read from report.json (harness ran tests)
    report_path = row_dir / "logs" / "run_evaluation" / "eval-run" / safe_model / instance_id / "report.json"
    if report_path.exists():
        try:
            report_data = json.loads(report_path.read_text())
            resolved = bool(report_data.get(instance_id, {}).get("resolved", False))
            reason_text = f"harness_resolved={resolved}"
        except Exception:
            pass

    # If no report, check exit status YAML
    if resolved is None:
        exit_status_files = sorted(row_dir.glob("exit_statuses_*.yaml"))
        if exit_status_files:
            try:
                status_doc = yaml.safe_load(exit_status_files[-1].read_text()) or {}
                by_status = status_doc.get("instances_by_exit_status", {})
                for status_name, ids in by_status.items():
                    if instance_id in (ids or []):
                        resolved = False
                        reason_text = f"exit_status={status_name}"
                        break
            except Exception:
                pass

    # Attach result
    if resolved is not None:
        row.evaluation_result = EvaluateResult(
            score=1.0 if resolved else 0.0,
            reason=reason_text or f"resolved={resolved}",
            is_score_valid=True,
            metrics={
                "resolved": MetricResult(
                    score=1.0 if resolved else 0.0,
                    is_score_valid=True,
                    reason=reason_text or f"resolved={resolved}",
                    value=int(resolved),
                )
            },
        )

    return row
