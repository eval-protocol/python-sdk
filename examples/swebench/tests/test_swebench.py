from typing import List
import os
import pytest
import requests
import yaml
from eval_protocol.data_loader.dynamic_data_loader import DynamicDataLoader
from eval_protocol.models import EvaluationRow, Message, EvaluateResult, MetricResult
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.remote_rollout_processor import RemoteRolloutProcessor
from eval_protocol.types.remote_rollout_processor import DataLoaderConfig
from eval_protocol.quickstart.utils import filter_longest_conversation

# Reuse the converter used by the built-in adapter
from eval_protocol.adapters.fireworks_tracing import convert_trace_dict_to_evaluation_row
import conftest


MODEL_ID = conftest.MODEL_ID_OPT
if not MODEL_ID:
    raise RuntimeError("--model-id is required. Example: --model-id 'fireworks_ai/accounts/.../models/<name>'")
CLI_CONCURRENCY = conftest.CONCURRENCY_OPT
CLI_MODEL_KWARGS = conftest.MODEL_KWARGS_OPT

# Build completion_params once (used by decorator)
COMPLETION_PARAMS = {"model": MODEL_ID}
if CLI_MODEL_KWARGS:
    COMPLETION_PARAMS["model_kwargs"] = CLI_MODEL_KWARGS


def fetch_traces_with_auth(config: DataLoaderConfig) -> List[EvaluationRow]:
    """
    Fetch traces directly from the Fireworks tracing proxy with Authorization header
    and convert them into EvaluationRows using the same converter as the adapter.
    """
    base_url = (config.model_base_url or "https://tracing.fireworks.ai").rstrip("/")
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        return []

    url = f"{base_url}/v1/traces"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {
        "tags": [f"rollout_id:{config.rollout_id}"],
        "max_retries": 5,
        "sleep_between_gets": 0.1,
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=300)
        print(f"[fetch_traces] status={resp.status_code} url={resp.url}")  # debug
        resp.raise_for_status()
        body = resp.json() or {}
        traces = body.get("traces", [])
        print(f"[fetch_traces] traces_found={len(traces)}")
    except Exception as e:
        print(f"[fetch_traces] error={e}")
        return []

    rows: List[EvaluationRow] = []
    for tr in traces:
        row = convert_trace_dict_to_evaluation_row(tr, include_tool_calls=True, span_name=None)
        if row:
            rows.append(row)
    return rows


def _merge_rows_into_one(rows: List[EvaluationRow]) -> List[EvaluationRow]:
    if not rows:
        return []
    # Use the first row as the base; merge messages from all rows
    base = rows[0]
    seen = set()
    merged_msgs: List[Message] = []
    for r in rows:
        for m in r.messages or []:
            # Dedup by role+name+content+tool_calls signature
            tool_sig = None
            if getattr(m, "tool_calls", None):
                tool_sig = tuple(
                    (tc.get("id"), tc.get("type"), (tc.get("function") or {}).get("name")) for tc in m.tool_calls
                )
            key = (m.role, getattr(m, "name", None), m.content, tool_sig)
            if key in seen:
                continue
            seen.add(key)
            merged_msgs.append(m)
    base.messages = merged_msgs
    return [base]


def fireworks_output_data_loader(config: DataLoaderConfig) -> DynamicDataLoader:
    return DynamicDataLoader(
        generators=[lambda: fetch_traces_with_auth(config)],
        preprocess_fn=_merge_rows_into_one,  # merge all tool/LLM traces into one row
    )


def rows_from_instance_ids(ids: list[str]) -> List[EvaluationRow]:
    out = []
    for idx, iid in enumerate(ids):
        out.append(
            EvaluationRow(
                messages=[Message(role="user", content=f"Run SWE-bench instance {iid}")],
                input_metadata={
                    "row_id": str(idx),  # ← use instance_id here
                    "instance_id": iid,  # ← explicit for debugging
                    "instance_index": str(idx),  # ← optional: keep index
                    "completion_params": {"model": MODEL_ID},
                },
            )
        )
    return out


def rows_from_indices(count: int) -> List[EvaluationRow]:
    out: List[EvaluationRow] = []
    for idx in range(count):
        metadata = {
            "row_id": str(idx),
            "instance_index": str(idx),
        }
        # Add model_kwargs to metadata so server can read from req.metadata
        if CLI_MODEL_KWARGS:
            metadata["model_kwargs"] = CLI_MODEL_KWARGS

        out.append(
            EvaluationRow(
                messages=[Message(role="user", content=f"Run SWE-bench index {idx}")],
                input_metadata=metadata,
            )
        )
    return out


def rows() -> List[EvaluationRow]:
    # Generate 10 rows by index; server maps index -> dataset instance via --slice
    return rows_from_indices(10)


# -------------------- Harness result attachment (UI pass/fail) --------------------
import json
from pathlib import Path


def _safe_model_id(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "-")


def attach_eval_result(row: EvaluationRow, model_id: str) -> EvaluationRow:
    """Attach evaluation result by reading harness report or exit status."""
    import logging

    logger = logging.getLogger(__name__)

    # Get row_id and instance_id
    try:
        row_id = str(row.input_metadata.row_id)  # ← use attribute, not .get()
    except Exception as e:
        logger.warning(f"Could not get row_id: {e}")
        return row

    row_dir = Path.cwd() / f"row_{row_id}"
    logger.info(f"[Row {row_id}] Looking for results in {row_dir}")

    # Find instance_id from preds.json
    preds_path = row_dir / "preds.json"
    instance_id = None
    if preds_path.exists():
        try:
            preds = json.loads(preds_path.read_text())
            instance_id = next(iter(preds.keys()), None)
            logger.info(f"[Row {row_id}] Found instance_id: {instance_id}")
        except Exception as e:
            logger.warning(f"[Row {row_id}] Could not read preds.json: {e}")

    if not instance_id:
        logger.warning(f"[Row {row_id}] No instance_id found, skipping eval result")
        return row

    resolved: bool | None = None
    reason_text: str | None = None

    # 1. Try to read from report.json (harness ran tests)
    safe_model = _safe_model_id(model_id)
    report_path = row_dir / "logs" / "run_evaluation" / "eval-run" / safe_model / instance_id / "report.json"

    if report_path.exists():
        logger.info(f"[Row {row_id}] Found report.json at {report_path}")
        try:
            report_data = json.loads(report_path.read_text())
            instance_data = report_data.get(instance_id, {})
            resolved = bool(instance_data.get("resolved", False))
            reason_text = f"harness_resolved={resolved}"
            logger.info(f"[Row {row_id}] Report says resolved={resolved}")
        except Exception as e:
            logger.error(f"[Row {row_id}] Failed to parse report.json: {e}")
    else:
        logger.info(f"[Row {row_id}] No report.json found at {report_path}")

    # 2. If no report, check exit status YAML (agent didn't produce a patch)
    if resolved is None:
        exit_status_files = sorted(row_dir.glob("exit_statuses_*.yaml"))
        if exit_status_files:
            exit_file = exit_status_files[-1]
            logger.info(f"[Row {row_id}] Reading exit status from {exit_file.name}")
            try:
                status_doc = yaml.safe_load(exit_file.read_text()) or {}
                by_status = status_doc.get("instances_by_exit_status", {})
                for status_name, ids in by_status.items():
                    if instance_id in (ids or []):
                        resolved = False
                        reason_text = f"exit_status={status_name}"
                        logger.info(f"[Row {row_id}] Exit status: {status_name}")
                        break
            except Exception as e:
                logger.error(f"[Row {row_id}] Failed to parse exit status: {e}")
        else:
            logger.warning(f"[Row {row_id}] No exit status YAML found")

    # 3. Attach result if we found anything
    if resolved is not None:
        logger.info(f"[Row {row_id}] Final: resolved={resolved}, reason={reason_text}")
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
    else:
        logger.warning(f"[Row {row_id}] Could not determine resolved status")

    return row


@evaluation_test(
    data_loaders=DynamicDataLoader(
        generators=[rows],
    ),
    rollout_processor=RemoteRolloutProcessor(
        remote_base_url="http://127.0.0.1:3000",
        model_base_url="https://tracing.fireworks.ai",
        timeout_seconds=1800,
        output_data_loader=fireworks_output_data_loader,
    ),
    completion_params=[COMPLETION_PARAMS],
    max_concurrent_rollouts=(CLI_CONCURRENCY or 2),
)
async def test_swebench_remote(row: EvaluationRow) -> EvaluationRow:
    assert len(row.messages) >= 1
    row = attach_eval_result(row, MODEL_ID)
    return row
