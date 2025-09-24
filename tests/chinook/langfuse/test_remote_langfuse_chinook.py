import os
import multiprocessing
import time
from datetime import datetime, timedelta
from typing import List
import atexit

import pytest
import requests

from eval_protocol.data_loader.dynamic_data_loader import DynamicDataLoader
from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.remote_rollout_processor import RemoteRolloutProcessor


def _start_remote_server():
    # Starts FastAPI server defined in remote_server.py using absolute import
    import importlib

    os.environ.setdefault("REMOTE_SERVER_HOST", "127.0.0.1")
    os.environ.setdefault("REMOTE_SERVER_PORT", "7077")
    mod = importlib.import_module("tests.chinook.langfuse.remote_server")
    mod.main()


def _ensure_server_running():
    host = os.getenv("REMOTE_SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("REMOTE_SERVER_PORT", "7077"))
    base_url = f"http://{host}:{port}"

    def _is_up() -> bool:
        try:
            r = requests.get(f"{base_url}/status", params={"rollout_id": "ping"}, timeout=1.0)
            return r.status_code in (200, 404)
        except Exception:
            return False

    if _is_up():
        return None

    # Launch in a background process
    proc = multiprocessing.Process(target=_start_remote_server, daemon=True)
    proc.start()

    # Poll for readiness up to 10s
    deadline = time.time() + 10
    while time.time() < deadline:
        if _is_up():
            break
        time.sleep(0.5)
    return proc


def remote_langfuse_data_generator() -> List[EvaluationRow]:
    # Ensure server is running BEFORE rollouts start (evaluation_test triggers rollouts before test body)
    _SERVER_PROC = _ensure_server_running()
    atexit.register(lambda: (_SERVER_PROC and _SERVER_PROC.is_alive() and _SERVER_PROC.terminate()))

    # Minimal single-user-turn message to trigger a response
    row = EvaluationRow(messages=[Message(role="user", content="Hello there! Please say hi back.")])
    return [row]


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Only run this test locally (skipped in CI)")
@pytest.mark.asyncio
@evaluation_test(
    data_loaders=DynamicDataLoader(
        generators=[remote_langfuse_data_generator],
    ),
    completion_params=[{"model": "fireworks_ai/accounts/fireworks/models/kimi-k2-instruct"}],
    rollout_processor=RemoteRolloutProcessor(
        remote_base_url="http://127.0.0.1:7077",
        num_turns=2,
        timeout_seconds=30,
    ),
    mode="pointwise",
)
async def test_remote_rollout_and_fetch_langfuse(row: EvaluationRow) -> EvaluationRow:
    """
    End-to-end test:
    - remote server started at import time
    - trigger remote rollout via RemoteRolloutProcessor (calls init/status)
    - fetch traces from Langfuse filtered by metadata; FAIL if none found
    """
    # Debug print IDs used for filtering
    print(
        "[Remote-E2E] IDs:",
        {
            "invocation_id": row.execution_metadata.invocation_id,
            "experiment_id": row.execution_metadata.experiment_id,
            "rollout_id": row.execution_metadata.rollout_id,
            "run_id": row.execution_metadata.run_id,
        },
    )

    # Attempt retrieval via adapter
    try:
        from eval_protocol.adapters.langfuse import create_langfuse_adapter

        adapter = create_langfuse_adapter()

        # Preferred: observations-level requester_metadata contains invocation_id (proxy annotates per-request)
        contains_val = row.execution_metadata.invocation_id or ""
        rows = []
        if contains_val:
            # Retry loop to allow ingestion/flush
            deadline = time.time() + 90
            while time.time() < deadline and not rows:
                rows = adapter.get_evaluation_rows(
                    limit=10,
                    from_timestamp=datetime.now() - timedelta(hours=2),
                    to_timestamp=datetime.now(),
                    include_tool_calls=False,
                    requester_metadata_contains=contains_val,
                )
                if rows:
                    break
                time.sleep(3)
        else:
            print("[Remote-E2E] Missing invocation_id; skipping observations filter")

        # If still empty, dump recent trace metadata for debugging
        if not rows:
            try:
                from langfuse import get_client  # pyright: ignore[reportPrivateImportUsage]

                lf = get_client()
                recent = lf.api.trace.list(limit=5, order_by="timestamp.desc")
                print("[Remote-E2E] Recent trace metadata dump (id, metadata, requester_metadata, tags):")
                if recent and getattr(recent, "data", None):
                    for t in recent.data:
                        try:
                            full = lf.api.trace.get(t.id)
                            print(
                                {
                                    "id": full.id,
                                    "metadata": getattr(full, "metadata", None),
                                    "requester_metadata": getattr(full, "requester_metadata", None),
                                    "tags": getattr(full, "tags", None),
                                }
                            )
                        except Exception as e:
                            print("[Remote-E2E] Failed to get trace details:", e)
                else:
                    print("[Remote-E2E] No recent traces found via list().")
            except Exception as e:
                print("[Remote-E2E] Langfuse debug fetch failed:", e)

        assert rows and len(rows) > 0, (
            "No Langfuse traces matched the metadata. Ensure the LiteLLM proxy is configured to forward "
            "Langfuse telemetry and that LANGFUSE_* env vars are set."
        )

        # Minimal sanity: rows contain session_data.langfuse_trace_id
        assert any((r.input_metadata.session_data or {}).get("langfuse_trace_id") for r in rows), (
            "Expected langfuse_trace_id in session_data for at least one row"
        )

    except ImportError:
        pytest.fail("Langfuse SDK not installed; cannot verify traces.")

    return row
