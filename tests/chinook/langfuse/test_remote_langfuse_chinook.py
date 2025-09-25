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
from eval_protocol.adapters.langfuse import create_langfuse_adapter
from eval_protocol.quickstart.utils import filter_longest_conversation

ROLLOUT_IDS = set()


@pytest.fixture(autouse=True)
def check_rollout_coverage():
    """Ensure we processed all expected rollout_ids"""
    global ROLLOUT_IDS
    ROLLOUT_IDS.clear()
    yield

    assert len(ROLLOUT_IDS) == 3, f"Expected to see {ROLLOUT_IDS} rollout_ids, but only saw {ROLLOUT_IDS}"


def fetch_langfuse_traces(rollout_id: str) -> List[EvaluationRow]:
    global ROLLOUT_IDS  # Track all rollout_ids we've seen
    ROLLOUT_IDS.add(rollout_id)

    adapter = create_langfuse_adapter()
    return adapter.get_evaluation_rows(tags=[f"rollout_id:{rollout_id}"])


def langfuse_output_data_loader(rollout_id: str) -> DynamicDataLoader:
    return DynamicDataLoader(
        generators=[lambda: fetch_langfuse_traces(rollout_id)], preprocess_fn=filter_longest_conversation
    )


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
    row = EvaluationRow(messages=[Message(role="user", content="What is the capital of France?")])
    return [row, row, row]


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Only run this test locally (skipped in CI)")
@pytest.mark.parametrize("completion_params", [{"model": "gpt-4o"}])
@evaluation_test(
    data_loaders=DynamicDataLoader(
        generators=[remote_langfuse_data_generator],
    ),
    rollout_processor=RemoteRolloutProcessor(
        remote_base_url="http://127.0.0.1:7077",
        num_turns=2,
        timeout_seconds=30,
        output_data_loader=langfuse_output_data_loader,
    ),
)
async def test_remote_rollout_and_fetch_langfuse(row: EvaluationRow) -> EvaluationRow:
    """
    End-to-end test:
    - remote server started at import time
    - trigger remote rollout via RemoteRolloutProcessor (calls init/status)
    - fetch traces from Langfuse filtered by metadata via output_data_loader; FAIL if none found
    """
    assert row.messages[0].content == "What is the capital of France?", "Row should have correct message content"
    assert len(row.messages) > 1, "Row should have a response. If this fails, we fellback to the original row."

    assert row.execution_metadata.rollout_id in ROLLOUT_IDS, (
        f"Row rollout_id {row.execution_metadata.rollout_id} should be in tracked rollout_ids: {ROLLOUT_IDS}"
    )

    return row
