# AUTO SERVER STARTUP: Server is automatically started and stopped by the test

import os
import subprocess
import socket
import time
from typing import List

import pytest
import requests

from eval_protocol.data_loader.dynamic_data_loader import DynamicDataLoader
from eval_protocol.models import EvaluationRow, Message, EvaluateResult
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.remote_rollout_processor import RemoteRolloutProcessor
from eval_protocol.adapters.fireworks_tracing import FireworksTracingAdapter
from eval_protocol.utils.evaluation_row_utils import filter_longest_conversation
from eval_protocol.types.remote_rollout_processor import DataLoaderConfig

ROLLOUT_IDS = set()


def find_available_port() -> int:
    """Find an available port on localhost"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    return port


SERVER_PORT = find_available_port()


def wait_for_server_to_startup(timeout: int = 120):
    start_time = time.time()
    while True:
        try:
            requests.get(f"http://127.0.0.1:{SERVER_PORT}")
            break
        except requests.exceptions.RequestException:
            time.sleep(1)
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Server did not start within {timeout} seconds")


@pytest.fixture(autouse=True)
def setup_remote_server():
    """Start the remote server"""
    # kill all Python processes matching "python -m tests.remote_server.remote_server"
    subprocess.run(["pkill", "-f", "python -m tests.remote_server.remote_server"], capture_output=True)

    host = "127.0.0.1"
    process = subprocess.Popen(
        [
            "python",
            "-m",
            "tests.remote_server.remote_server",
            "--host",
            host,
            "--port",
            str(SERVER_PORT),
        ]
    )
    # wait for the server to startup by polling
    wait_for_server_to_startup()
    yield
    process.terminate()
    process.wait()


@pytest.fixture(autouse=True)
def check_rollout_coverage():
    """Ensure we processed all expected rollout_ids"""
    global ROLLOUT_IDS
    ROLLOUT_IDS.clear()
    yield

    assert len(ROLLOUT_IDS) == 3, f"Expected to see 3 rollout_ids, but only saw {ROLLOUT_IDS}"


def fetch_fireworks_traces(config: DataLoaderConfig) -> List[EvaluationRow]:
    global ROLLOUT_IDS  # Track all rollout_ids we've seen
    ROLLOUT_IDS.add(config.rollout_id)

    base_url = config.model_base_url or "https://tracing.fireworks.ai"
    adapter = FireworksTracingAdapter(base_url=base_url)
    return adapter.get_evaluation_rows(tags=[f"rollout_id:{config.rollout_id}"], max_retries=7)


def fireworks_output_data_loader(config: DataLoaderConfig) -> DynamicDataLoader:
    return DynamicDataLoader(
        generators=[lambda: fetch_fireworks_traces(config)], preprocess_fn=filter_longest_conversation
    )


def rows() -> List[EvaluationRow]:
    """Generate local rows with rich input_metadata to verify it survives remote traces."""
    base_dataset_info = {
        "requirements": ["Answer with the capital city of France."],
        "total_requirements": 1,
        "original_prompt": "What is the capital of France?",
    }

    row = EvaluationRow(messages=[Message(role="user", content="What is the capital of France?")])
    row.input_metadata.dataset_info = dict(base_dataset_info)

    return [row, row, row]


@pytest.mark.parametrize(
    "completion_params",
    [{"model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b", "temperature": 0.5}],
)
@evaluation_test(
    data_loaders=DynamicDataLoader(
        generators=[rows],
    ),
    rollout_processor=RemoteRolloutProcessor(
        remote_base_url=f"http://127.0.0.1:{SERVER_PORT}",
        timeout_seconds=180,
        output_data_loader=fireworks_output_data_loader,
    ),
)
async def test_remote_rollout_and_fetch_fireworks(row: EvaluationRow) -> EvaluationRow:
    """
    End-to-end test:
    - AUTO SERVER STARTUP: Server is automatically started and stopped by the test
    - trigger remote rollout via RemoteRolloutProcessor (calls init/status)
    - fetch traces from Langfuse via Fireworks tracing proxy filtered by metadata via output_data_loader; FAIL if none found
    """
    row.evaluation_result = EvaluateResult(score=0.0, reason="Dummy evaluation result")

    assert row.messages[0].content == "What is the capital of France?", "Row should have correct message content"
    assert len(row.messages) > 1, "Row should have a response. If this fails, we fellback to the original row."

    assert row.execution_metadata.rollout_id in ROLLOUT_IDS, (
        f"Row rollout_id {row.execution_metadata.rollout_id} should be in tracked rollout_ids: {ROLLOUT_IDS}"
    )
    assert row.input_metadata.completion_params["model"] == "fireworks_ai/accounts/fireworks/models/gpt-oss-120b"
    assert row.input_metadata.completion_params["temperature"] == 0.5, "Row should have temperature at top level"

    assert row.input_metadata.row_id is not None

    assert row.input_metadata.dataset_info is not None
    assert row.input_metadata.dataset_info["requirements"] == ["Answer with the capital city of France."]
    assert row.input_metadata.dataset_info["total_requirements"] == 1
    assert row.input_metadata.dataset_info["original_prompt"] == "What is the capital of France?"

    assert "data_loader_type" in row.input_metadata.dataset_info
    assert "data_loader_num_rows" in row.input_metadata.dataset_info

    return row
