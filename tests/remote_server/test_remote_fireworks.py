# AUTO SERVER STARTUP: Server is automatically started and stopped by the test

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
import eval_protocol.pytest.remote_rollout_processor as remote_rollout_processor_module
from eval_protocol.types.remote_rollout_processor import DataLoaderConfig


ROLLOUT_IDS = set()


@pytest.fixture(autouse=True)
def check_rollout_coverage(monkeypatch):
    """
    Ensure we attempted to fetch remote traces for each rollout.

    This wraps the built-in default_fireworks_output_data_loader (without making it configurable)
    and tracks rollout_ids passed through its DataLoaderConfig.
    """
    global ROLLOUT_IDS
    ROLLOUT_IDS.clear()

    original_loader = remote_rollout_processor_module.default_fireworks_output_data_loader

    def wrapped_loader(config: DataLoaderConfig) -> DynamicDataLoader:
        ROLLOUT_IDS.add(config.rollout_id)
        return original_loader(config)

    monkeypatch.setattr(remote_rollout_processor_module, "default_fireworks_output_data_loader", wrapped_loader)
    yield
    assert len(ROLLOUT_IDS) == 3, f"Expected to see 3 rollout_ids, but only saw {ROLLOUT_IDS}"


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
    [{"model": "accounts/fireworks/models/gpt-oss-120b", "temperature": 0.5}],
)
@evaluation_test(
    data_loaders=DynamicDataLoader(
        generators=[rows],
    ),
    rollout_processor=RemoteRolloutProcessor(
        remote_base_url=f"http://127.0.0.1:{SERVER_PORT}",
        timeout_seconds=180,
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
