# MANUAL SERVER STARTUP REQUIRED:
#
# For Python server testing, start:
# python -m tests.remote_server.remote_server (runs on http://127.0.0.1:3000)
#
# For TypeScript server testing, start:
# cd tests/remote_server/typescript-server
# npm install
# npm start
#
# The TypeScript server should be running on http://127.0.0.1:3000
# You only need to start one of the servers!

import subprocess
import socket
import time
from typing import List

import pytest
import requests

from eval_protocol.data_loader.dynamic_data_loader import DynamicDataLoader
from eval_protocol.models import EvaluationRow, Message, Status
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.remote_rollout_processor import RemoteRolloutProcessor


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
    subprocess.run(["pkill", "-f", "python -m tests.remote_server.remote_server"])

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
            "--force-early-error",
            "test error",
        ]
    )
    # wait for the server to startup by pollingK
    wait_for_server_to_startup()
    yield
    process.terminate()
    process.wait()


def rows() -> List[EvaluationRow]:
    row = EvaluationRow(messages=[Message(role="user", content="What is the capital of France?")])
    return [row]


@pytest.mark.parametrize("completion_params", [{"model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b"}])
@evaluation_test(
    data_loaders=DynamicDataLoader(
        generators=[rows],
    ),
    rollout_processor=RemoteRolloutProcessor(
        remote_base_url=f"http://127.0.0.1:{SERVER_PORT}",
        timeout_seconds=120,
    ),
)
async def test_remote_rollout_and_fetch_fireworks_propagate_status(row: EvaluationRow) -> EvaluationRow:
    assert row.rollout_status.code == Status.Code.INTERNAL
    assert row.rollout_status.message == "test error"
    return row
