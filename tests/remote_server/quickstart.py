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

import os
from typing import List

import pytest

from eval_protocol.data_loader.dynamic_data_loader import DynamicDataLoader
from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.remote_rollout_processor import RemoteRolloutProcessor


def rows() -> List[EvaluationRow]:
    row = EvaluationRow(messages=[Message(role="user", content="What is the capital of France?")])
    return [row, row, row]


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Only run this test locally (skipped in CI)")
@pytest.mark.parametrize("completion_params", [{"model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b"}])
@evaluation_test(
    data_loaders=DynamicDataLoader(
        generators=[rows],
    ),
    rollout_processor=RemoteRolloutProcessor(
        remote_base_url="http://127.0.0.1:3000",
        timeout_seconds=30,
    ),
)
async def test_remote_rollout_and_fetch_fireworks(row: EvaluationRow) -> EvaluationRow:
    """
    End-to-end test:
    - REQUIRES MANUAL SERVER STARTUP: python -m tests.remote_server.remote_server
    - trigger remote rollout via RemoteRolloutProcessor (calls init/status)
    - fetch traces from Langfuse via Fireworks tracing proxy (uses default FireworksTracingAdapter)
    - FAIL if no traces found or rollout_id missing
    """
    assert row.messages[0].content == "What is the capital of France?", "Row should have correct message content"
    assert len(row.messages) > 1, "Row should have a response. If this fails, we fellback to the original row."
    assert row.execution_metadata.rollout_id, "Row should have a rollout_id from the remote rollout"

    return row
