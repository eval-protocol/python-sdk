# MANUAL SERVER STARTUP REQUIRED:
#
# For Python server testing, start:
# python -m tests.remote_server.remote_server (runs on http://127.0.0.1:7077)
#
# For TypeScript server testing, start:
# cd /Users/derekxu/Documents/code/python-sdk/tests/remote_server/typescript-server
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
    return adapter.get_evaluation_rows(tags=[f"rollout_id:{rollout_id}"], max_retries=5)


def langfuse_output_data_loader(rollout_id: str) -> DynamicDataLoader:
    return DynamicDataLoader(
        generators=[lambda: fetch_langfuse_traces(rollout_id)], preprocess_fn=filter_longest_conversation
    )


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
        output_data_loader=langfuse_output_data_loader,
        model_base_url="https://tracing.fireworks.ai/project_id/cmg5fd57b0006y107kuxkcrhk",
    ),
)
async def test_remote_rollout_and_fetch_langfuse(row: EvaluationRow) -> EvaluationRow:
    """
    End-to-end test:
    - REQUIRES MANUAL SERVER STARTUP: python -m tests.remote_server.remote_server
    - trigger remote rollout via RemoteRolloutProcessor (calls init/status)
    - fetch traces from Langfuse filtered by metadata via output_data_loader; FAIL if none found
    """
    assert row.messages[0].content == "What is the capital of France?", "Row should have correct message content"
    assert len(row.messages) > 1, "Row should have a response. If this fails, we fellback to the original row."

    assert row.execution_metadata.rollout_id in ROLLOUT_IDS, (
        f"Row rollout_id {row.execution_metadata.rollout_id} should be in tracked rollout_ids: {ROLLOUT_IDS}"
    )

    return row
