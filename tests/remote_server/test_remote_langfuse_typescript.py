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

    # Verify we've seen the expected number of rollout_ids after test is done
    expected_rollout_count = 3
    assert len(ROLLOUT_IDS) == expected_rollout_count, (
        f"Expected to see {expected_rollout_count} rollout_ids, but only saw {len(ROLLOUT_IDS)}: {ROLLOUT_IDS}"
    )


def fetch_langfuse_traces(rollout_id: str) -> List[EvaluationRow]:
    global ROLLOUT_IDS  # Track all rollout_ids we've seen
    ROLLOUT_IDS.add(rollout_id)

    adapter = create_langfuse_adapter()
    return adapter.get_evaluation_rows(tags=[f"rollout_id:{rollout_id}"])


def langfuse_output_data_loader(rollout_id: str) -> DynamicDataLoader:
    return DynamicDataLoader(
        generators=[lambda: fetch_langfuse_traces(rollout_id)], preprocess_fn=filter_longest_conversation
    )


def rows() -> List[EvaluationRow]:
    # Minimal single-user-turn message to trigger a response
    row = EvaluationRow(messages=[Message(role="user", content="What is the capital of France?")])
    return [row, row, row]


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Only run this test locally (skipped in CI)")
@pytest.mark.parametrize("completion_params", [{"model": "gpt-5"}])
@evaluation_test(
    data_loaders=DynamicDataLoader(
        generators=[rows],
    ),
    rollout_processor=RemoteRolloutProcessor(
        remote_base_url="http://127.0.0.1:3000",
        timeout_seconds=30,
        output_data_loader=langfuse_output_data_loader,
    ),
)
async def test_remote_rollout_and_fetch_langfuse_typescript(row: EvaluationRow) -> EvaluationRow:
    """
    End-to-end test:
    - remote server started at import time
    - trigger remote rollout via RemoteRolloutProcessor (calls init/status)
    - fetch traces from Langfuse filtered by metadata via output_data_loader; FAIL if none found
    """
    assert row.messages[0].content == "What is the capital of France?", "Row should have correct message content"
    assert row.execution_metadata.rollout_id in ROLLOUT_IDS, (
        f"Row rollout_id {row.execution_metadata.rollout_id} should be in tracked rollout_ids: {ROLLOUT_IDS}"
    )

    return row
