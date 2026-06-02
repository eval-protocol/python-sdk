import os

import pytest

from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest import RunloopRolloutProcessor, evaluation_test


BLUEPRINT_ID = os.environ.get("RUNLOOP_BLUEPRINT_ID")
pytestmark = pytest.mark.skipif(BLUEPRINT_ID is None, reason="RUNLOOP_BLUEPRINT_ID is required for live Runloop smoke")


def rows() -> list[EvaluationRow]:
    return [EvaluationRow(messages=[Message(role="user", content="What is the capital of France?")])]


@evaluation_test(
    completion_params=[{"model": "accounts/fireworks/models/gpt-oss-120b"}],
    input_rows=[rows()],
    rollout_processor=RunloopRolloutProcessor(
        blueprint_id=BLUEPRINT_ID or "bpt_your_blueprint_id",
        server_command=(
            "python -m uvicorn examples.runloop_remote_rollout.server:app "
            "--host 0.0.0.0 --port 8000"
        ),
        port=8000,
        timeout_seconds=180,
    ),
)
async def test_runloop_remote_rollout(row: EvaluationRow) -> EvaluationRow:
    assert row.messages
    return row
