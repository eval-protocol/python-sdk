# REMOTE SERVER OPTIONS:
#
# Option 1: Use Vercel dev server locally (recommended for development)
# cd eval_protocol/quickstart/svg_agent/vercel_svg_server
# vercel dev
# Then change remote_base_url to: "http://localhost:3000"
#
# Option 2: Use deployed Vercel production server (current configuration)
# No setup needed - uses the deployed serverless function
# Currently using: https://vercel-svg-server-qntltzfaq-xzrdereks-projects.vercel.app
#
# Option 3: Use local Python server (for testing)
# python -m tests.remote_server.remote_server
# Then change remote_base_url to: "http://127.0.0.1:3000"

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
        # For local Vercel dev: "http://localhost:3000"
        # For production Vercel: (current setting)
        remote_base_url="https://vercel-svg-server.vercel.app",
        timeout_seconds=30,
    ),
)
async def test_remote_rollout_and_fetch_fireworks(row: EvaluationRow) -> EvaluationRow:
    """
    End-to-end test with Vercel production server:
    - Uses deployed Vercel serverless function (no manual startup needed)
    - trigger remote rollout via RemoteRolloutProcessor (calls init/status)
    - fetch traces from Langfuse via Fireworks tracing proxy (uses default FireworksTracingAdapter)
    - FAIL if no traces found or rollout_id missing
    """
    assert row.messages[0].content == "What is the capital of France?", "Row should have correct message content"
    assert len(row.messages) > 1, "Row should have a response. If this fails, we fellback to the original row."
    assert row.execution_metadata.rollout_id, "Row should have a rollout_id from the remote rollout"

    return row
