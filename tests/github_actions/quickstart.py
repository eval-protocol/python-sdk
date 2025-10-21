# MANUAL SETUP REQUIRED:
#
# For GitHub Actions testing, you need:
# 1. GitHub repository with rollout.yml workflow (see .github/workflows/rollout.yml)
# 2. Repository secrets configured: FIREWORKS_API_KEY
# 3. Environment variables: GITHUB_TOKEN (with repo and workflow permissions)
#
# The GitHub Actions workflow should accept model, metadata, and model_base_url inputs
# and include: run-name: rollout:${{ fromJSON(inputs.metadata).rollout_id }}

import os
from typing import List

import pytest

from eval_protocol.data_loader.dynamic_data_loader import DynamicDataLoader
from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.github_action_rollout_processor import GithubActionRolloutProcessor


def rows() -> List[EvaluationRow]:
    row = EvaluationRow(messages=[Message(role="user", content="What is the capital of France?")])
    return [row, row, row]


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Only run this test locally (skipped in CI)")
@pytest.mark.parametrize("completion_params", [{"model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b"}])
@evaluation_test(
    data_loaders=DynamicDataLoader(
        generators=[rows],
    ),
    rollout_processor=GithubActionRolloutProcessor(
        owner="eval-protocol",
        repo="python-sdk",
        workflow_id="rollout.yml",  # or you can use numeric ID like "12345678"
        ref=os.getenv("GITHUB_REF", "main"),
        poll_interval=3.0,  # For multi-turn, you'll likely want higher poll interval
        timeout_seconds=300,
    ),
)
async def test_github_actions_quickstart(row: EvaluationRow) -> EvaluationRow:
    """
    End-to-end test:
    - REQUIRES MANUAL SETUP: GitHub Actions workflow with secrets configured
    - trigger GitHub Actions rollout via GithubActionRolloutProcessor
    - fetch traces from Fireworks tracing proxy (uses default FireworksTracingAdapter)
    - FAIL if no traces found or rollout_id missing
    """
    assert row.messages[0].content == "What is the capital of France?", "Row should have correct message content"
    assert len(row.messages) > 1, "Row should have a response. If this fails, we fell back to the original row."
    assert row.execution_metadata.rollout_id, "Row should have a rollout_id from the GitHub Actions rollout"

    return row
