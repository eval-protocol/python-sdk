# MANUAL GITHUB ACTIONS SETUP REQUIRED:
#
# This test requires a GitHub repository with the rollout.yml workflow.
# The workflow should be available at .github/workflows/rollout.yml
#
# Required GitHub secrets:
# - FIREWORKS_API_KEY: Your Fireworks API key for model calls
#
# Required environment variables for this test:
# - GITHUB_TOKEN: GitHub token with workflow dispatch permissions
# - GITHUB_OWNER: GitHub repository owner (e.g., "your-org")
# - GITHUB_REPO: GitHub repository name (e.g., "your-repo")
# - GITHUB_REF: Branch/ref to run workflow on (e.g., "main")

import os
from typing import List

import pytest

from eval_protocol.data_loader.dynamic_data_loader import DynamicDataLoader
from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.github_action_rollout_processor import GithubActionRolloutProcessor

ROLLOUT_IDS = set()


@pytest.fixture(autouse=True)
def check_rollout_coverage():
    """Ensure we processed all expected rollout_ids"""
    global ROLLOUT_IDS
    ROLLOUT_IDS.clear()
    yield

    assert len(ROLLOUT_IDS) == 3, f"Expected to see 3 rollout_ids, but only saw {ROLLOUT_IDS}"


def rows() -> List[EvaluationRow]:
    row = EvaluationRow(messages=[Message(role="user", content="What is the capital of France?")])
    return [row, row, row]


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Only run this test locally (skipped in CI)")
@pytest.mark.parametrize("completion_params", [{"model": "accounts/fireworks/models/gpt-oss-120b"}])
@evaluation_test(
    data_loaders=DynamicDataLoader(
        generators=[rows],
    ),
    rollout_processor=GithubActionRolloutProcessor(
        owner="eval-protocol",
        repo="python-sdk",
        workflow_id="rollout.yml",  # or you can use numeric ID like "12345678"
        ref=os.getenv("GITHUB_REF", "main"),
        timeout_seconds=300,
    ),
)
async def test_github_actions_rollout_direct_artifacts(row: EvaluationRow) -> EvaluationRow:
    """
    End-to-end test for GitHub Actions rollout processor with direct artifact fetching:
    - REQUIRES GITHUB REPOSITORY WITH WORKFLOW: .github/workflows/rollout.yml
    - REQUIRES ENVIRONMENT VARIABLES: GITHUB_TOKEN, GITHUB_OWNER, GITHUB_REPO, GITHUB_REF
    - REQUIRES GITHUB SECRET: FIREWORKS_API_KEY
    - Triggers GitHub Actions workflow via GithubActionRolloutProcessor
    - Fetches conversation traces directly from GitHub Actions artifacts
    - FAIL if no trace artifact found (indicates workflow didn't run or save trace properly)
    """
    # Track rollout IDs for coverage check
    global ROLLOUT_IDS
    ROLLOUT_IDS.add(row.execution_metadata.rollout_id)

    assert row.messages[0].content == "What is the capital of France?", "Row should have correct message content"
    assert len(row.messages) > 1, "Row should have a response. If this fails, we fell back to the original row."

    return row
