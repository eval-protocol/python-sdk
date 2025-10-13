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
from eval_protocol.models import EvaluationRow, InputMetadata
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.github_action_rollout_processor import GithubActionRolloutProcessor
from eval_protocol.types.remote_rollout_processor import DataLoaderConfig
from eval_protocol.adapters.fireworks_tracing import FireworksTracingAdapter
from eval_protocol.quickstart.utils import filter_longest_conversation

ROLLOUT_IDS = set()


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
    return adapter.get_evaluation_rows(tags=[f"rollout_id:{config.rollout_id}"], max_retries=5)


def fireworks_output_data_loader(config: DataLoaderConfig) -> DynamicDataLoader:
    return DynamicDataLoader(
        generators=[lambda: fetch_fireworks_traces(config)], preprocess_fn=filter_longest_conversation
    )


def rows() -> List[EvaluationRow]:
    return [
        EvaluationRow(input_metadata=InputMetadata(row_id=str(i)))
        for i in range(3)  # In this example we use index to associate rows.
    ]


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
        output_data_loader=fireworks_output_data_loader,
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
