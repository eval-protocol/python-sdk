# GitHub Actions rollout processor test
#
# Pattern: Test creates empty rows with row_id, worker loads dataset by row_id
# Setup: GitHub repo with rollout.yml, FIREWORKS_API_KEY secret, GITHUB_TOKEN env var

import os
from typing import List

import pytest

from eval_protocol.data_loader.dynamic_data_loader import DynamicDataLoader
from eval_protocol.models import EvaluationRow, InputMetadata
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.github_action_rollout_processor import GithubActionRolloutProcessor
import eval_protocol.pytest.github_action_rollout_processor as github_action_rollout_processor_module
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

    original_loader = github_action_rollout_processor_module.default_fireworks_output_data_loader

    def wrapped_loader(config: DataLoaderConfig) -> DynamicDataLoader:
        ROLLOUT_IDS.add(config.rollout_id)
        return original_loader(config)

    monkeypatch.setattr(github_action_rollout_processor_module, "default_fireworks_output_data_loader", wrapped_loader)
    yield
    assert len(ROLLOUT_IDS) == 3, f"Expected to see 3 rollout_ids, but only saw {ROLLOUT_IDS}"


def rows() -> List[EvaluationRow]:
    return [
        EvaluationRow(input_metadata=InputMetadata(row_id=str(i)))
        for i in range(
            3
        )  # In this example we use index to associate rows. Dataset is assumed to be accessible to the worker.
    ]


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Only run this test locally (skipped in CI)")
@pytest.mark.parametrize(
    "completion_params", [{"model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b", "temperature": 0.5}]
)
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
async def test_github_actions_rollout(row: EvaluationRow) -> EvaluationRow:
    """Test GitHub Actions rollout with worker-controlled dataset."""
    assert row.execution_metadata.rollout_id is not None

    # This dataset is built into github_actions/rollout_worker.py
    if row.messages[0].content == "What is the capital of France?":
        assert row.input_metadata.row_id == "0"
    elif row.messages[0].content == "What is the capital of Germany?":
        assert row.input_metadata.row_id == "1"
    elif row.messages[0].content == "What is the capital of Italy?":
        assert row.input_metadata.row_id == "2"
    else:
        assert False, "Row should have correct message content"
    assert len(row.messages) > 1, "Row should have a response. If this fails, we fell back to the original row."

    return row
