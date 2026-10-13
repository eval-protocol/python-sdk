import os

import pytest

from eval_protocol.models import EvaluationRow
from eval_protocol.pytest import AgentRolloutProcessor
from eval_protocol.pytest.evaluation_test import evaluation_test
from tests.klavis_helpers import MODEL, judge_klavis_response


# Live datasets require a Klavis connection to Gmail/Notion and their seeded test data.
@pytest.mark.skipif(
    not (os.environ.get("KLAVIS_AUTH_TOKEN") and os.environ.get("FIREWORKS_API_KEY")),
    reason="Live Klavis MCP test requires KLAVIS_AUTH_TOKEN and FIREWORKS_API_KEY",
)
@evaluation_test(
    input_dataset=["tests/pytest/datasets/klavis_mcp_test.jsonl"],
    rollout_processor=AgentRolloutProcessor(),
    completion_params=[{"model": f"fireworks_ai/{MODEL}"}],
    mode="pointwise",
    mcp_config_path="tests/pytest/mcp_configurations/klavis_strata_mcp.json",
)
async def test_pytest_klavis_mcp(row: EvaluationRow) -> EvaluationRow:
    assert row.rollout_status.is_finished(), row.rollout_status
    return await judge_klavis_response(row)
