"""
Example for using Braintrust with the aha judge.
"""

import os

import pytest

from eval_protocol.models import EvaluationRow
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.default_single_turn_rollout_process import SingleTurnRolloutProcessor
from eval_protocol.quickstart.utils import split_multi_turn_rows
from eval_protocol.adapters.braintrust import create_braintrust_adapter
from eval_protocol.quickstart import aha_judge

# adapter = create_braintrust_adapter()


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Skip in CI")
@pytest.mark.asyncio
@evaluation_test(
    input_rows=[
        #         adapter.get_evaluation_rows(
        #             btql_query=f"""
        # select: *
        # from: project_logs('{os.getenv("BRAINTRUST_PROJECT_ID")}') traces
        # filter: is_root = true
        # limit: 10
        # """
        #         )
        []
    ],
    completion_params=[
        {"model": "gpt-4.1"},
        {
            "max_tokens": 131000,
            "extra_body": {"reasoning_effort": "medium"},
            "model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b",
        },
        {
            "max_tokens": 131000,
            "extra_body": {"reasoning_effort": "low"},
            "model": "fireworks_ai/accounts/fireworks/models/gpt-oss-20b",
        },
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    preprocess_fn=split_multi_turn_rows,
    max_concurrent_rollouts=64,
    mode="all",
)
async def test_llm_judge(rows: list[EvaluationRow]) -> list[EvaluationRow]:
    return await aha_judge(rows)
