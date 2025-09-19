"""
Example for using Braintrust with the aha judge.
"""

import os

import pytest

from eval_protocol import (
    evaluation_test,
    aha_judge,
    multi_turn_assistant_to_ground_truth,
    EvaluationRow,
    SingleTurnRolloutProcessor,
    create_braintrust_adapter,
)
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
    preprocess_fn=multi_turn_assistant_to_ground_truth,
    max_concurrent_rollouts=64,
    aggregation_method="bootstrap",
)
async def test_llm_judge(row: EvaluationRow) -> EvaluationRow:
    return await aha_judge(row)
