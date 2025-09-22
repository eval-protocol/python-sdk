"""
Example for using Braintrust with the aha judge.
"""

import os

import pytest

# Skip entire module in CI to prevent import-time side effects
if os.environ.get("CI") == "true":
    pytest.skip("Skip quickstart in CI", allow_module_level=True)

from eval_protocol import (
    evaluation_test,
    aha_judge,
    multi_turn_assistant_to_ground_truth,
    EvaluationRow,
    SingleTurnRolloutProcessor,
    create_braintrust_adapter,
    DefaultParameterIdGenerator,
)

# adapter = create_braintrust_adapter()
# input_rows = [
#     adapter.get_evaluation_rows(
#         btql_query=f"""
#     select: *
#     from: project_logs('{os.getenv("BRAINTRUST_PROJECT_ID")}') traces
#     filter: is_root = true
#     limit: 10
#     """
#     )
# ]
input_rows = []
# uncomment when dataloader is fixed


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Skip in CI")
@pytest.mark.parametrize(
    "completion_params",
    [
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
)
@evaluation_test(
    input_rows=[input_rows],
    rollout_processor=SingleTurnRolloutProcessor(),
    preprocess_fn=multi_turn_assistant_to_ground_truth,
    max_concurrent_evaluations=2,
)
async def test_llm_judge(row: EvaluationRow) -> EvaluationRow:
    return await aha_judge(row)
