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
    EvaluationRow,
    SingleTurnRolloutProcessor,
    DynamicDataLoader,
    create_braintrust_adapter,
    multi_turn_assistant_to_ground_truth,
)


# uncomment when dataloader is fixed
def braintrust_data_generator():
    adapter = create_braintrust_adapter()
    return adapter.get_evaluation_rows(
        btql_query=f"""
        select: *
        from: project_logs('{os.getenv("BRAINTRUST_PROJECT_ID")}') traces
        filter: is_root = true
        limit: 10
        """
    )


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
    data_loaders=DynamicDataLoader(
        generators=[braintrust_data_generator],
        preprocess_fn=multi_turn_assistant_to_ground_truth,
    ),
    rollout_processor=SingleTurnRolloutProcessor(),
    max_concurrent_evaluations=2,
)
async def test_llm_judge(row: EvaluationRow) -> EvaluationRow:
    return await aha_judge(row)
