"""
Example for using Langfuse with the aha judge.
"""

from datetime import datetime
import os

import pytest

from eval_protocol import (
    evaluation_test,
    aha_judge,
    multi_turn_assistant_to_ground_truth,
    EvaluationRow,
    SingleTurnRolloutProcessor,
    create_langfuse_adapter,
    DefaultParameterIdGenerator,
)

from eval_protocol.quickstart import aha_judge

adapter = create_langfuse_adapter()
input_rows = adapter.get_evaluation_rows(
    to_timestamp=datetime(2025, 9, 12, 0, 11, 18),
    limit=711,
    sample_size=50,
    sleep_between_gets=3.0,
    max_retries=5,
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
    input_rows=[input_rows],
    rollout_processor=SingleTurnRolloutProcessor(),
    preprocess_fn=multi_turn_assistant_to_ground_truth,
    max_concurrent_evaluations=2,
)
async def test_llm_judge(row: EvaluationRow) -> EvaluationRow:
    return await aha_judge(row)
