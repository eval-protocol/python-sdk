"""Evaluate Langfuse traces with Fireworks-only rollout (no LiteLLM router).

This uses SingleTurnRolloutProcessor to call Fireworks directly via the
litellm client (not the proxy server) and then runs the AHA judge (also on
Fireworks by default). Scores are pushed back to Langfuse.
"""

from datetime import datetime
import os

import pytest

from eval_protocol import (
    DynamicDataLoader,
    EvaluationRow,
    SingleTurnRolloutProcessor,
    aha_judge,
    create_langfuse_adapter,
    evaluation_test,
    multi_turn_assistant_to_ground_truth,
)


def langfuse_fireworks_data_generator() -> list[EvaluationRow]:
    adapter = create_langfuse_adapter()
    return adapter.get_evaluation_rows(
        environment=os.getenv("LANGFUSE_ENVIRONMENT", "local"),
        limit=int(os.getenv("LANGFUSE_LIMIT", "100")),
        sample_size=int(os.getenv("LANGFUSE_SAMPLE_SIZE", "20")),
        include_tool_calls=bool(int(os.getenv("LANGFUSE_INCLUDE_TOOL_CALLS", "1"))),
        sleep_between_gets=float(os.getenv("LANGFUSE_SLEEP", "0.5")),
        max_retries=int(os.getenv("LANGFUSE_MAX_RETRIES", "3")),
        from_timestamp=None,
        to_timestamp=datetime.utcnow(),
    )


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Skip in CI")
@pytest.mark.skipif(
    not os.getenv("FIREWORKS_API_KEY"),
    reason="Requires FIREWORKS_API_KEY",
)
@pytest.mark.parametrize(
    "completion_params",
    [
        {
            "model": os.getenv("FIREWORKS_COMPLETION_MODEL", "accounts/fireworks/models/kimi-k2-instruct"),
            "api_key": os.getenv("FIREWORKS_API_KEY"),
            "base_url": os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"),
            "temperature": float(os.getenv("FIREWORKS_TEMPERATURE", "0.2")),
            "max_tokens": int(os.getenv("FIREWORKS_MAX_TOKENS", "2048")),
        },
    ],
)
@evaluation_test(
    data_loaders=DynamicDataLoader(
        generators=[langfuse_fireworks_data_generator],
        preprocess_fn=multi_turn_assistant_to_ground_truth,
    ),
    rollout_processor=SingleTurnRolloutProcessor(),
    max_concurrent_evaluations=int(os.getenv("FIREWORKS_MAX_CONCURRENCY", "2")),
)
async def test_llm_judge_fireworks_only(row: EvaluationRow) -> EvaluationRow:
    adapter = create_langfuse_adapter()
    return await aha_judge(row, adapter=adapter)
