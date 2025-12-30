import pytest

pytest.skip(
    "Skipping Chinook langgraph integration tests (requires external services/credentials).",
    allow_module_level=True,
)

from eval_protocol.models import EvaluateResult, EvaluationRow, Message
from eval_protocol.pytest import evaluation_test

from eval_protocol.pytest.default_langchain_rollout_processor import LangGraphRolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig, CompletionParams

from tests.chinook.langgraph.graph import build_graph
from typing import Any, Dict
from openai import OpenAI
import os


def build_graph_kwargs(cp: CompletionParams) -> Dict[str, Any]:
    # Minimal runnable config mapping; not used by current graph but kept for API parity
    model = cp.get("model")
    provider = cp.get("provider")
    return {"config": {"model": model, "provider": provider}}


@pytest.mark.asyncio
@pytest.mark.skipif(os.getenv("FIREWORKS_API_KEY") in (None, ""), reason="FIREWORKS_API_KEY not set")
@evaluation_test(
    input_messages=[[[Message(role="user", content="What is the total number of tracks in the database?")]]],
    completion_params=[{"model": "accounts/fireworks/models/kimi-k2-instruct", "provider": "fireworks"}],
    rollout_processor=LangGraphRolloutProcessor(
        graph_factory=lambda _: build_graph(),
        build_graph_kwargs=build_graph_kwargs,
        input_key="messages",
        output_key="messages",
    ),
    passed_threshold=1.0,
)
async def test_langgraph_simple_query(row: EvaluationRow) -> EvaluationRow:
    last_assistant_message = row.last_assistant_message()
    if last_assistant_message is None or not last_assistant_message.content:
        row.evaluation_result = EvaluateResult(score=0.0, reason="No assistant message found")
        return row

    # Ensure role mapping is correct
    assert row.messages and row.messages[0].role == "user"
    assert row.messages[-1].role == "assistant"
    score_value = 1.0 if "3503" in last_assistant_message.content else 0.0
    reason_text = last_assistant_message.content[:500]

    row.evaluation_result = EvaluateResult(score=score_value, reason=reason_text)
    return row
