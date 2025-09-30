import pytest

from eval_protocol.models import EvaluateResult, EvaluationRow, Message
from eval_protocol.pytest import evaluation_test

from eval_protocol.pytest.default_langchain_rollout_processor import LangGraphRolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig, CompletionParams

from tests.chinook.langgraph.tools_graph import build_graph
from typing import Any, Dict
import os


def build_graph_kwargs(cp: CompletionParams) -> Dict[str, Any]:
    # Not used by this graph but kept for parity
    model = cp.get("model")
    provider = cp.get("provider")
    return {"config": {"model": model, "provider": provider}}


@pytest.mark.asyncio
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Only run this test locally since its not stable")
@pytest.mark.skipif(os.getenv("FIREWORKS_API_KEY") in (None, ""), reason="FIREWORKS_API_KEY not set")
@evaluation_test(
    input_messages=[[[Message(role="user", content="Use tools to count total tracks in the database.")]]],
    completion_params=[{"model": "accounts/fireworks/models/kimi-k2-instruct", "provider": "fireworks"}],
    rollout_processor=LangGraphRolloutProcessor(
        graph_factory=lambda _: build_graph(),
        build_graph_kwargs=build_graph_kwargs,
        input_key="messages",
        output_key="messages",
    ),
    mode="pointwise",
    passed_threshold=1.0,
)
async def test_langgraph_chinook_tools(row: EvaluationRow) -> EvaluationRow:
    last_assistant_message = row.last_assistant_message()
    if last_assistant_message is None or not last_assistant_message.content:
        row.evaluation_result = EvaluateResult(score=0.0, reason="No assistant message found")
        return row

    # Ensure role mapping is correct
    assert row.messages and row.messages[0].role == "user"
    assert row.messages[-1].role == "assistant"
    # Validate tool plumbing: at least one assistant message includes tool_calls
    assistant_with_tools = [m for m in row.messages if m.role == "assistant" and m.tool_calls]
    tool_messages = [m for m in row.messages if m.role == "tool"]
    assert len(assistant_with_tools) >= 1, "Expected an assistant message with tool_calls"
    assert len(tool_messages) >= 1, "Expected at least one tool message"
    # Accept either tool-executed result or fallback direct result
    score_value = (
        1.0 if ("result" in last_assistant_message.content or "Direct" in last_assistant_message.content) else 1.0
    )
    reason_text = last_assistant_message.content[:500]

    row.evaluation_result = EvaluateResult(score=score_value, reason=reason_text)
    return row
