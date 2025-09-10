from typing import Any, Dict, List

from eval_protocol.models import EvaluationRow, EvaluateResult, Message
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.default_langchain_rollout_processor import LangGraphRolloutProcessor

from examples.langgraph.reasoning_gpt_oss_120b_graph import build_reasoning_graph
import os
import pytest


def adapter(raw_rows: List[Dict[str, Any]]) -> List[EvaluationRow]:
    rows: List[EvaluationRow] = []
    for raw in raw_rows:
        prompt = raw.get("prompt", "Explain why the sky is blue.")
        rows.append(
            EvaluationRow(
                name=raw.get("name", "row"),
                messages=[Message(role="user", content=prompt)],
                ground_truth=raw.get("gt"),
                input_metadata={"dataset_info": raw},
            )
        )
    return rows


def build_graph_kwargs(cp: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "config": {
            "model": cp.get("model", "accounts/fireworks/models/gpt-oss-120b"),
            "temperature": cp.get("temperature", 0.0),
            "reasoning_effort": cp.get("reasoning_effort"),
        }
    }


def graph_factory(graph_kwargs: Dict[str, Any]) -> Any:
    cfg = graph_kwargs.get("config", {}) if isinstance(graph_kwargs, dict) else {}
    model = cfg.get("model") or "accounts/fireworks/models/gpt-oss-120b"
    temperature = cfg.get("temperature", 0.0)
    reasoning_effort = cfg.get("reasoning_effort")
    return build_reasoning_graph(
        model=model,
        model_provider="fireworks",
        temperature=temperature,
        reasoning_effort=reasoning_effort,
    )


processor = LangGraphRolloutProcessor(
    graph_factory=graph_factory,
    build_graph_kwargs=build_graph_kwargs,
)


@pytest.mark.skipif(os.getenv("FIREWORKS_API_KEY") in (None, ""), reason="FIREWORKS_API_KEY not set")
@evaluation_test(
    input_dataset=["examples/langgraph/data/simple_prompts.jsonl"],
    dataset_adapter=adapter,
    rollout_processor=processor,
    completion_params=[
        {"model": "accounts/fireworks/models/gpt-oss-120b", "temperature": 0.0, "reasoning_effort": "low"}
    ],
    mode="pointwise",
)
async def test_langgraph_reasoning_pointwise(row: EvaluationRow) -> EvaluationRow:
    has_reply = 1.0 if any(m.role == "assistant" for m in (row.messages or [])) else 0.0
    # LOL this doesn't work yet https://github.com/langchain-ai/langgraph/discussions/3547#discussioncomment-13528371
    # assert row.messages[-1].role == "assistant" and row.messages[-1].reasoning_content is not None
    row.evaluation_result = EvaluateResult(
        score=has_reply,
        reason="assistant replied" if has_reply else "no assistant reply",
        metrics={"has_reply": {"is_score_valid": True, "score": has_reply, "reason": "reply presence"}},
    )
    return row
