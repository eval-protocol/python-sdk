from typing import Any, Dict, List

from eval_protocol.models import EvaluationRow, EvaluateResult, Message
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.default_langchain_rollout_processor import LangGraphRolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig

from .simple_graph import build_simple_graph
import os
import pytest


def adapter(raw_rows: List[Dict[str, Any]]) -> List[EvaluationRow]:
    rows: List[EvaluationRow] = []
    for raw in raw_rows:
        prompt = raw.get("prompt", "Say hello")
        rows.append(
            EvaluationRow(
                name=raw.get("name", "row"),
                messages=[Message(role="user", content=prompt)],
                ground_truth=raw.get("gt"),
                input_metadata={"dataset_info": raw},
            )
        )
    return rows


def graph_factory(config: RolloutProcessorConfig) -> Any:
    cp = config.completion_params or {}
    model = cp.get("model") or "accounts/fireworks/models/kimi-k2-instruct"
    temperature = cp.get("temperature", 0.0)
    # Provider is fixed to fireworks for this example; can be extended via cp if needed
    return build_simple_graph(model=model, model_provider="fireworks", temperature=temperature)


processor = LangGraphRolloutProcessor(graph_factory=graph_factory)


@pytest.mark.skipif(os.getenv("FIREWORKS_API_KEY") in (None, ""), reason="FIREWORKS_API_KEY not set")
@evaluation_test(
    input_dataset=["examples/langgraph/data/simple_prompts.jsonl"],
    dataset_adapter=adapter,
    rollout_processor=processor,
    completion_params=[{"model": "accounts/fireworks/models/kimi-k2-instruct", "temperature": 0.0}],
    mode="pointwise",
)
async def test_langgraph_pointwise(row: EvaluationRow) -> EvaluationRow:
    # Example scoring: did assistant reply?
    has_reply = 1.0 if any(m.role == "assistant" for m in (row.messages or [])) else 0.0
    row.evaluation_result = EvaluateResult(
        score=has_reply,
        reason="assistant replied" if has_reply else "no assistant reply",
        metrics={"has_reply": {"is_score_valid": True, "score": has_reply, "reason": "reply presence"}},
    )
    return row
