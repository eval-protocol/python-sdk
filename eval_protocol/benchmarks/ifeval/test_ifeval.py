"""
IFEval benchmark test using the evaluation_test decorator.

This test evaluates model responses against instruction-following constraints
from IFBench (Out-of-Distribution IFEval test set).

Run with:
    pytest eval_protocol/benchmarks/ifeval/test_ifeval.py -v
"""

import asyncio
import json
import os
from pathlib import Path

import pytest

from eval_protocol.models import EvaluateResult, EvaluationRow, Message, MetricResult
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.default_single_turn_rollout_process import SingleTurnRolloutProcessor
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig

from .reward import ifeval_partial_credit_reward


def _load_ifbench_messages() -> list[list[list[Message]]]:
    """Load IFBench test data as messages with ground truth."""
    data_path = Path(__file__).parent / "data" / "ifbench_test_sample.jsonl"
    messages_list: list[list[Message]] = []

    with open(data_path) as f:
        for line in f:
            row = json.loads(line)
            # Convert to Message objects
            messages = [Message(role=m["role"], content=m["content"]) for m in row["messages"]]
            # Add ground truth as a system message (will be extracted later)
            messages.insert(0, Message(role="system", content=f"__GT__:{row['ground_truth']}"))
            messages_list.append(messages)

    return [messages_list]


def _coerce_content_to_str(content: str | list | None) -> str:
    """Convert message content to string."""
    if isinstance(content, list):
        return "".join(str(p.get("text", p)) if isinstance(p, dict) else str(p) for p in content)
    return str(content or "")


_IFBENCH_MESSAGES = _load_ifbench_messages()


class IFEvalGroundTruthRolloutProcessor(RolloutProcessor):
    """Extract ground truth from __GT__ system messages, then run single-turn rollouts."""

    def __init__(self) -> None:
        super().__init__()
        self.single_turn_processor = SingleTurnRolloutProcessor()

    def __call__(
        self, rows: list[EvaluationRow], config: RolloutProcessorConfig
    ) -> list[asyncio.Task[EvaluationRow]]:
        processed: list[EvaluationRow] = []
        for r in rows:
            gt_tokens: list[str] = []
            for m in r.messages:
                if m.role == "system":
                    content_str = _coerce_content_to_str(m.content)
                    if content_str.startswith("__GT__:"):
                        gt_tokens.append(content_str)
            if gt_tokens:
                r.ground_truth = gt_tokens[-1].split(":", 1)[1].strip()
                filtered: list[Message] = []
                for m in r.messages:
                    if m.role == "system":
                        content_str = _coerce_content_to_str(m.content)
                        if content_str.startswith("__GT__:"):
                            continue
                    filtered.append(m)
                r.messages = filtered
            processed.append(r)
        return self.single_turn_processor(processed, config)


@pytest.mark.skipif(
    not os.getenv("FIREWORKS_API_KEY"),
    reason="FIREWORKS_API_KEY not set",
)
@evaluation_test(
    input_messages=_IFBENCH_MESSAGES,
    completion_params=[
        {"model": "fireworks_ai/accounts/fireworks/models/qwen3-8b"}
    ],
    rollout_processor=IFEvalGroundTruthRolloutProcessor(),
    aggregation_method="mean",
    passed_threshold=0.5,
    num_runs=1,
    mode="pointwise",
)
def test_ifeval_benchmark(row: EvaluationRow) -> EvaluationRow:
    """
    Evaluate instruction-following constraints.

    Returns partial credit score (0.0 to 1.0) representing the fraction
    of constraints satisfied in the response.
    """
    # Get the assistant's response
    assistant_msgs = [m for m in row.messages if m.role == "assistant"]
    response = _coerce_content_to_str(assistant_msgs[-1].content) if assistant_msgs else ""

    # Evaluate against ground truth constraints
    score = ifeval_partial_credit_reward(response, row.ground_truth)

    row.evaluation_result = EvaluateResult(
        score=score,
        reason=f"Satisfied {score * 100:.1f}% of constraints",
        is_score_valid=True,
        metrics={
            "ifeval_partial_credit": MetricResult(
                score=score,
                is_score_valid=True,
                reason="Partial credit score based on fraction of constraints satisfied",
            )
        },
    )
    return row
