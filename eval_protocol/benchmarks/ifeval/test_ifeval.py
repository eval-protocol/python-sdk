"""
IFEval benchmark test using the evaluation_test decorator.

This test evaluates model responses against instruction-following constraints
from IFBench (Out-of-Distribution IFEval test set).

Run with:
    pytest eval_protocol/benchmarks/ifeval/test_ifeval.py -v
"""

import json
from pathlib import Path
from typing import Any

from eval_protocol.models import EvaluateResult, EvaluationRow, InputMetadata, Message, MetricResult
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.default_single_turn_rollout_process import SingleTurnRolloutProcessor

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


class IFEvalRolloutProcessor(SingleTurnRolloutProcessor):
    """Preprocess rows to extract ground_truth from __GT__ messages."""

    def preprocess_row(self, row: EvaluationRow) -> EvaluationRow:
        """Extract ground truth and remove __GT__ messages."""
        filtered_messages: list[Message] = []
        for m in row.messages:
            content_str = _coerce_content_to_str(m.content)
            if m.role == "system" and content_str.startswith("__GT__:"):
                # Extract ground truth
                row.ground_truth = content_str.split(":", 1)[1].strip()
            else:
                filtered_messages.append(m)
        row.messages = filtered_messages
        return row


@evaluation_test(
    input_messages=_IFBENCH_MESSAGES,
    completion_params=[
        {"model": "fireworks_ai/accounts/fireworks/models/qwen3-8b"}
    ],
    rollout_processor=IFEvalRolloutProcessor(),
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
