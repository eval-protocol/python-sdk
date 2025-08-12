import os
from typing import Any, Dict, List

from eval_protocol.models import EvaluateResult, EvaluationRow, Message, MetricResult
from eval_protocol.pytest.default_single_turn_rollout_process import (
    default_single_turn_rollout_processor,
)
from eval_protocol.pytest.evaluation_test import evaluation_test
from examples.aime2025_chat_completion.main import _extract_boxed_text, _normalize_to_int_or_none

SYSTEM_PROMPT = (
    "You are a helpful math assistant. Please reason step by step, and put your " "final answer within \\boxed{...}."
)

"""
This test consumes the AIME2025 dataset directly from Hugging Face JSONL URLs via
the evaluation_test dataset loader + adapter. By default, max_dataset_rows=2 to
keep CI fast; set it to None to run the full dataset.
"""


def _ep_int(var_name: str, default_value: int | None) -> int | None:
    """Read EP_*-prefixed integer or 'None' from environment for easy overrides."""
    raw = os.getenv(var_name)
    if raw is None:
        return default_value
    raw_stripped = raw.strip().lower()
    if raw_stripped == "none":
        return None
    try:
        return int(raw_stripped)
    except ValueError:
        return default_value


def aime2025_dataset_adapter(rows: List[Dict[str, Any]]) -> List[EvaluationRow]:
    """
    Convert raw AIME2025 rows (with keys 'question' and 'answer') to EvaluationRow.
    Limits handled by evaluation_test's max_dataset_rows, so adapter is simple.
    """
    converted: List[EvaluationRow] = []
    for r in rows:
        question = r.get("question", "")
        answer = r.get("answer", None)
        messages = [
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=str(question)),
        ]
        converted.append(EvaluationRow(messages=messages, ground_truth=str(answer) if answer is not None else None))
    return converted


@evaluation_test(
    model=["fireworks_ai/accounts/fireworks/models/gpt-oss-120b"],
    input_dataset=[
        "https://huggingface.co/datasets/opencompass/AIME2025/raw/main/aime2025-I.jsonl",
        "https://huggingface.co/datasets/opencompass/AIME2025/raw/main/aime2025-II.jsonl",
    ],
    dataset_adapter=aime2025_dataset_adapter,
    rollout_input_params=[
        {"extra_body": {"reasoning_effort": "low"}},
        {},
        {"extra_body": {"reasoning_effort": "high"}},
    ],
    rollout_processor=default_single_turn_rollout_processor,
    aggregation_method="mean",
    threshold_of_success=None,
    num_runs=2,
    max_dataset_rows=2,
    max_concurrent_rollouts=4,
    mode="pointwise",
)
def test_aime2025_pointwise(row: EvaluationRow) -> EvaluationRow:
    """
    Pointwise evaluation of AIME2025 rows: extract final integer from assistant message and compare to ground truth.
    """
    # After rollout, the last message should be assistant's response
    assistant_msgs = [m for m in row.messages if m.role == "assistant"]
    content = assistant_msgs[-1].content if assistant_msgs else ""

    extracted_text = _extract_boxed_text(content or "")
    extracted_int = _normalize_to_int_or_none(extracted_text)
    # Ground truth comes from dataset_adapter
    gt_int = _normalize_to_int_or_none(row.ground_truth or "")

    is_valid = extracted_int is not None and gt_int is not None
    score = 1.0 if (is_valid and extracted_int == gt_int) else 0.0

    metrics = {
        "exact_match": MetricResult(
            score=score,
            is_score_valid=is_valid,
            reason=(
                "Parsed both integers and they matched"
                if score == 1.0
                else ("Parsed integers did not match" if is_valid else "Failed to parse integer")
            ),
            data={
                "extracted_text": extracted_text,
                "extracted_int": extracted_int,
                "ground_truth_int": gt_int,
            },
        )
    }

    row.evaluation_result = EvaluateResult(
        score=score,
        reason=("Answer correct" if score == 1.0 else "Answer incorrect"),
        is_score_valid=is_valid,
        metrics=metrics,
    )
    return row
