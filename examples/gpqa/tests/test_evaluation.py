from typing import Any, Dict, List

import re

from eval_protocol.models import EvaluateResult, EvaluationRow, Message, MetricResult
from eval_protocol.pytest.evaluation_test import evaluation_test
from eval_protocol.pytest.default_single_turn_rollout_process import (
    default_single_turn_rollout_processor,
)


SYSTEM_PROMPT = (
    "You are a helpful assistant. Read the question and options carefully. "
    "Express your final answer strictly as a single letter: A, B, C, or D."
)


def extract_abcd_letter(text: str) -> str | None:
    if not text:
        return None
    m = re.search(r"\b([ABCD])\b", text.upper())
    return m.group(1) if m else None


def _build_gpqa_messages_from_hf(max_samples: int | None = 2) -> List[List[Message]]:
    """
    Load GPQA (diamond) from the reference blob CSV and construct prompts.
    For full dataset, call with max_samples=None.
    """
    from datasets import load_dataset  # type: ignore

    url = "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"
    ds = load_dataset("csv", data_files=url, split="train")
    messages_list: List[List[Message]] = []
    # We will store the correct letter in a trailing system message for lookup (not given to the model)
    for ex in ds:
        if max_samples is not None and len(messages_list) >= max_samples:
            break
        q = str(ex.get("Question", ""))
        correct = str(ex.get("Correct Answer", "")).strip()
        inc1 = str(ex.get("Incorrect Answer 1", ""))
        inc2 = str(ex.get("Incorrect Answer 2", ""))
        inc3 = str(ex.get("Incorrect Answer 3", ""))
        choices = [correct, inc1, inc2, inc3]
        user_content = (
            f"{q}\n\n(A) {choices[0]}\n(B) {choices[1]}\n(C) {choices[2]}\n(D) {choices[3]}\n\nAnswer with one letter."
        )
        messages_list.append(
            [
                Message(role="system", content=SYSTEM_PROMPT),
                Message(role="user", content=user_content),
                Message(role="system", content=f"__GT__:A"),
            ]
        )
    if not messages_list:
        raise RuntimeError("Failed to load GPQA messages: no rows found from source")
    return messages_list


_GPQA_INPUT_MESSAGES = _build_gpqa_messages_from_hf(max_samples=2)


@evaluation_test(
    model=["fireworks_ai/accounts/fireworks/models/gpt-oss-120b"],
    input_messages=_GPQA_INPUT_MESSAGES,
    rollout_input_params=[{"temperature": 0.0, "max_tokens": 512}],
    rollout_processor=default_single_turn_rollout_processor,
    aggregation_method="mean",
    threshold_of_success=None,
    num_runs=1,
    max_dataset_rows=2,
    mode="pointwise",
)
def test_gpqa_pointwise(row: EvaluationRow) -> EvaluationRow:
    assistant_msgs = [m for m in row.messages if m.role == "assistant"]
    content = assistant_msgs[-1].content if assistant_msgs else ""

    pred = extract_abcd_letter(content or "")
    # Retrieve GT from the trailing system message we appended
    gt_tokens = [m.content for m in row.messages if m.role == "system" and (m.content or "").startswith("__GT__:")]
    gt = gt_tokens[-1].split(":", 1)[1].strip() if gt_tokens else None

    is_valid = pred is not None and gt in {"A", "B", "C", "D"}
    score = 1.0 if (is_valid and pred == gt) else 0.0

    row.evaluation_result = EvaluateResult(
        score=score,
        reason=("Correct option" if score == 1.0 else "Incorrect option"),
        is_score_valid=is_valid,
        metrics={
            "exact_match": MetricResult(
                score=score,
                is_score_valid=is_valid,
                reason=("Matched" if score == 1.0 else "Not matched"),
                data={"pred": pred, "gt": gt},
            )
        },
    )
    return row


