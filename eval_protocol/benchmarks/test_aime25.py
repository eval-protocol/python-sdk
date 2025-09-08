from typing import Any, Dict, List, Optional

from eval_protocol.models import (
    EvaluateResult,
    EvaluationRow,
    Message,
    MetricResult,
    ChatCompletionContentPartTextParam,
)
from eval_protocol.pytest.default_single_turn_rollout_process import (
    SingleTurnRolloutProcessor,
)
from eval_protocol.pytest.evaluation_test import evaluation_test

SYSTEM_PROMPT = (
    "You are a helpful math assistant. Please reason step by step, and put your final answer within \\boxed{...}."
)


def _coerce_content_to_str(
    content: str | list[ChatCompletionContentPartTextParam] | None,
) -> str:
    if isinstance(content, list):
        return "".join([getattr(p, "text", str(p)) for p in content])
    return str(content or "")


def _extract_boxed_text(text: str) -> str:
    import re

    if not text:
        return ""

    pattern_boxed = r"boxed{(.*?)}|framebox{(.*?)}"
    matches = re.findall(pattern_boxed, text, re.DOTALL)
    if matches:
        for match in matches[::-1]:
            for group in match:
                if group:
                    return group.split(",")[-1].strip()
    matches_digits = re.findall(r"\d+", text, re.DOTALL)
    if matches_digits:
        return matches_digits[-1]
    return ""


def _normalize_to_int_or_none(s: Optional[str]) -> Optional[int]:
    import re

    if s is None:
        return None
    m = re.match(r"\d+", str(s).strip())
    if not m:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None


def aime2025_dataset_adapter(rows: List[Dict[str, Any]]) -> List[EvaluationRow]:
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
    input_dataset=[
        "https://huggingface.co/datasets/opencompass/AIME2025/raw/main/aime2025-I.jsonl",
        "https://huggingface.co/datasets/opencompass/AIME2025/raw/main/aime2025-II.jsonl",
    ],
    dataset_adapter=aime2025_dataset_adapter,
    completion_params=[
        {
            "max_tokens": 131000,
            "extra_body": {"reasoning_effort": "low"},
            "model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b",
        }
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    aggregation_method="mean",
    passed_threshold=0.8,
    num_runs=8,
    max_dataset_rows=2,
    max_concurrent_rollouts=4,
    mode="pointwise",
)
def test_aime25_pointwise(row: EvaluationRow) -> EvaluationRow:
    assistant_msgs = [m for m in row.messages if m.role == "assistant"]
    raw_content = assistant_msgs[-1].content if assistant_msgs else ""
    content_str = _coerce_content_to_str(raw_content)

    extracted_text = _extract_boxed_text(content_str)
    extracted_int = _normalize_to_int_or_none(extracted_text)
    gt_int = _normalize_to_int_or_none(str(row.ground_truth))

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
