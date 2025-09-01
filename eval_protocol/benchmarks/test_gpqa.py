import asyncio
import csv
import io
import re

import requests

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
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig

SYSTEM_PROMPT = (
    "You are a helpful assistant. Read the question and options carefully. "
    "Express your final answer strictly as a single letter: A, B, C, or D."
)


def _load_gpqa_messages_from_csv() -> list[list[list[Message]]]:
    url = "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    messages_list: list[list[Message]] = []
    reader = csv.DictReader(io.StringIO(resp.text))
    for ex in reader:
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
            ]
        )
    if not messages_list:
        raise RuntimeError("Failed to load GPQA messages: no rows found from source")
    return [messages_list]


def _coerce_content_to_str(
    content: str | list[ChatCompletionContentPartTextParam] | None,
) -> str:
    if isinstance(content, list):
        return "".join([getattr(p, "text", str(p)) for p in content])
    return str(content or "")


def _extract_abcd_letter(text: str) -> str | None:
    if not text:
        return None
    m = re.search(r"\b([ABCD])\b", text.upper())
    return m.group(1) if m else None


_GPQA_INPUT_MESSAGES = _load_gpqa_messages_from_csv()


def _strip_gt_messages(msgs: list[Message]) -> list[Message]:
    result: list[Message] = []
    for m in msgs:
        content_str = _coerce_content_to_str(m.content)
        if not (m.role == "system" and content_str.startswith("__GT__:")):
            result.append(m)
    return result


class GPQAStripGTRolloutProcessor(RolloutProcessor):
    """Preprocess rows to set ground_truth and remove __GT__ messages, then delegate to SingleTurnRolloutProcessor."""

    def __init__(self):
        super().__init__()
        self.single_turn_processor = SingleTurnRolloutProcessor()

    def __call__(self, rows: list[EvaluationRow], config: RolloutProcessorConfig) -> list[asyncio.Task[EvaluationRow]]:
        """Preprocess rows and delegate to SingleTurnRolloutProcessor."""
        processed: list[EvaluationRow] = []

        for r in rows:
            gt_tokens: list[str] = []
            for m in r.messages:
                if m.role == "system":
                    content_str = _coerce_content_to_str(m.content)
                    if content_str.startswith("__GT__:"):
                        gt_tokens.append(content_str)
            if gt_tokens:
                gt_val = gt_tokens[-1].split(":", 1)[1].strip()
                r.ground_truth = gt_val
                filtered: list[Message] = []
                for m in r.messages:
                    if m.role == "system":
                        content_str = _coerce_content_to_str(m.content)
                        if content_str.startswith("__GT__:"):
                            continue
                    filtered.append(m)
                r.messages = filtered
            processed.append(r)

        # Delegate to SingleTurnRolloutProcessor
        return self.single_turn_processor(processed, config)


@evaluation_test(
    input_messages=_GPQA_INPUT_MESSAGES,
    completion_params=[
        {"extra_body": {"reasoning_effort": "low"}, "model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b"}
    ],
    rollout_processor=GPQAStripGTRolloutProcessor(),
    aggregation_method="mean",
    passed_threshold=0.6,
    num_runs=8,
    mode="pointwise",
)
def test_gpqa_pointwise(row: EvaluationRow) -> EvaluationRow:
    assistant_msgs = [m for m in row.messages if m.role == "assistant"]
    raw_content = assistant_msgs[-1].content if assistant_msgs else ""
    content_str = _coerce_content_to_str(raw_content)

    pred = _extract_abcd_letter(content_str)
    # GPQA diamond CSV constructs options so that the correct answer is always A
    gt = "A"

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
