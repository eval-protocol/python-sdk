from typing import Any, Dict, List, Optional

from eval_protocol.models import (
    EvaluateResult,
    EvaluationRow,
    Message,
    MetricResult,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
)
from eval_protocol.pytest.default_single_turn_rollout_process import (
    SingleTurnRolloutProcessor,
)
from eval_protocol.pytest.evaluation_test import evaluation_test
from eval_protocol.training import GEPATrainer
from eval_protocol.training.gepa_utils import build_reflection_lm

SYSTEM_PROMPT = (
    "You are a helpful math assistant. Please reason step by step, and put your final answer within \\boxed{...}."
)


def _coerce_content_to_str(
    content: str | list[ChatCompletionContentPartParam] | None,
) -> str:
    if isinstance(content, list):
        return "".join(
            getattr(p, "text", str(p)) if isinstance(p, ChatCompletionContentPartTextParam) else "" for p in content
        )
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


def _build_feedback_text(
    *,
    extracted_int: Optional[int],
    gt_int: Optional[int],
    is_valid: bool,
    raw_model_answer: str,
    ground_truth: Optional[str],
) -> str:
    """
    Build a feedback string similar in spirit to the GEPA `metric_with_feedback`.

    Cases:
    - Parse failure (model or gold): explain integer formatting and show correct answer.
    - Correct: "Your answer is correct. The correct answer is '...'."
    - Incorrect: "Your answer is incorrect. The correct answer is '...'."
    """
    correct_answer_display = str(gt_int if gt_int is not None else (ground_truth or ""))

    if not is_valid:
        # Could not parse either the model answer or the gold answer as an integer.
        feedback_text = (
            "The final answer must be a valid integer and nothing else. "
            f"You responded with '{raw_model_answer}', which couldn't be parsed as a python integer. "
            "Please ensure your answer is a valid integer without any additional text or formatting."
        )
        if correct_answer_display:
            feedback_text += f" The correct answer is '{correct_answer_display}'."
        return feedback_text

    if extracted_int == gt_int:
        return f"Your answer is correct. The correct answer is '{correct_answer_display}'."
    else:
        return f"Your answer is incorrect. The correct answer is '{correct_answer_display}'."

    # TODO: our dataset does not contain written solutions, so we cannot provide feedback on the solution. maybe need to add it later.
    # they're using https://huggingface.co/datasets/AI-MO/aimo-validation-aime


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

    feedback_text = _build_feedback_text(
        extracted_int=extracted_int,
        gt_int=gt_int,
        is_valid=is_valid,
        raw_model_answer=content_str,
        ground_truth=str(row.ground_truth),
    )

    row.evaluation_result = EvaluateResult(
        score=score,
        reason=feedback_text,
        is_score_valid=is_valid,
        metrics=metrics,
    )
    return row


if __name__ == "__main__":
    trainer = GEPATrainer(test_aime25_pointwise)
    reflection_lm = build_reflection_lm("gpt-5")

    optimized_program = trainer.train(
        num_threads=32,
        track_stats=True,
        reflection_minibatch_size=3,
        reflection_lm=reflection_lm,
    )

    print(trainer.evaluate(optimized_program))
