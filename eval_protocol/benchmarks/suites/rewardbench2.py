from typing import List

from typing import Dict

from datasets import load_dataset

from eval_protocol.benchmarks.registry import export_benchmark
from eval_protocol.models import EvaluateResult, EvaluationRow, Message, MetricResult
from eval_protocol.pytest.evaluation_test import evaluation_test

import asyncio
import os
from openai import OpenAI
from eval_protocol.pytest.types import RolloutProcessorConfig


# RB2 v2 4-way ranking prompt (adapted for EP)
PROMPT_V2_SYSTEM = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by four AI assistants to the user question displayed below. "
    "You should choose the assistant that follows the user's instructions and answers the user's question best. Your evaluation should consider "
    "factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by "
    "comparing the four responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were "
    "presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names "
    "of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "
    '"[[A]]" if assistant A is best, "[[B]]" if assistant B is best, "[[C]]" if assistant C is best, and "[[D]]" if assistant D is best.'
)


PROMPT_V2_USER_TEMPLATE = (
    "[User Question]\n{question}\n\n"
    "[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n"
    "[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]\n\n"
    "[The Start of Assistant C's Answer]\n{answer_c}\n[The End of Assistant C's Answer]\n\n"
    "[The Start of Assistant D's Answer]\n{answer_d}\n[The End of Assistant D's Answer]"
)


def _build_rewardbench2_non_ties_messages() -> List[List[Message]]:
    """Load RewardBench 2 (non-Ties) and construct LMJ prompts without shuffling.

    Returns a list where each element is a list of Messages (system+user) for a single evaluation row.
    """
    # Allow a light-weight build by default; override with EP_RB2_BUILD_MAX_ROWS=none to construct all
    _build_cap = os.getenv("EP_RB2_BUILD_MAX_ROWS", "200")
    build_max_rows: int | None
    if _build_cap.strip().lower() == "none":
        build_max_rows = None
    else:
        try:
            build_max_rows = int(_build_cap)
        except ValueError:
            build_max_rows = 200

    ds = load_dataset("allenai/reward-bench-2", split="test")
    # Filter out Ties subset for initial scope
    ds = ds.filter(lambda ex: ex.get("subset", "").lower() != "ties")
    if build_max_rows is not None:
        ds = ds.select(range(min(build_max_rows, len(ds))))

    rows: List[List[Message]] = []
    for ex in ds:
        question = ex.get("prompt", "")
        chosen = ex.get("chosen", []) or []
        rejected = ex.get("rejected", []) or []
        # Require at least 1 chosen and 3 rejected for 4-way judging
        if not isinstance(chosen, list) or not isinstance(rejected, list):
            continue
        if len(chosen) < 1 or len(rejected) < 3:
            continue

        answer_a = str(chosen[0])
        answer_b = str(rejected[0])
        answer_c = str(rejected[1])
        answer_d = str(rejected[2])

        user_content = PROMPT_V2_USER_TEMPLATE.format(
            question=str(question),
            answer_a=answer_a,
            answer_b=answer_b,
            answer_c=answer_c,
            answer_d=answer_d,
        )

        messages = [
            Message(role="system", content=PROMPT_V2_SYSTEM),
            Message(role="user", content=user_content),
        ]
        rows.append(messages)

    return rows


def _parse_verdict(text: str) -> str:
    """Return one of {'A','B','C','D','error'} based on bracketed verdict tags.
    Matches RewardBench v2 parsing behavior.
    """
    if not text:
        return "error"
    if "[[A]]" in text:
        return "A"
    if "[[B]]" in text:
        return "B"
    if "[[C]]" in text:
        return "C"
    if "[[D]]" in text:
        return "D"
    return "error"


_INPUT_MESSAGES = _build_rewardbench2_non_ties_messages()


def _subset_for_row(messages: List[Message]) -> str:
    # Infer subset by reloading dataset index; for speed we prebuild a map lazily once.
    # We key by the exact 4-way concatenation to avoid collisions.
    static_index: Dict[str, str] = getattr(_subset_for_row, "_idx", {})  # type: ignore[attr-defined]
    if not static_index:
        ds = load_dataset("allenai/reward-bench-2", split="test").filter(
            lambda ex: ex.get("subset", "").lower() != "ties"
        )
        idx: Dict[str, str] = {}
        for ex in ds:
            question = str(ex.get("prompt", ""))
            chosen = (ex.get("chosen", []) or [])
            rejected = (ex.get("rejected", []) or [])
            if not isinstance(chosen, list) or not isinstance(rejected, list):
                continue
            if len(chosen) < 1 or len(rejected) < 3:
                continue
            key = "\n".join(
                [
                    question,
                    str(chosen[0]),
                    str(rejected[0]),
                    str(rejected[1]),
                    str(rejected[2]),
                ]
            )
            idx[key] = str(ex.get("subset", ""))
        setattr(_subset_for_row, "_idx", idx)  # type: ignore[attr-defined]
        static_index = idx
    # reconstruct key from messages (same order we built user template)
    user_text = next((m.content for m in messages if m.role == "user"), "") or ""
    try:
        # extract in the same way we formatted: find segments
        import re

        q = re.search(r"\[User Question\]\n([\s\S]*?)\n\n\[The Start of Assistant A's Answer\]", user_text)
        a = re.search(r"\[The Start of Assistant A's Answer\]\n([\s\S]*?)\n\[The End of Assistant A's Answer\]", user_text)
        b = re.search(r"\[The Start of Assistant B's Answer\]\n([\s\S]*?)\n\[The End of Assistant B's Answer\]", user_text)
        c = re.search(r"\[The Start of Assistant C's Answer\]\n([\s\S]*?)\n\[The End of Assistant C's Answer\]", user_text)
        d = re.search(r"\[The Start of Assistant D's Answer\]\n([\s\S]*?)\n\[The End of Assistant D's Answer\]", user_text)
        key = "\n".join([*(g.group(1) if g else "" for g in [q, a, b, c, d])])
    except (Exception,):
        key = ""
    return static_index.get(key, "Unknown")


async def _call_openai_chat(messages: List[Dict], model: str, base_url: str | None) -> str:
    client_kwargs = {}
    if base_url:
        client_kwargs["base_url"] = base_url
    api_key = os.environ.get("FIREWORKS_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if api_key:
        client_kwargs["api_key"] = api_key
    client = OpenAI(**client_kwargs)
    resp = client.chat.completions.create(model=model, messages=messages, temperature=0, max_tokens=2048)
    return resp.choices[0].message.content or ""


async def rb2_non_ties_openai_rollout(rows: List[EvaluationRow], config: RolloutProcessorConfig):
    cp = config.completion_params
    model = cp.get("model")
    base_url = cp.get("base_url")
    max_concurrent = getattr(config, "max_concurrent_rollouts", 4) or 4
    sem = asyncio.Semaphore(max_concurrent)

    async def _one(r: EvaluationRow) -> EvaluationRow:
        async with sem:
            user_msgs = [m for m in r.messages if m.role == "user"]
            sys_msgs = [m for m in r.messages if m.role == "system"]
            msgs: List[Dict] = []
            if sys_msgs:
                msgs.append({"role": "system", "content": sys_msgs[-1].content})
            msgs.append({"role": "user", "content": user_msgs[-1].content if user_msgs else ""})
            try:
                content = await _call_openai_chat(msgs, model, base_url)
            except Exception:
                content = ""
            r.messages.append(Message(role="assistant", content=content))
            return r

    tasks = [asyncio.create_task(_one(row)) for row in rows]
    for t in asyncio.as_completed(tasks):
        yield await t


@export_benchmark("rewardbench2_lmaj")
@evaluation_test(
    # Use input_messages so we can construct exact LMJ prompts and skip shuffling deterministically.
    input_messages=_INPUT_MESSAGES,
    completion_params=[
        {
            "model": "accounts/fireworks/models/gpt-oss-120b",
            "base_url": "https://api.fireworks.ai/inference/v1",
            "temperature": 0,
        }
    ],
    rollout_processor=rb2_non_ties_openai_rollout,
    aggregation_method="mean",
    passed_threshold=None,
    num_runs=1,
    # Keep the default small for quick iteration; override with EP_MAX_DATASET_ROWS=None to run all
    max_dataset_rows=40,
    max_concurrent_rollouts=4,
    mode="pointwise",
)
def test_rewardbench2_lmaj_pointwise(row: EvaluationRow) -> EvaluationRow:
    """RewardBench 2 non‑Ties, LM‑as‑a‑judge, 4‑way ranking, no shuffling (A is ground truth)."""
    assistant_msgs = [m for m in row.messages if m.role == "assistant"]
    content = assistant_msgs[-1].content if assistant_msgs else ""
    winner = _parse_verdict(str(content) if content is not None else "")

    # No shuffling: the correct answer is always A
    if winner == "A":
        score = 1.0
    elif winner in {"B", "C", "D"}:
        score = 0.0
    else:
        # Treat unparsable verdicts as a soft tie (same convention as RB2 scripts)
        score = 0.25

    # Capture per-subset signal by inferring subset label from prompt text
    subset_name = _subset_for_row(row.messages)
    metrics = {
        "non_ties_accuracy": MetricResult(
            score=1.0 if winner == "A" else 0.0,
            is_score_valid=True,
            reason=(
                "Judge picked A"
                if winner == "A"
                else (
                    "Judge picked non‑A"
                    if winner in {"B", "C", "D"}
                    else "Unparsable verdict; assigned 0.25 to overall score"
                )
            ),
        ),
        f"subset::{subset_name}": MetricResult(
            score=1.0 if winner == "A" else 0.0,
            is_score_valid=True,
            reason="per-subset accuracy",
        ),
    }

    row.evaluation_result = EvaluateResult(
        score=score,
        reason=("Correct" if score == 1.0 else ("Incorrect" if score == 0.0 else "Unparsable verdict")),
        is_score_valid=True,
        metrics=metrics,
    )
    return row


