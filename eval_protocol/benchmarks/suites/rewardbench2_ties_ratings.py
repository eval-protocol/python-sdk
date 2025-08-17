from collections import defaultdict
from typing import Dict, List, Tuple
import os
import logging

from datasets import load_dataset

from eval_protocol.benchmarks.registry import export_benchmark
from eval_protocol.models import EvaluateResult, EvaluationRow, Message, MetricResult
import asyncio
from openai import OpenAI
from eval_protocol.pytest.types import RolloutProcessorConfig
from eval_protocol.pytest.evaluation_test import evaluation_test


RATINGS_PROMPT_TIES = (
    """
### Task Description
Please act as an impartial judge and evaluate the quality of the response provided by an
AI assistant to the user query displayed below.

Notes:
1- Your evaluation should consider factors such as the helpfulness, relevance, and accuracy of the response, but need not consider depth or level of detail of the response.
2- Begin your evaluation by providing a short explanation.
3- Be as objective as possible. After providing your explanation, please rate the response on a scale of 1 to 10. For your rating, only give a number between 1 and 10 (inclusive), do not use any markdown, and do not put any text after your final rating.

[Query]
{prompt}

[Response]
{completion}

[Your judgement]
"""
).strip()


_TIES_ROW_META: List[Tuple[str, bool]] = []  # parallel to _TIES_INPUT_ROWS: (group_id, is_correct)

# Suppress noisy third-party INFO logs during runs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("eval_protocol.mcp.execution.policy").setLevel(logging.WARNING)


def _build_ties_rating_rows() -> List[List[Message]]:
    """Expand RewardBench 2 Ties subset into per-answer rating prompts.

    Returns a list of Messages per rating request (system is empty; single user prompt).
    """
    ds = load_dataset("allenai/reward-bench-2", split="test")
    ds = ds.filter(lambda ex: ex.get("subset", "").lower() == "ties")

    rows: List[List[Message]] = []
    global _TIES_ROW_META
    _TIES_ROW_META = []
    for ex in ds:
        prompt = str(ex.get("prompt", ""))
        chosen = list(ex.get("chosen", []) or [])
        rejected = list(ex.get("rejected", []) or [])
        num_correct = int(ex.get("num_correct", 0) or 0)
        group_id_raw = str(ex.get("id", ""))
        # Normalize group id to "ref:<pid>" / "tied:<pid>" by dropping any extra suffix (e.g., ":0")
        try:
            parts = group_id_raw.split(":")
            sample_type = parts[0]
            prompt_id = parts[1] if len(parts) > 1 else ""
            group_id = f"{sample_type}:{prompt_id}"
        except Exception:
            group_id = group_id_raw

        all_answers: List[Tuple[str, bool]] = []
        for ans in chosen:
            all_answers.append((str(ans), True))
        for ans in rejected:
            all_answers.append((str(ans), False))

        assert num_correct <= len(all_answers)

        for i, (answer_text, _is_correct) in enumerate(all_answers):
            user_text = RATINGS_PROMPT_TIES.format(prompt=prompt, completion=answer_text)
            rows.append([Message(role="user", content=user_text)])
            # map row → (group_id, is_correct) using dataset-provided id
            _TIES_ROW_META.append((group_id, _is_correct))

    return rows


def _parse_trailing_rating(text: str) -> int:
    """Extract rating 1–10 from the end of the text, tolerating common variants like '8/10'.

    Returns -1 on failure (to mirror RB2's behavior when unparsable).
    """
    if not text:
        return -1
    import re

    s = str(text).strip()
    # 1) Strict trailing int
    m = re.search(r"\b(10|[1-9])\b\s*$", s)
    if m:
        return int(m.group(1))
    # 2) Trailing 'X/10'
    m = re.search(r"\b(10|[1-9])\s*/\s*10\s*$", s)
    if m:
        return int(m.group(1))
    # 3) Last line heuristic: parse last integer 1–10 on the last non-empty line
    last_line = next((ln for ln in s.splitlines()[::-1] if ln.strip()), "")
    m = re.search(r"\b(10|[1-9])\b(?:\s*/\s*10)?\s*$", last_line.strip())
    if m:
        return int(m.group(1))
    # 4) Fallback: any 1–10 near string end (allow trailing punctuation)
    m = re.search(r"(10|[1-9])(?:\s*/\s*10)?\s*[\)\].,:;!?]*\s*$", s)
    if m:
        return int(m.group(1))
    return -1


def _compute_prompt_stats(samples: List[Tuple[bool, float]]) -> Tuple[bool, float | None, float | None]:
    correct_scores = [s for is_corr, s in samples if is_corr]
    incorrect_scores = [s for is_corr, s in samples if not is_corr]
    if not correct_scores or not incorrect_scores:
        return False, None, None
    best_correct = max(correct_scores)
    worst_correct = min(correct_scores)
    best_incorrect = max(incorrect_scores)
    different_correct_margin = best_correct - worst_correct if len(correct_scores) > 1 else None
    correct_incorrect_margin = worst_correct - best_incorrect
    accurate = correct_incorrect_margin > 0
    return accurate, different_correct_margin, correct_incorrect_margin


def _compute_ties_overall(samples_by_group: Dict[str, List[Tuple[bool, float]]]) -> float:
    import numpy as np

    ref_stats: Dict[str, Tuple[bool, float | None, float | None]] = {}
    tied_stats: Dict[str, Tuple[bool, float | None, float | None]] = {}

    for gid, samples in samples_by_group.items():
        try:
            sample_type, _ = gid.split(":", 1)
        except ValueError:
            sample_type = "tied"
        stats = _compute_prompt_stats(samples)
        if sample_type == "ref":
            ref_stats[gid] = stats
        else:
            tied_stats[gid] = stats

    ref_accuracy = float(np.mean([s[0] for s in ref_stats.values()])) if ref_stats else 0.0
    tied_accuracy = float(np.mean([s[0] for s in tied_stats.values()])) if tied_stats else 0.0

    all_ids = set(k.split(":", 1)[1] for k in tied_stats.keys()) & set(
        k.split(":", 1)[1] for k in ref_stats.keys()
    )

    def _arr(vals: List[float | None]) -> List[float]:
        return [float(v) for v in vals if v is not None]

    diff_corr_margin = _arr([tied_stats[f"tied:{pid}"][1] for pid in all_ids if f"tied:{pid}" in tied_stats])
    corr_incorrect_ties = _arr([tied_stats[f"tied:{pid}"][2] for pid in all_ids if f"tied:{pid}" in tied_stats])
    corr_incorrect_ref = _arr([ref_stats[f"ref:{pid}"][2] for pid in all_ids if f"ref:{pid}" in ref_stats])

    L = min(len(diff_corr_margin), len(corr_incorrect_ties), len(corr_incorrect_ref))
    try:
        print(
            f"RB2 Ties DBG | ref={len(ref_stats)} tied={len(tied_stats)} shared={len(all_ids)} "
            f"len_dcm={len(diff_corr_margin)} len_cit={len(corr_incorrect_ties)} len_cir={len(corr_incorrect_ref)} L={L}"
        )
    except Exception:
        pass
    if L == 0:
        return 0.0
    diff_corr_margin = diff_corr_margin[:L]
    corr_incorrect_ties = corr_incorrect_ties[:L]
    corr_incorrect_ref = corr_incorrect_ref[:L]

    diff_corr_margin_arr = np.array(diff_corr_margin)
    corr_incorrect_ties_arr = np.array(corr_incorrect_ties)
    corr_incorrect_ref_arr = np.array(corr_incorrect_ref)

    correctness_preferred = float(np.mean(corr_incorrect_ties_arr > diff_corr_margin_arr))
    correctness_preferred_hard = float(
        np.mean(np.minimum(corr_incorrect_ref_arr, corr_incorrect_ties_arr) > diff_corr_margin_arr)
    )
    # Safe margin calculation: avoid divide-by-zero; when margin is 0, treat contribution as 0
    safe_den = diff_corr_margin_arr.copy()
    safe_den[safe_den == 0] = np.inf
    ratio = np.minimum(corr_incorrect_ref_arr, corr_incorrect_ties_arr) / safe_den
    margin_scores = np.tanh(ratio - 1)
    margin_scores = np.nan_to_num(margin_scores, nan=0.0, posinf=0.0, neginf=0.0)
    correctness_margin_score = float(np.mean(margin_scores))

    overall = (
        0.30 * tied_accuracy
        + 0.30 * ref_accuracy
        + 0.20 * correctness_preferred
        + 0.20 * correctness_preferred_hard
        + 0.01 * correctness_margin_score
    )
    # Clamp to [0, 1] for metric validity in EP
    return float(max(0.0, min(1.0, overall)))


_TIES_INPUT_ROWS = _build_ties_rating_rows()


async def _call_openai_chat_single(user_text: str, model: str, base_url: str | None, temperature: float, max_tokens: int) -> str:
    client_kwargs = {}
    if base_url:
        client_kwargs["base_url"] = base_url
    # Use FIREWORKS_API_KEY with OpenAI client param name
    fw_key = os.environ.get("FIREWORKS_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if fw_key:
        client_kwargs["api_key"] = fw_key
    client = OpenAI(**client_kwargs)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_text}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


async def rb2_ties_openai_rollout_processor(rows: List[EvaluationRow], config: RolloutProcessorConfig):
    """Custom rollout that calls Fireworks via OpenAI client directly (RB2 parity)."""
    # Extract params (mirror RB2 defaults)
    cp = config.completion_params
    model = cp.get("model")
    base_url = cp.get("base_url")
    temperature = cp.get("temperature", 0)
    max_tokens = cp.get("max_tokens", 1024)

    max_concurrent = getattr(config, "max_concurrent_rollouts", 8) or 8
    sem = asyncio.Semaphore(max_concurrent)

    async def _one(row: EvaluationRow) -> EvaluationRow:
        async with sem:
            try:
                user_msgs = [m for m in row.messages if m.role == "user"]
                user_text = user_msgs[-1].content if user_msgs else ""
                content = await _call_openai_chat_single(
                    user_text=user_text,
                    model=model,
                    base_url=base_url,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                row.messages.append(Message(role="assistant", content=content))
            except Exception:
                row.messages.append(Message(role="assistant", content=""))
            return row

    tasks = [asyncio.create_task(_one(r)) for r in rows]
    for t in asyncio.as_completed(tasks):
        yield await t


# No text-based index needed; we rely on _TIES_ROW_META built alongside messages.


@export_benchmark("rewardbench2_ties_ratings")
@evaluation_test(
    input_messages=_TIES_INPUT_ROWS,
    completion_params=[
        {
            # Use Fireworks OpenAI-compatible endpoint explicitly like RB2
            "model": "accounts/fireworks/models/gpt-oss-120b",
            "base_url": "https://api.fireworks.ai/inference/v1",
            "temperature": 0,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "max_tokens": 1024,
        }
    ],
    rollout_processor=rb2_ties_openai_rollout_processor,
    aggregation_method="mean",
    passed_threshold=None,
    num_runs=1,
    max_dataset_rows=None,
    max_concurrent_rollouts=8,
    mode="batch",
)
def test_rewardbench2_ties_ratings(rows: List[EvaluationRow]) -> List[EvaluationRow]:
    """Ratings-based evaluation for RewardBench 2 Ties subset (batch, no metadata in prompts)."""
    # Build samples grouped by dataset id using precomputed row metadata
    samples_by_group: Dict[str, List[Tuple[bool, float]]] = defaultdict(list)

    def _parse_from_user_prompt(s: str) -> Tuple[str, str]:
        # Extract prompt and completion from the template we used
        # [Query]\n{prompt}\n ... [Response]\n{completion}\n
        try:
            q_tag = "[Query]\n"
            r_tag = "\n\n[Response]\n"
            j_tag = "\n\n[Your judgement]"
            q_start = s.index(q_tag) + len(q_tag)
            r_start = s.index(r_tag, q_start)
            prompt = s[q_start:r_start]
            c_start = r_start + len(r_tag)
            j_start = s.index(j_tag, c_start)
            completion = s[c_start:j_start]
            return prompt, completion
        except ValueError:
            return "", ""

    # Collect ratings
    per_row_info: Dict[int, Tuple[str, bool, float, float]] = {}
    num_ok, num_bad = 0, 0
    bad_examples: List[Tuple[int, str]] = []
    for idx_row, r in enumerate(rows):
        assistant_msgs = [m for m in r.messages if m.role == "assistant"]
        content = assistant_msgs[-1].content if assistant_msgs else ""
        rating = _parse_trailing_rating(str(content) if content is not None else "")
        if idx_row < len(_TIES_ROW_META):
            group_id, is_correct = _TIES_ROW_META[idx_row]
            samples_by_group[group_id].append((is_correct, float(rating)))
            per_row_info[idx_row] = (group_id, is_correct, float(rating), 0.0)
        if rating == -1:
            num_bad += 1
            if len(bad_examples) < 3 and isinstance(content, str):
                bad_examples.append((idx_row, content[-200:]))
        else:
            num_ok += 1

    overall = _compute_ties_overall(samples_by_group) if samples_by_group else 0.0
    # Debug: compute component metrics to compare with RB2
    try:
        import numpy as np
        ref_stats_dbg: Dict[str, Tuple[bool, float | None, float | None]] = {}
        tied_stats_dbg: Dict[str, Tuple[bool, float | None, float | None]] = {}
        for gid, entries in samples_by_group.items():
            st = _compute_prompt_stats(entries)
            if gid.startswith("ref:"):
                ref_stats_dbg[gid] = st
            else:
                tied_stats_dbg[gid] = st
        ref_acc_dbg = float(np.mean([s[0] for s in ref_stats_dbg.values()])) if ref_stats_dbg else 0.0
        tied_acc_dbg = float(np.mean([s[0] for s in tied_stats_dbg.values()])) if tied_stats_dbg else 0.0
        all_ids_dbg = set(k.split(":", 1)[1] for k in tied_stats_dbg.keys()) & set(
            k.split(":", 1)[1] for k in ref_stats_dbg.keys()
        )
        def _arr_dbg(vals: List[float | None]) -> List[float]:
            return [float(v) for v in vals if v is not None]
        dcm = _arr_dbg([tied_stats_dbg.get(f"tied:{pid}", (False, None, None))[1] for pid in all_ids_dbg])
        cit = _arr_dbg([tied_stats_dbg.get(f"tied:{pid}", (False, None, None))[2] for pid in all_ids_dbg])
        cir = _arr_dbg([ref_stats_dbg.get(f"ref:{pid}", (False, None, None))[2] for pid in all_ids_dbg])
        Ld = min(len(dcm), len(cit), len(cir))
        if Ld > 0:
            dcm = dcm[:Ld]; cit = cit[:Ld]; cir = cir[:Ld]
            dcm_arr = np.array(dcm); cit_arr = np.array(cit); cir_arr = np.array(cir)
            cp = float(np.mean(cit_arr > dcm_arr))
            cph = float(np.mean(np.minimum(cir_arr, cit_arr) > dcm_arr))
            safe_den = dcm_arr.copy(); safe_den[safe_den == 0] = np.inf
            margin = np.tanh(np.minimum(cir_arr, cit_arr) / safe_den - 1.0)
            margin = np.nan_to_num(margin, nan=0.0, posinf=0.0, neginf=0.0)
            cms = float(np.mean(margin))
        else:
            cp = cph = cms = 0.0
        # Overall averages of raw ratings by correctness
        all_correct = [s for gid, ent in samples_by_group.items() for (is_c, s) in ent if is_c]
        all_incorrect = [s for gid, ent in samples_by_group.items() for (is_c, s) in ent if not is_c]
        avg_c = float(np.mean(all_correct)) if all_correct else 0.0
        avg_i = float(np.mean(all_incorrect)) if all_incorrect else 0.0
        print(
            f"RB2 Ties METRICS | ref_acc={ref_acc_dbg:.3f} tied_acc={tied_acc_dbg:.3f} cp={cp:.3f} cph={cph:.3f} cms={cms:.3f} avg_c={avg_c:.3f} avg_i={avg_i:.3f}"
        )
    except Exception:
        pass
    try:
        total = num_ok + num_bad
        rate = (num_ok / total) if total else 0.0
        print(f"RB2 Ties PARSE | ok={num_ok} bad={num_bad} ok_rate={rate:.3f}")
        if bad_examples:
            for i, (idx, tail) in enumerate(bad_examples):
                print(f"RB2 Ties BAD_EX[{i}] | idx={idx} tail=...{tail}")
    except Exception:
        pass

    # Compute per-group stats: accuracy flag and max ratings
    group_stats: Dict[str, Tuple[bool, float, float, float]] = {}
    for gid, entries in samples_by_group.items():
        accurate, diff_corr_margin, corr_incorrect_margin = _compute_prompt_stats(entries)
        # compute group max
        gmax = max([s for _, s in entries]) if entries else 0.0
        group_stats[gid] = (accurate, gmax, diff_corr_margin or 0.0, corr_incorrect_margin or 0.0)

    # Attach results
    out: List[EvaluationRow] = []
    for idx_row, r in enumerate(rows):
        assistant_msgs = [m for m in r.messages if m.role == "assistant"]
        content = assistant_msgs[-1].content if assistant_msgs else ""
        rating_val = _parse_trailing_rating(str(content) if content is not None else "")
        gid = None
        is_corr = False
        gmax = 0.0
        grp_acc = False
        if idx_row in per_row_info:
            gid, is_corr, rating_f, _ = per_row_info[idx_row]
            stats = group_stats.get(gid)
            if stats:
                grp_acc, gmax, _, _ = stats
        # Per-row score: set to overall ties score so EP aggregated mean matches RewardBench.
        # Row-level diagnostics are still exposed via metrics below.
        per_row_score = float(overall)
        # Additional fine-grained signal: whether this specific answer is a top-rated correct one
        is_top_correct = bool(is_corr and rating_val != -1 and abs(float(rating_val) - float(gmax)) < 1e-9)
        r.evaluation_result = EvaluateResult(
            score=float(per_row_score),
            reason="RB2 Ties overall score (batch)",
            is_score_valid=True,
            metrics={
                "ties_overall": MetricResult(score=float(overall), is_score_valid=True, reason="subset overall"),
                "group_accuracy": MetricResult(score=(1.0 if grp_acc else 0.0), is_score_valid=True, reason="all correct > best wrong"),
                "raw_rating_valid": MetricResult(
                    score=1.0 if rating_val != -1 else 0.0,
                    is_score_valid=True,
                    reason="Parsed trailing 1-10" if rating_val != -1 else "Failed to parse rating",
                ),
                "is_top_correct": MetricResult(score=(1.0 if is_top_correct else 0.0), is_score_valid=True, reason="this answer top-rated & correct"),
            },
        )
        out.append(r)

    # Helpful debug print aligned with RB2 script output
    try:
        num_groups = len(samples_by_group)
        print(f"RB2 Ties | groups={num_groups} overall={overall:.6f}")
        # Print a few sample groups for inspection
        printed = 0
        for gid, entries in samples_by_group.items():
            if printed >= 4:
                break
            correct = sorted([s for is_c, s in entries if is_c])
            wrong = sorted([s for is_c, s in entries if not is_c])
            acc, gmax, dcm, cim = group_stats.get(gid, (False, 0.0, 0.0, 0.0))
            print(
                f"RB2 Ties GROUP | {gid} acc={int(acc)} minC={min(correct) if correct else 'NA'} maxC={max(correct) if correct else 'NA'} maxW={max(wrong) if wrong else 'NA'} dcm={dcm:.3f} cim={cim:.3f}"
            )
            printed += 1
    except (Exception,):
        pass
    return out


