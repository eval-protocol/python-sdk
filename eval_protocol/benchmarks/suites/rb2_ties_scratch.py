from collections import defaultdict
from typing import Dict, List, Tuple
import logging

from datasets import load_dataset

from eval_protocol.benchmarks.registry import export_benchmark
from eval_protocol.models import EvaluateResult, EvaluationRow, Message, MetricResult
from eval_protocol.pytest.evaluation_test import evaluation_test


logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("eval_protocol.mcp.execution.policy").setLevel(logging.WARNING)


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


def _parse_trailing_rating(text: str) -> int:
    if not text:
        return -1
    import re

    m = re.search(r"\b([1-9]|10)\b\s*$", str(text).strip())
    return int(m.group(1)) if m else -1


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
    return float(max(0.0, min(1.0, overall)))


def _build_rows_and_meta() -> Tuple[List[List[Message]], List[Tuple[str, bool]]]:
    ds = load_dataset("allenai/reward-bench-2", split="test")
    ds = ds.filter(lambda ex: ex.get("subset", "").lower() == "ties")

    rows: List[List[Message]] = []
    meta: List[Tuple[str, bool]] = []
    for ex in ds:
        prompt = str(ex.get("prompt", ""))
        chosen = list(ex.get("chosen", []) or [])
        rejected = list(ex.get("rejected", []) or [])
        num_correct = int(ex.get("num_correct", 0) or 0)
        group_id = str(ex.get("id", ""))  # already "ref:123" or "tied:123"

        all_answers: List[Tuple[str, bool]] = []
        for ans in chosen:
            all_answers.append((str(ans), True))
        for ans in rejected:
            all_answers.append((str(ans), False))

        for answer_text, is_correct in all_answers:
            user_text = RATINGS_PROMPT_TIES.format(prompt=prompt, completion=answer_text)
            rows.append([Message(role="user", content=user_text)])
            meta.append((group_id, is_correct))
    return rows, meta


_INPUT_ROWS, _ROW_META = _build_rows_and_meta()


@export_benchmark("rb2_ties_scratch")
@evaluation_test(
    input_messages=_INPUT_ROWS,
    completion_params=[{"model": "dummy/local", "temperature": 0, "max_tokens": 1}],
    # Use default no-op rollout_processor; we compute metrics in-function
    aggregation_method="mean",
    passed_threshold=None,
    num_runs=1,
    max_dataset_rows=200,
    max_concurrent_rollouts=4,
    mode="batch",
)
def test_rb2_ties_scratch(rows: List[EvaluationRow]) -> List[EvaluationRow]:
    """Scratch Ties suite: use a dummy judge (10 for correct, 1 for incorrect) to validate grouping+scoring.

    This avoids any external model calls and lets us verify we compute RB2's Ties score correctly.
    """
    samples_by_group: Dict[str, List[Tuple[bool, float]]] = defaultdict(list)

    # Populate ratings using dummy judge
    for idx_row, r in enumerate(rows):
        group_id, is_correct = _ROW_META[idx_row]
        rating = 10.0 if is_correct else 1.0
        samples_by_group[group_id].append((is_correct, rating))

    overall = _compute_ties_overall(samples_by_group) if samples_by_group else 0.0

    # Compute per-group accuracy and max
    group_stats: Dict[str, Tuple[bool, float]] = {}
    for gid, entries in samples_by_group.items():
        accurate, _, _ = _compute_prompt_stats(entries)
        gmax = max([s for _, s in entries]) if entries else 0.0
        group_stats[gid] = (accurate, gmax)

    out: List[EvaluationRow] = []
    for idx_row, r in enumerate(rows):
        group_id, is_correct = _ROW_META[idx_row]
        accurate, gmax = group_stats.get(group_id, (False, 0.0))
        rating_val = 10.0 if is_correct else 1.0
        is_top_correct = bool(is_correct and abs(float(rating_val) - float(gmax)) < 1e-9)
        r.evaluation_result = EvaluateResult(
            score=float(overall),
            reason="RB2 Ties overall score (scratch dummy)",
            is_score_valid=True,
            metrics={
                "ties_overall": MetricResult(score=float(overall), is_score_valid=True, reason="subset overall"),
                "group_accuracy": MetricResult(score=(1.0 if accurate else 0.0), is_score_valid=True, reason="all correct > best wrong"),
                "raw_rating_valid": MetricResult(score=1.0, is_score_valid=True, reason="Dummy rating"),
                "is_top_correct": MetricResult(score=(1.0 if is_top_correct else 0.0), is_score_valid=True, reason="dummy top correct"),
            },
        )
        out.append(r)

    try:
        num_groups = len(samples_by_group)
        print(f"RB2 Ties | groups={num_groups} overall={overall:.6f}")
    except Exception:
        pass
    return out


