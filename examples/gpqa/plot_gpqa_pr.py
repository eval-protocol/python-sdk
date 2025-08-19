import csv
import math
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from eval_protocol.dataset_logger import default_logger
from eval_protocol.models import EvaluationRow, Message


def _get_last_assistant(row: EvaluationRow) -> Optional[Message]:
    assistants = [m for m in row.messages if m.role == "assistant"]
    return assistants[-1] if assistants else None


def _get_gt_letter(row: EvaluationRow) -> str:
    # GPQA diamond GT is A; prefer explicit ground_truth if present
    try:
        if getattr(row, "ground_truth", None):
            g = str(row.ground_truth).strip().upper()
            if g in {"A", "B", "C", "D"}:
                return g
    except Exception:
        pass
    return "A"


def _get_letter_logprobs(row: EvaluationRow) -> Optional[Dict[str, float]]:
    last = _get_last_assistant(row)
    if not last:
        return None
    cps = getattr(last, "control_plane_step", None)
    if not isinstance(cps, dict):
        return None
    lps = cps.get("letter_logprobs")
    if isinstance(lps, dict):
        # Ensure keys are A-D only
        return {k: float(v) for k, v in lps.items() if k in {"A", "B", "C", "D"}}
    return None


def _normalize_probs_from_logprobs(lp: Dict[str, float]) -> Dict[str, float]:
    # Ensure we have entries for all A-D; fill missing with a small logprob
    letters = ["A", "B", "C", "D"]
    if not lp:
        return {}
    min_lp = min(lp.values()) if lp else -20.0
    filled = {k: lp.get(k, min_lp - 5.0) for k in letters}
    exps = {k: math.exp(v) for k, v in filled.items()}
    s = sum(exps.values()) or 1.0
    return {k: exps[k] / s for k in letters}


def _precision_recall(points: List[Tuple[float, int]]) -> Tuple[List[float], List[float]]:
    """
    Given a list of (score, y_true) pairs, compute PR by ranking by score desc
    and accumulating TP/FP.
    """
    # Sort by score desc
    points_sorted = sorted(points, key=lambda x: x[0], reverse=True)
    total_pos = sum(1 for _, y in points_sorted if y == 1)
    if total_pos == 0:
        return [0.0], [0.0]

    precisions, recalls = [], []
    tp, fp = 0, 0
    for i, (_, y) in enumerate(points_sorted, start=1):
        if y == 1:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / i)
        recalls.append(tp / total_pos)
    return precisions, recalls


def _interp_monotonic_precision(precisions: List[float]) -> List[float]:
    # Interpolated (non-increasing) precision used in AP computations
    if not precisions:
        return []
    out = precisions[:]
    max_so_far = 0.0
    # Traverse from end to start, taking running max
    for i in range(len(out) - 1, -1, -1):
        if out[i] > max_so_far:
            max_so_far = out[i]
        out[i] = max_so_far
    return out


def main(
    rollout_id: Optional[str] = None,
    output_path: str = "outputs/gpqa_pr_curve.png",
    csv_out: Optional[str] = None,
    csv_pr_out: Optional[str] = None,
) -> None:
    rows: List[EvaluationRow] = default_logger.read(rollout_id=rollout_id)
    # Filter for GPQA runs and rows that have extracted letter logprobs
    filtered: List[EvaluationRow] = []
    for r in rows:
        try:
            nm = ((r.eval_metadata.name if r.eval_metadata else "") or "").lower()
            if "gpqa" not in nm:
                continue
        except Exception:
            continue

        if _get_letter_logprobs(r):
            filtered.append(r)

    if not filtered:
        raise SystemExit("No GPQA rows with letter logprobs found. Run the GPQA eval first.")

    pairs: List[Tuple[float, int]] = []
    per_row: List[Dict[str, object]] = []
    for r in filtered:
        lps = _get_letter_logprobs(r)
        if not lps:
            continue
        probs = _normalize_probs_from_logprobs(lps)
        if not probs:
            continue
        # Confidence is the max probability among A-D
        pred_letter = max(probs.items(), key=lambda x: x[1])[0]
        score = probs[pred_letter]
        gt = _get_gt_letter(r)
        y_true = 1 if pred_letter == gt else 0
        pairs.append((score, y_true))
        per_row.append(
            {
                "rollout_id": getattr(r.execution_metadata, "rollout_id", None),
                "row_id": getattr(r.input_metadata, "row_id", None),
                "pred": pred_letter,
                "gt": gt,
                "prob_A": probs.get("A"),
                "prob_B": probs.get("B"),
                "prob_C": probs.get("C"),
                "prob_D": probs.get("D"),
                "conf": score,
                "y_true": y_true,
            }
        )

    if not pairs:
        raise SystemExit("Found GPQA rows but none with valid A logprob + pred letter; cannot compute PR.")

    precision_raw, recall = _precision_recall(pairs)
    precision_interp = _interp_monotonic_precision(precision_raw)

    # Optional CSVs for debugging
    if csv_out:
        os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "rollout_id",
                    "row_id",
                    "pred",
                    "gt",
                    "prob_A",
                    "prob_B",
                    "prob_C",
                    "prob_D",
                    "conf",
                    "y_true",
                ],
            )
            w.writeheader()
            w.writerows(per_row)

    if csv_pr_out:
        os.makedirs(os.path.dirname(csv_pr_out) or ".", exist_ok=True)
        with open(csv_pr_out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["rank", "precision_raw", "precision_interp", "recall"])
            for i, (p_raw, p_interp, r_) in enumerate(zip(precision_raw, precision_interp, recall), start=1):
                w.writerow([i, p_raw, p_interp, r_])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(6, 5))
    # Plot interpolated precision for a monotonic PR curve
    plt.plot(recall, precision_interp, label="GPQA (interp)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall using A token confidence")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved PR curve to {output_path} with {len(pairs)} points.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot GPQA PR curve from logged rows")
    parser.add_argument("--rollout-id", help="Filter by rollout_id", default=None)
    parser.add_argument("--out", help="Output image path", default="outputs/gpqa_pr_curve.png")
    parser.add_argument("--csv-out", help="Write per-row scores to CSV", default=None)
    parser.add_argument("--csv-pr-out", help="Write PR curve points to CSV", default=None)
    args = parser.parse_args()
    main(rollout_id=args.rollout_id, output_path=args.out, csv_out=args.csv_out, csv_pr_out=args.csv_pr_out)
