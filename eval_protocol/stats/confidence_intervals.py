from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from ..models import EvaluationRow


def _default_question_id(row: EvaluationRow) -> str:
    """Best-effort stable question identifier across repeats.

    Prefers `row.input_metadata.row_id` (which is set once and preserved across deep copies),
    and falls back to the last user message content when not available.
    """
    # Prefer explicit row_id if present
    try:
        if row.input_metadata is not None and getattr(row.input_metadata, "row_id", None):
            return str(row.input_metadata.row_id)
    except Exception:
        pass

    # Fallback: use last user message content
    try:
        user_msgs = [m.content for m in row.messages if getattr(m, "role", None) == "user"]
        if user_msgs and user_msgs[-1]:
            return str(user_msgs[-1])
    except Exception:
        pass

    # Final fallback: use Python id for uniqueness within this process
    return f"row-{id(row)}"


def compute_fixed_set_mu_ci(
    rows: List[EvaluationRow],
    *,
    z_value: float = 1.96,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Compute the benchmark-conditional 95% CI for the mean accuracy μ on a fixed item set.

    This treats questions/items as fixed and repeats as within-item Bernoulli draws.
    For each question i with m_i repeats and s_i successes, the per-question mean is
    ybar_i = s_i / m_i. The estimator of μ is the average of per-question means:
        mu_hat = (1/Q) * sum_i ybar_i.

    The plug-in standard error for the CI of μ uses the within-item variances only:
        Var(mu_hat) ≈ (1/Q^2) * sum_i [ ybar_i (1 - ybar_i) / (m_i - 1) ], for m_i >= 2.

    Notes:
    - When m_i == 1, the unbiased correction is undefined. In that case we fall back to
      ybar_i (1 - ybar_i) / m_i as a conservative estimate. GPQA typically has m_i >= 2.
    - Scores are taken from `row.evaluation_result.score` when available and numeric.

    Returns:
        (mu_hat, ci_low, ci_high, standard_error). Returns (None, None, None, None) if insufficient data.
    """
    if not rows:
        return None, None, None, None

    # Group scores by question id
    question_to_scores: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        try:
            er = getattr(r, "evaluation_result", None)
            if er is None:
                continue
            score = getattr(er, "score", None)
            if score is None:
                continue
            # Ensure numeric float
            s_val = float(score)
            if math.isnan(s_val):
                continue
            qid = _default_question_id(r)
            question_to_scores[qid].append(s_val)
        except Exception:
            # Skip malformed rows
            continue

    Q = len(question_to_scores)
    if Q == 0:
        return None, None, None, None

    # Compute per-question means and the plug-in variance contribution
    ybars: List[float] = []
    var_terms: List[float] = []
    for scores in question_to_scores.values():
        m_i = len(scores)
        if m_i == 0:
            continue
        ybar_i = sum(scores) / m_i
        ybars.append(ybar_i)
        # Unbiased within-item variance estimate for Bernoulli mean
        if m_i >= 2:
            var_terms.append(ybar_i * (1.0 - ybar_i) / (m_i - 1))
        else:
            # Conservative fallback when only a single repeat exists
            var_terms.append(ybar_i * (1.0 - ybar_i) / m_i)

    if not ybars:
        return None, None, None, None

    mu_hat = sum(ybars) / len(ybars)

    # Standard error for CI of μ
    se_sq = sum(var_terms) / (Q * Q)
    standard_error = math.sqrt(se_sq) if se_sq > 0.0 else 0.0

    margin = z_value * standard_error
    ci_low = max(0.0, mu_hat - margin)
    ci_high = min(1.0, mu_hat + margin)

    return float(mu_hat), float(ci_low), float(ci_high), float(standard_error)
