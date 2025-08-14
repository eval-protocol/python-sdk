import os
import random
from typing import List

import pytest

from eval_protocol.models import EvaluateResult, EvaluationRow, Message
from eval_protocol.pytest import NoOpRolloutProcessor, evaluation_test


# skip in CI since it will intentionally fail. This is useful for local generation of logs
@pytest.mark.skipif(os.getenv("CI") == "true", reason="Skipping flaky test in CI")
@evaluation_test(
    input_messages=[[Message(role="user", content="Return HEADS or TAILS at random.")]],
    completion_params=[{"model": "dummy/local-model"}],
    rollout_processor=NoOpRolloutProcessor(),
    mode="pointwise",
    num_runs=5,
)
def test_flaky_passes_sometimes(row: EvaluationRow) -> EvaluationRow:
    """
    A deliberately flaky evaluation that only passes occasionally.

    With num_runs=5 and a success probability of ~0.3 per run, the aggregated mean
    will clear the threshold (0.8) only rarely. Uses the no-op rollout to avoid any
    actual model calls.
    """
    # Stochastic score: 1.0 with 30% probability, else 0.0
    score = 1.0 if random.random() < 0.3 else 0.0
    row.evaluation_result = EvaluateResult(score=score, reason=f"stochastic={score}")
    return row
