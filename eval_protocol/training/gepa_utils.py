import os
from typing import Optional

import dspy
from dspy.clients.lm import LM
from dspy.primitives import Example, Prediction
from dspy.teleprompt.gepa.gepa_utils import DSPyTrace, ScoreWithFeedback
from dspy.teleprompt.gepa.gepa import GEPAFeedbackMetric

from eval_protocol.pytest.types import TestFunction
from eval_protocol.models import EvaluationRow, Message


REFLECTION_LM_CONFIGS = {
    "gpt-5": {
        "model": "gpt-5",
        "temperature": 1.0,
        "max_tokens": 32000,
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": "https://api.openai.com/v1",
    },
    "kimi-k2-instruct-0905": {
        "model": "accounts/fireworks/models/kimi-k2-instruct-0905",
        "temperature": 0.6,  # Kimi recommended temperature
        "max_tokens": 131000,
        "api_key": os.getenv("FIREWORKS_API_KEY"),
        "base_url": "https://api.fireworks.ai/inference/v1",
    },
}


def build_reflection_lm(reflection_lm_name: str) -> LM:
    reflection_lm_config = REFLECTION_LM_CONFIGS[reflection_lm_name]
    return dspy.LM(
        model=reflection_lm_config["model"],
        temperature=reflection_lm_config["temperature"],
        max_tokens=reflection_lm_config["max_tokens"],
        api_key=reflection_lm_config["api_key"],
        base_url=reflection_lm_config["base_url"],
    )


def gold_and_pred_to_row(gold: Example, pred: Prediction) -> EvaluationRow:
    """
    Convert a GEPA (gold, pred) pair into an EvaluationRow for an EP `@evaluation_test`.

    Assumptions (aligned with common DSPy usage):
    - `gold.answer` holds the ground-truth answer.
    - `pred.answer` holds the model's final answer text.
    """
    gt = gold.get("answer", None)
    ground_truth_str: Optional[str] = str(gt) if gt is not None else None

    content = pred.get("answer", "")

    return EvaluationRow(
        messages=[
            Message(role="assistant", content=str(content))
        ],  # TODO: for some evals, you might need system / user message too.
        ground_truth=ground_truth_str,
    )


def row_to_prediction(row: EvaluationRow) -> ScoreWithFeedback:
    """
    Convert an EvaluationRow into a GEPA-compatible ScoreWithFeedback
    (implemented as a dspy.Prediction subclass in dspy.teleprompt.gepa).
    """
    if row.evaluation_result is None:
        return dspy.Prediction(
            score=0.0,
            feedback="No evaluation_result was produced by the evaluation_test.",
        )

    score = float(row.evaluation_result.score or 0.0)
    feedback = row.evaluation_result.reason or f"This trajectory got a score of {score}."
    return dspy.Prediction(score=score, feedback=feedback)


def ep_test_to_gepa_metric(
    test_fn: TestFunction,
) -> GEPAFeedbackMetric:
    """
    Adapter: convert an EP-style `test_fn(row: EvaluationRow) -> EvaluationRow` into
    a GEPAFeedbackMetric-compatible callable.

    The resulting metric:
    - Constructs an EvaluationRow from (gold, pred) using a simple heuristic.
    - Applies the EP test_fn to populate `row.evaluation_result`.
    - Returns a dspy.Prediction(score, feedback) derived from that result.
    """

    def metric(
        gold: Example,
        pred: Prediction,
        trace: Optional[DSPyTrace] = None,
        pred_name: Optional[str] = None,
        pred_trace: Optional[DSPyTrace] = None,
    ) -> ScoreWithFeedback:
        row = gold_and_pred_to_row(gold, pred)

        evaluated_row: EvaluationRow = test_fn(row)  # pyright: ignore
        # TODO: this is problematic. for groupwise, we will have to extend this to handle list[EvaluationRow]

        return row_to_prediction(evaluated_row)

    return metric
