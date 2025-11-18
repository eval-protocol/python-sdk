"""
Example of using a rapidfuzz-based Python grader with OpenAI RFT via Eval Protocol.

We:
- Define a grading function over a duck-typed `row` that uses rapidfuzz.WRatio
- Wrap it in an @evaluation_test for normal eval usage
- Convert the grading function into a Python grader spec with
  `build_python_grader_from_evaluation_test`
"""

from typing import Any

from eval_protocol.integrations.openai_rft import build_python_grader_from_evaluation_test
from eval_protocol.models import EvaluateResult, EvaluationRow, Message
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.default_no_op_rollout_processor import NoOpRolloutProcessor


# Tiny inline demo dataset so this evaluation_test is runnable via pytest.
DEMO_ROWS = [
    EvaluationRow(
        messages=[
            Message(role="user", content="fuzzy wuzzy had no hair"),
            Message(role="assistant", content="fuzzy wuzzy was a bear"),
        ],
        ground_truth="fuzzy wuzzy had no hair",
    )
]


@evaluation_test(
    input_rows=[DEMO_ROWS],
    rollout_processor=NoOpRolloutProcessor(),
    aggregation_method="mean",
    mode="pointwise",
)
def rapidfuzz_eval(row: EvaluationRow, **kwargs: Any) -> EvaluationRow:
    """
    Example @evaluation_test that scores a row using rapidfuzz.WRatio and
    attaches an EvaluateResult.
    """
    # For EP evals, we compare the EvaluationRow's ground_truth to the last assistant message.
    reference = row.ground_truth

    assistant_msgs = [m for m in row.messages if m.role == "assistant"]
    last_assistant_content = assistant_msgs[-1].content if assistant_msgs else ""
    prediction = last_assistant_content if isinstance(last_assistant_content, str) else ""

    from rapidfuzz import fuzz, utils

    score = float(
        fuzz.WRatio(
            str(prediction),
            str(reference),
            processor=utils.default_process,
        )
        / 100.0
    )
    row.evaluation_result = EvaluateResult(score=score)
    return row


RAPIDFUZZ_PYTHON_GRADER_SPEC: dict = build_python_grader_from_evaluation_test(rapidfuzz_eval)
