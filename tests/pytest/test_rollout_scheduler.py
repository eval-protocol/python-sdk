from eval_protocol.pytest import evaluation_test, SingleTurnRolloutProcessor
from eval_protocol.models import EvaluationRow, Message, EvaluateResult, InputMetadata
from typing import List

@evaluation_test(
    completion_params=[{"model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b"}],
    input_rows=[
        [
            EvaluationRow(
                messages=[
                    Message(role="system", content=f"You are a helpful assistant, and this is row {i}"),
                    Message(role="user", content="What is the capital of France?"),
                ],
                input_metadata=InputMetadata(row_id=f"row-{i}"),
            )
            for i in range(10)
        ]
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    num_runs=4,
    mode="pointwise",
)
def test_rollout_scheduler(row: EvaluationRow) -> EvaluationRow:
    row.evaluation_result = EvaluateResult(score=0.5, reason="Dummy evaluation result")
    return row


@evaluation_test(
    completion_params=[{"model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b"}],
    input_rows=[
        [
            EvaluationRow(
                messages=[
                    Message(role="system", content=f"You are a helpful assistant, and this is row {i}"),
                    Message(role="user", content="What is the capital of France?"),
                ],
                input_metadata=InputMetadata(row_id=f"row-{i}"),
            )
            for i in range(10)
        ]
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    num_runs=4,
    mode="groupwise",
)
def test_rollout_scheduler_groupwise(rows: List[EvaluationRow]) -> List[EvaluationRow]:
    for i,row in enumerate(rows):
        row.evaluation_result = EvaluateResult(score=0.1 * i, reason="Dummy evaluation result")
    return rows