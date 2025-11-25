from eval_protocol.data_loader.dynamic_data_loader import DynamicDataLoader
from eval_protocol.models import EvaluationRow, Message, EvaluateResult
from eval_protocol.pytest import evaluation_test
from typing import List

def generator() -> list[EvaluationRow]:
    return [EvaluationRow(messages=[Message(role="user", content="What is 2 + 2?")]) for _ in range(2)]

@evaluation_test(
    data_loaders=DynamicDataLoader(
        generators=[generator],
    ),
    mode="all",
)
def test_data_loader_stable_row_id_with_same_content(rows: List[EvaluationRow]) -> List[EvaluationRow]:
    """Test that the row id is stable even when the data loader is called multiple times."""
    row_ids = set()
    for row in rows:
        row_ids.add(row.input_metadata.row_id)
        row.evaluation_result = EvaluateResult(score=0.0, reason="Dummy evaluation result")
    assert len(row_ids) == 2
    return rows
