from eval_protocol.pytest import evaluation_test
import os
from eval_protocol.models import EvaluateResult, EvaluationRow, InputMetadata
from typing import List
from typing import Dict, Any


def jsonl_dataset_adapter(rows: List[Dict[str, Any]]) -> List[EvaluationRow]:
    eval_rows: List[EvaluationRow] = []
    for row in rows:
        eval_rows.append(
            EvaluationRow(
                messages=row["conversation_messages"],
                input_metadata=InputMetadata(
                    row_id=str(row["prompt_id"]),
                ),
                evaluation_result=EvaluateResult(score=row["final_score"]),
            )
        )
    return eval_rows


@evaluation_test(
    input_dataset=[os.path.join(os.path.dirname(__file__), "data", "import_data.jsonl")],
    dataset_adapter=jsonl_dataset_adapter,
)
def test_import_logs(row: EvaluationRow) -> EvaluationRow:
    """
    The existence of this test ensures that importing external logs from
    arbitrary jsonl files works by just specifying input_dataset to file path
    and a dataset_adapter function.
    """
    return row
