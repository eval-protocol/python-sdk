from typing import Any, Dict, List

from eval_protocol.models import EvaluationRow, Message


def gsm8k_to_evaluation_row(data: List[Dict[str, Any]]) -> List[EvaluationRow]:
    return [
        EvaluationRow(
            messages=[Message(role="user", content=row["user_query"])], ground_truth=row["ground_truth_for_eval"]
        )
        for row in data
    ]
