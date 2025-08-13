from typing import Any, Dict, List

from eval_protocol.models import EvaluationRow, Message


def word_count_to_evaluation_row(data: List[Dict[str, Any]]) -> List[EvaluationRow]:
    """Convert gsm8k dataset format to EvaluationRow for word_count evaluation."""
    return [
        EvaluationRow(
            messages=[Message(role="user", content=row["user_query"])], ground_truth=row["ground_truth_for_eval"]
        )
        for row in data
    ]
