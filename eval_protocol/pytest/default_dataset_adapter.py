from typing import Any, Dict, List

from eval_protocol.models import EvaluationRow


def default_dataset_adapter(rows: List[Dict[str, Any]]) -> List[EvaluationRow]:
    """
    Default dataset adapter that simply returns the rows as is.
    """
    return [EvaluationRow(**row) for row in rows]