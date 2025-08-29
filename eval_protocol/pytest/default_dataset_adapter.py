from typing import Any
from eval_protocol.models import EvaluationRow


def default_dataset_adapter(rows: list[dict[str, Any]]) -> list[EvaluationRow]:  # pyright: ignore[reportExplicitAny]
    """
    Default dataset adapter that simply returns the rows as is.
    """
    return [EvaluationRow(**row) for row in rows]  # pyright: ignore[reportAny]
