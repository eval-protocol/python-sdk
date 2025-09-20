"""
Base adapter interface for Eval Protocol.
"""

from abc import ABC, abstractmethod
from typing import List

from eval_protocol.models import EvaluationRow


class BaseAdapter(ABC):
    """Abstract base class for all Eval Protocol adapters."""

    @abstractmethod
    def get_evaluation_rows(self, *args, **kwargs) -> List[EvaluationRow]:
        """Get evaluation rows from the data source."""
        pass

    def upload_scores(self, rows: List[EvaluationRow], model_name: str, mean_score: float) -> None:
        """Upload evaluation scores back to the data source for tracking and analysis."""
        pass

    def upload_score(self, row: EvaluationRow, model_name: str) -> None:
        """Upload evaluation score for a single row back to the data source."""
        pass
