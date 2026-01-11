from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from eval_protocol.models import EvaluationRow

LOG_EVENT_TYPE = "log"


class DatasetLogger(ABC):
    """
    Abstract base class for logging EvaluationRow objects.
    Implementations should provide methods for storing and retrieving logs.
    """

    @abstractmethod
    def log(self, row: "EvaluationRow") -> None:
        """
        Store a single EvaluationRow log.

        Args:
            row (EvaluationRow): The evaluation row to log.
        """
        pass

    @abstractmethod
    def read(
        self,
        rollout_id: Optional[str] = None,
        invocation_ids: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List["EvaluationRow"]:
        """
        Retrieve EvaluationRow logs with optional filtering.

        Args:
            rollout_id (Optional[str]): If provided, filter logs by this rollout_id.
            invocation_ids (Optional[List[str]]): If provided, filter logs by these invocation_ids.
            limit (Optional[int]): If provided, limit the number of rows returned (most recent first).

        Returns:
            List[EvaluationRow]: List of retrieved evaluation rows.
        """
        pass
