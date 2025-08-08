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
    def read(self, row_id: Optional[str] = None) -> List["EvaluationRow"]:
        """
        Retrieve EvaluationRow logs.

        Args:
            row_id (Optional[str]): If provided, filter logs by this row_id.

        Returns:
            List[EvaluationRow]: List of retrieved evaluation rows.
        """
        pass
