from abc import ABC, abstractmethod
from typing import List, Optional


class EvaluationRowStore(ABC):
    """
    Abstract base class for evaluation row storage.

    Stores arbitrary row data as JSON keyed by a unique string `rollout_id`.
    Implementations can use different storage backends (SQLite, TinyDB, etc.)
    """

    @property
    @abstractmethod
    def db_path(self) -> str:
        """Return the path to the database file."""
        pass

    @abstractmethod
    def upsert_row(self, data: dict) -> None:
        """
        Insert or update a row by rollout_id.

        Args:
            data: Row data containing execution_metadata.rollout_id
        """
        pass

    @abstractmethod
    def read_rows(self, rollout_id: Optional[str] = None) -> List[dict]:
        """
        Read rows, optionally filtered by rollout_id.

        Args:
            rollout_id: If provided, filter to this specific rollout

        Returns:
            List of row data dictionaries
        """
        pass

    @abstractmethod
    def delete_row(self, rollout_id: str) -> int:
        """
        Delete a row by rollout_id.

        Args:
            rollout_id: The rollout_id to delete

        Returns:
            Number of rows deleted
        """
        pass

    @abstractmethod
    def delete_all_rows(self) -> int:
        """
        Delete all rows.

        Returns:
            Number of rows deleted
        """
        pass
