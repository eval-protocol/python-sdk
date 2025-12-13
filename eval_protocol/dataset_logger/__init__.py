import os

from eval_protocol.dataset_logger.dataset_logger import DatasetLogger
from eval_protocol.dataset_logger.evaluation_row_store import EvaluationRowStore


def get_evaluation_row_store(db_path: str) -> EvaluationRowStore:
    """
    Factory to get the configured storage backend.

    Uses EP_STORAGE environment variable to select backend:
    - "tinydb" (default): Uses TinyDB with JSON file storage
    - "sqlite": Uses SQLite with peewee ORM

    Args:
        db_path: Path to the database file

    Returns:
        EvaluationRowStore implementation
    """
    storage_type = os.getenv("EP_STORAGE", "tinydb").lower()

    if storage_type == "sqlite":
        from eval_protocol.dataset_logger.sqlite_evaluation_row_store import SqliteEvaluationRowStore

        return SqliteEvaluationRowStore(db_path)
    else:
        from eval_protocol.dataset_logger.tinydb_evaluation_row_store import TinyDBEvaluationRowStore

        return TinyDBEvaluationRowStore(db_path)


def _get_default_db_filename() -> str:
    """Get the default database filename based on storage backend."""
    storage_type = os.getenv("EP_STORAGE", "tinydb").lower()
    return "logs.db" if storage_type == "sqlite" else "logs.json"


def _get_default_logger():
    """Get the default logger based on configuration."""
    # Allow disabling logger to avoid environment-specific constraints in simple CLI runs.
    if os.getenv("DISABLE_EP_SQLITE_LOG", "0").strip() == "1":

        class _NoOpLogger(DatasetLogger):
            def log(self, row):
                return None

            def read(self, rollout_id=None):
                return []

        return _NoOpLogger()

    # Import here to avoid circular imports
    from eval_protocol.dataset_logger.dataset_logger_adapter import DatasetLoggerAdapter

    return DatasetLoggerAdapter()


# Lazy property that creates the logger only when accessed
class _LazyLogger(DatasetLogger):
    def __init__(self):
        self._logger: DatasetLogger | None = None

    def _get_logger(self):
        if self._logger is None:
            self._logger = _get_default_logger()
        return self._logger

    def log(self, row):
        return self._get_logger().log(row)

    def read(self, rollout_id=None):
        return self._get_logger().read(rollout_id)


default_logger: DatasetLogger = _LazyLogger()
