import os
from typing import List, Optional

from eval_protocol.dataset_logger.dataset_logger import DatasetLogger
from eval_protocol.dataset_logger.sqlite_dataset_logger_adapter import SqliteDatasetLoggerAdapter


# Allow disabling sqlite logger to avoid environment-specific constraints in simple CLI runs.
def _get_default_logger():
    if os.getenv("DISABLE_EP_SQLITE_LOG", "0").strip() != "1":
        return SqliteDatasetLoggerAdapter()
    else:

        class _NoOpLogger(DatasetLogger):
            def log(self, row):
                return None

            def read(self, rollout_id=None, invocation_ids=None, limit=None):
                return []

        return _NoOpLogger()


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

    def read(
        self,
        rollout_id: Optional[str] = None,
        invocation_ids: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ):
        return self._get_logger().read(rollout_id=rollout_id, invocation_ids=invocation_ids, limit=limit)


default_logger: DatasetLogger = _LazyLogger()
