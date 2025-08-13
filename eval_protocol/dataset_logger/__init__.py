import os

from eval_protocol.dataset_logger.dataset_logger import DatasetLogger
from eval_protocol.dataset_logger.sqlite_dataset_logger_adapter import SqliteDatasetLoggerAdapter

# Allow disabling sqlite logger to avoid environment-specific constraints in simple CLI runs.
if os.getenv("DISABLE_EP_SQLITE_LOG", "0").strip() != "1":
    default_logger = SqliteDatasetLoggerAdapter()
else:

    class _NoOpLogger(DatasetLogger):
        def log(self, row):
            return None

        def read(self, rollout_id=None):
            return []

    default_logger = _NoOpLogger()
