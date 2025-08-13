from eval_protocol.dataset_logger.sqlite_dataset_logger_adapter import SqliteDatasetLoggerAdapter
import os

# Allow disabling sqlite logger to avoid environment-specific constraints in simple CLI runs.
if os.getenv("EP_SQLITE_LOG", "0").strip() == "1":
    default_logger = SqliteDatasetLoggerAdapter()
else:
    class _NoOpLogger:
        def log(self, row):
            return None

        def read(self, rollout_id=None):
            return []

    default_logger = _NoOpLogger()
