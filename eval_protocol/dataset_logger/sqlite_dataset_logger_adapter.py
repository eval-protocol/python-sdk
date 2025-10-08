import os
from typing import List, Optional

from eval_protocol.dataset_logger.dataset_logger import LOG_EVENT_TYPE, DatasetLogger
from eval_protocol.dataset_logger.sqlite_evaluation_row_store import SqliteEvaluationRowStore
from eval_protocol.directory_utils import find_eval_protocol_dir
from eval_protocol.event_bus import event_bus
from eval_protocol.event_bus.logger import logger
from eval_protocol.models import EvaluationRow


class SqliteDatasetLoggerAdapter(DatasetLogger):
    def __init__(self, db_path: Optional[str] = None, store: Optional[SqliteEvaluationRowStore] = None):
        eval_protocol_dir = find_eval_protocol_dir()
        if db_path is not None and store is not None:
            raise ValueError("Provide only one of db_path or store, not both.")
        if store is not None:
            self.db_path = store.db_path
            self._store = store
        else:
            self.db_path = db_path if db_path is not None else os.path.join(eval_protocol_dir, "logs.db")
            self._store = SqliteEvaluationRowStore(self.db_path)

    def log(self, row: "EvaluationRow") -> None:
        data = row.model_dump(exclude_none=True, mode="json")
        rollout_id = data.get("execution_metadata", {}).get("rollout_id", "unknown")
        logger.debug(f"[EVENT_BUS_EMIT] Starting to log row with rollout_id: {rollout_id}")

        self._store.upsert_row(data=data)
        logger.debug(f"[EVENT_BUS_EMIT] Successfully stored row in database for rollout_id: {rollout_id}")

        try:
            logger.debug(f"[EVENT_BUS_EMIT] Emitting event '{LOG_EVENT_TYPE}' for rollout_id: {rollout_id}")
            event_bus.emit(LOG_EVENT_TYPE, EvaluationRow(**data))
            logger.debug(f"[EVENT_BUS_EMIT] Successfully emitted event for rollout_id: {rollout_id}")
        except Exception as e:
            # Avoid breaking storage due to event emission issues
            logger.error(f"[EVENT_BUS_EMIT] Failed to emit row_upserted event for rollout_id {rollout_id}: {e}")
            pass

    def read(self, rollout_id: Optional[str] = None) -> List["EvaluationRow"]:
        from eval_protocol.models import EvaluationRow

        results = self._store.read_rows(rollout_id=rollout_id)
        return [EvaluationRow(**data) for data in results]
