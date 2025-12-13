import os
from typing import TYPE_CHECKING, List, Optional

from eval_protocol.dataset_logger.dataset_logger import LOG_EVENT_TYPE, DatasetLogger
from eval_protocol.dataset_logger.evaluation_row_store import EvaluationRowStore
from eval_protocol.directory_utils import find_eval_protocol_dir
from eval_protocol.event_bus import event_bus
from eval_protocol.event_bus.logger import logger

if TYPE_CHECKING:
    from eval_protocol.models import EvaluationRow


class DatasetLoggerAdapter(DatasetLogger):
    """
    Dataset logger that uses the configured storage backend.

    The storage backend is selected based on the EP_STORAGE environment variable:
    - "tinydb" (default): Uses TinyDB with JSON file storage
    - "sqlite": Uses SQLite with peewee ORM
    """

    def __init__(self, db_path: Optional[str] = None, store: Optional[EvaluationRowStore] = None):
        eval_protocol_dir = find_eval_protocol_dir()
        if db_path is not None and store is not None:
            raise ValueError("Provide only one of db_path or store, not both.")
        if store is not None:
            self.db_path = store.db_path
            self._store = store
        else:
            # Import here to avoid circular imports
            from eval_protocol.dataset_logger import _get_default_db_filename, get_evaluation_row_store

            default_db = _get_default_db_filename()
            self.db_path = db_path if db_path is not None else os.path.join(eval_protocol_dir, default_db)
            self._store = get_evaluation_row_store(self.db_path)

    def log(self, row: "EvaluationRow") -> None:
        data = row.model_dump(exclude_none=True, mode="json")
        rollout_id = data.get("execution_metadata", {}).get("rollout_id", "unknown")
        logger.debug(f"[EVENT_BUS_EMIT] Starting to log row with rollout_id: {rollout_id}")

        self._store.upsert_row(data=data)
        logger.debug(f"[EVENT_BUS_EMIT] Successfully stored row in database for rollout_id: {rollout_id}")

        try:
            from eval_protocol.models import EvaluationRow as EvalRow

            logger.debug(f"[EVENT_BUS_EMIT] Emitting event '{LOG_EVENT_TYPE}' for rollout_id: {rollout_id}")
            event_bus.emit(LOG_EVENT_TYPE, EvalRow(**data))
            logger.debug(f"[EVENT_BUS_EMIT] Successfully emitted event for rollout_id: {rollout_id}")
        except Exception as e:
            # Avoid breaking storage due to event emission issues
            logger.error(f"[EVENT_BUS_EMIT] Failed to emit row_upserted event for rollout_id {rollout_id}: {e}")
            pass

    def read(self, rollout_id: Optional[str] = None) -> List["EvaluationRow"]:
        from eval_protocol.models import EvaluationRow

        results = self._store.read_rows(rollout_id=rollout_id)
        return [EvaluationRow(**data) for data in results]
