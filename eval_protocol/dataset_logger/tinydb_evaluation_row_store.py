import json
import logging
import os
import time
from typing import List, Optional

from tinydb import Query, TinyDB
from tinyrecord.transaction import transaction

from eval_protocol.dataset_logger.evaluation_row_store import EvaluationRowStore

logger = logging.getLogger(__name__)


class TinyDBEvaluationRowStore(EvaluationRowStore):
    """
    TinyDB-based evaluation row store.

    Stores data as plain JSON files, which are human-readable and
    don't suffer from SQLite's binary format corruption issues.

    Uses tinyrecord for atomic transactions to handle concurrent access
    from multiple processes safely.
    """

    def __init__(self, db_path: str):
        # Handle case where db_path might be in the root directory
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._db_path = db_path
        self._db = self._open_db_with_retry()
        self._table = self._db.table("evaluation_rows")

    def _open_db_with_retry(self, max_retries: int = 3) -> TinyDB:
        """Open TinyDB with retry logic to handle transient JSON decode errors."""
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                return TinyDB(self._db_path)
            except json.JSONDecodeError as e:
                last_error = e
                logger.warning(f"TinyDB JSON decode error on attempt {attempt + 1}: {e}")
                # Wait a bit and retry - the file might be mid-write
                time.sleep(0.1 * (attempt + 1))
                # Try to recover by removing the corrupted file
                if attempt == max_retries - 1 and os.path.exists(self._db_path):
                    try:
                        logger.warning(f"Removing corrupted TinyDB file: {self._db_path}")
                        os.remove(self._db_path)
                        return TinyDB(self._db_path)
                    except Exception:
                        pass
        raise last_error if last_error else RuntimeError("Failed to open TinyDB")

    @property
    def db_path(self) -> str:
        return self._db_path

    def upsert_row(self, data: dict) -> None:
        rollout_id = data["execution_metadata"]["rollout_id"]
        if rollout_id is None:
            raise ValueError("execution_metadata.rollout_id is required to upsert a row")

        Row = Query()
        condition = Row.execution_metadata.rollout_id == rollout_id

        # tinyrecord doesn't support upsert directly, so we implement it manually
        # within a transaction for atomicity
        with transaction(self._table) as tr:
            # Clear cache to ensure fresh read in multi-process scenarios
            self._table.clear_cache()
            # Check if document exists
            existing = self._table.search(condition)
            if existing:
                # Update existing document
                tr.update(data, condition)
            else:
                # Insert new document
                tr.insert(data)

    def read_rows(self, rollout_id: Optional[str] = None) -> List[dict]:
        """Read rows with retry logic for transient JSON decode errors."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Clear cache to ensure fresh read in multi-process scenarios
                self._table.clear_cache()
                if rollout_id is not None:
                    Row = Query()
                    return list(self._table.search(Row.execution_metadata.rollout_id == rollout_id))
                return list(self._table.all())
            except json.JSONDecodeError as e:
                logger.warning(f"TinyDB JSON decode error on read attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))
                else:
                    # Return empty list on final failure rather than crash
                    logger.warning("Failed to read TinyDB after retries, returning empty list")
                    return []
        return []

    def delete_row(self, rollout_id: str) -> int:
        Row = Query()
        condition = Row.execution_metadata.rollout_id == rollout_id

        with transaction(self._table) as tr:
            # Clear cache to ensure fresh read in multi-process scenarios
            self._table.clear_cache()
            # Check if document exists before removal to get accurate count
            existing = self._table.search(condition)
            if existing:
                tr.remove(condition)
                return len(existing)
        return 0

    def delete_all_rows(self) -> int:
        count = len(self._table)
        self._table.truncate()
        return count
