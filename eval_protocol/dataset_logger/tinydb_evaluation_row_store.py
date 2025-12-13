import os
from typing import List, Optional

from tinydb import Query, TinyDB
from tinyrecord.transaction import transaction

from eval_protocol.dataset_logger.evaluation_row_store import EvaluationRowStore


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
        self._db = TinyDB(db_path)
        self._table = self._db.table("evaluation_rows")

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
            # Check if document exists
            existing = self._table.search(condition)
            if existing:
                # Update existing document
                tr.update(data, condition)
            else:
                # Insert new document
                tr.insert(data)

    def read_rows(self, rollout_id: Optional[str] = None) -> List[dict]:
        # Clear cache to ensure fresh read in multi-process scenarios
        self._table.clear_cache()
        if rollout_id is not None:
            Row = Query()
            return list(self._table.search(Row.execution_metadata.rollout_id == rollout_id))
        return list(self._table.all())

    def delete_row(self, rollout_id: str) -> int:
        Row = Query()
        with transaction(self._table) as tr:
            tr.remove(Row.execution_metadata.rollout_id == rollout_id)
        # Return count after removal (we don't have access to removed count in transaction)
        return 1

    def delete_all_rows(self) -> int:
        count = len(self._table)
        self._table.truncate()
        return count
