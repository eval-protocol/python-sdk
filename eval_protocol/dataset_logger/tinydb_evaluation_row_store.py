import os
from typing import List, Optional

from tinydb import Query, TinyDB

from eval_protocol.dataset_logger.evaluation_row_store import EvaluationRowStore


class TinyDBEvaluationRowStore(EvaluationRowStore):
    """
    TinyDB-based evaluation row store.

    Stores data as plain JSON files, which are human-readable and
    don't suffer from SQLite's binary format corruption issues.
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
        self._table.upsert(data, Row.execution_metadata.rollout_id == rollout_id)

    def read_rows(self, rollout_id: Optional[str] = None) -> List[dict]:
        if rollout_id is not None:
            Row = Query()
            return list(self._table.search(Row.execution_metadata.rollout_id == rollout_id))
        return list(self._table.all())

    def delete_row(self, rollout_id: str) -> int:
        Row = Query()
        removed = self._table.remove(Row.execution_metadata.rollout_id == rollout_id)
        return len(removed)

    def delete_all_rows(self) -> int:
        count = len(self._table)
        self._table.truncate()
        return count
