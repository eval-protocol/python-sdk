import os
from turtle import st
from typing import TYPE_CHECKING, List, Optional

from eval_protocol.dataset_logger.dataset_logger import DatasetLogger
from eval_protocol.dataset_logger.directory_utils import find_eval_protocol_dir
from eval_protocol.dataset_logger.sqlite_evaluation_row_store import SqliteEvaluationRowStore

if TYPE_CHECKING:
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
        row_id = row.input_metadata.row_id
        data = row.model_dump(exclude_none=True, mode="json")
        self._store.upsert_row(row_id=row_id, data=data)

    def read(self, row_id: Optional[str] = None) -> List["EvaluationRow"]:
        from eval_protocol.models import EvaluationRow

        results = self._store.read_rows(row_id=row_id)
        return [EvaluationRow(**data) for data in results]
