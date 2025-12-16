import os
from typing import List, Optional

from peewee import CharField, Model, SqliteDatabase
from playhouse.sqlite_ext import JSONField

from eval_protocol.event_bus.sqlite_event_bus_database import (
    SQLITE_HARDENED_PRAGMAS,
    check_and_repair_database,
    execute_with_sqlite_retry,
)
from eval_protocol.models import EvaluationRow


class SqliteEvaluationRowStore:
    """
    Lightweight reusable SQLite store for evaluation rows.

    Stores arbitrary row data as JSON keyed by a unique string `rollout_id`.
    Uses hardened SQLite settings for concurrency safety.
    """

    def __init__(self, db_path: str, auto_repair: bool = True):
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._db_path = db_path

        # Check and optionally repair corrupted database
        check_and_repair_database(db_path, auto_repair=auto_repair)

        # Use hardened pragmas for concurrency safety
        self._db = SqliteDatabase(self._db_path, pragmas=SQLITE_HARDENED_PRAGMAS)

        class BaseModel(Model):
            class Meta:
                database = self._db

        class EvaluationRow(BaseModel):  # type: ignore
            rollout_id = CharField(unique=True)
            data = JSONField()

        self._EvaluationRow = EvaluationRow

        self._db.connect()
        # Use safe=True to avoid errors when tables/indexes already exist
        self._db.create_tables([EvaluationRow], safe=True)

    @property
    def db_path(self) -> str:
        return self._db_path

    def upsert_row(self, data: dict) -> None:
        rollout_id = data["execution_metadata"]["rollout_id"]
        if rollout_id is None:
            raise ValueError("execution_metadata.rollout_id is required to upsert a row")

        execute_with_sqlite_retry(lambda: self._do_upsert(rollout_id, data))

    def _do_upsert(self, rollout_id: str, data: dict) -> None:
        """Internal method to perform the actual upsert within a transaction."""
        # Use IMMEDIATE instead of EXCLUSIVE for better concurrency
        # IMMEDIATE acquires a reserved lock immediately but allows concurrent reads
        with self._db.atomic("IMMEDIATE"):
            if self._EvaluationRow.select().where(self._EvaluationRow.rollout_id == rollout_id).exists():
                self._EvaluationRow.update(data=data).where(self._EvaluationRow.rollout_id == rollout_id).execute()
            else:
                self._EvaluationRow.create(rollout_id=rollout_id, data=data)

    def read_rows(self, rollout_id: Optional[str] = None) -> List[dict]:
        if rollout_id is None:
            query = self._EvaluationRow.select().dicts()
        else:
            query = self._EvaluationRow.select().dicts().where(self._EvaluationRow.rollout_id == rollout_id)
        results = list(query)
        return [result["data"] for result in results]

    def delete_row(self, rollout_id: str) -> int:
        return self._EvaluationRow.delete().where(self._EvaluationRow.rollout_id == rollout_id).execute()

    def delete_all_rows(self) -> int:
        return self._EvaluationRow.delete().execute()
