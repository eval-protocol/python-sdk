import os
from typing import List, Optional

from peewee import CharField, Model, SqliteDatabase
from playhouse.sqlite_ext import JSONField

from eval_protocol.models import EvaluationRow


class SqliteEvaluationRowStore:
    """
    Lightweight reusable SQLite store for evaluation rows.

    Stores arbitrary row data as JSON keyed by a unique string `rollout_id`.
    """

    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._db_path = db_path
        self._db = SqliteDatabase(self._db_path, pragmas={"journal_mode": "wal"})

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

        with self._db.atomic("EXCLUSIVE"):
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
