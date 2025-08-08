import os
from typing import List, Optional

from peewee import CharField, Model, SqliteDatabase
from playhouse.sqlite_ext import JSONField


class SqliteEvaluationRowStore:
    """
    Lightweight reusable SQLite store for evaluation rows.

    Stores arbitrary row data as JSON keyed by a unique string `row_id`.
    """

    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._db_path = db_path
        self._db = SqliteDatabase(self._db_path)

        class BaseModel(Model):
            class Meta:
                database = self._db

        class EvaluationRow(BaseModel):  # type: ignore
            row_id = CharField(unique=True)
            data = JSONField()

        self._EvaluationRow = EvaluationRow

        self._db.connect()
        self._db.create_tables([EvaluationRow])

    @property
    def db_path(self) -> str:
        return self._db_path

    def upsert_row(self, row_id: str, data: dict) -> None:
        if self._EvaluationRow.select().where(self._EvaluationRow.row_id == row_id).exists():
            self._EvaluationRow.update(data=data).where(self._EvaluationRow.row_id == row_id).execute()
        else:
            self._EvaluationRow.create(row_id=row_id, data=data)

    def read_rows(self, row_id: Optional[str] = None) -> List[dict]:
        if row_id is None:
            query = self._EvaluationRow.select().dicts()
        else:
            query = self._EvaluationRow.select().dicts().where(self._EvaluationRow.row_id == row_id)
        results = list(query)
        return [result["data"] for result in results]

    def delete_row(self, row_id: Optional[str] = None) -> None:
        if row_id is None:
            self._EvaluationRow.delete().execute()
        else:
            self._EvaluationRow.delete().where(self._EvaluationRow.row_id == row_id).execute()
