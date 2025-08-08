import os
from typing import TYPE_CHECKING, List, Optional

from peewee import CharField, Model, SqliteDatabase
from playhouse.sqlite_ext import JSONField

from eval_protocol.dataset_logger.dataset_logger import DatasetLogger
from eval_protocol.dataset_logger.directory_utils import find_eval_protocol_dir

if TYPE_CHECKING:
    from eval_protocol.models import EvaluationRow


class SqliteDatasetLoggerAdapter(DatasetLogger):
    def __init__(self, db_path: Optional[str] = None):
        eval_protocol_dir = find_eval_protocol_dir()
        self.db_path = os.path.join(eval_protocol_dir, "logs.db")
        db = SqliteDatabase(self.db_path)

        class BaseModel(Model):
            class Meta:
                database = db

        class EvaluationRow(BaseModel):
            row_id = CharField(unique=True)
            data = JSONField()

        self.EvaluationRow = EvaluationRow

        db.connect()
        db.create_tables([EvaluationRow])

    def log(self, row: "EvaluationRow") -> None:
        row_id = row.input_metadata.row_id
        data = row.model_dump(exclude_none=True, mode="json")
        # if row_id already exists, update the row
        if self.EvaluationRow.select().where(self.EvaluationRow.row_id == row_id).exists():
            self.EvaluationRow.update(data=data).where(self.EvaluationRow.row_id == row_id).execute()
        else:
            self.EvaluationRow.create(row_id=row_id, data=data)

    def read(self, row_id: Optional[str] = None) -> List["EvaluationRow"]:
        from eval_protocol.models import EvaluationRow

        if row_id is None:
            query = self.EvaluationRow.select().dicts()
        else:
            query = self.EvaluationRow.select().dicts().where(self.EvaluationRow.row_id == row_id)
        results = list(query)
        return [EvaluationRow(**result["data"]) for result in results]
