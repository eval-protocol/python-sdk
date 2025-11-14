import os
import logging
import time
from typing import List, Optional

from peewee import CharField, Model, SqliteDatabase, FloatField
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
        self._logger = logging.getLogger(__name__)

        class BaseModel(Model):
            class Meta:
                database = self._db

        class EvaluationRow(BaseModel):  # type: ignore
            rollout_id = CharField(unique=True)
            data = JSONField()
            updated_at = FloatField(default=lambda: time.time())

        self._EvaluationRow = EvaluationRow

        self._db.connect()
        # Use safe=True to avoid errors when tables/indexes already exist
        self._db.create_tables([EvaluationRow], safe=True)
        # Attempt to add updated_at column for existing installations
        try:
            columns = {c.name for c in self._db.get_columns(self._EvaluationRow._meta.table_name)}
            if "updated_at" not in columns:
                self._db.execute_sql(
                    f'ALTER TABLE "{self._EvaluationRow._meta.table_name}" ADD COLUMN "updated_at" REAL'
                )
                # Backfill with current time
                now_ts = time.time()
                self._EvaluationRow.update(updated_at=now_ts).execute()
        except Exception:
            # Best-effort; ignore if migration not needed or fails
            pass

    @property
    def db_path(self) -> str:
        return self._db_path

    def upsert_row(self, data: dict) -> None:
        rollout_id = data["execution_metadata"]["rollout_id"]
        if rollout_id is None:
            raise ValueError("execution_metadata.rollout_id is required to upsert a row")

        with self._db.atomic("EXCLUSIVE"):
            if self._EvaluationRow.select().where(self._EvaluationRow.rollout_id == rollout_id).exists():
                self._EvaluationRow.update(data=data, updated_at=time.time()).where(
                    self._EvaluationRow.rollout_id == rollout_id
                ).execute()
            else:
                self._EvaluationRow.create(rollout_id=rollout_id, data=data, updated_at=time.time())

    def read_rows(self, rollout_id: Optional[str] = None) -> List[dict]:
        # Build base query
        if rollout_id is None:
            model_query = self._EvaluationRow.select().order_by(self._EvaluationRow.updated_at.desc())
        else:
            model_query = self._EvaluationRow.select().where(self._EvaluationRow.rollout_id == rollout_id)

        # Log SQL for debugging
        try:
            sql_text, sql_params = model_query.sql()
            self._logger.debug(
                "[SQLITE_READ_ROWS] db=%s sql=%s params=%s", self._db_path, sql_text, sql_params
            )
        except Exception as e:
            self._logger.debug("[SQLITE_READ_ROWS] Failed to render SQL for debug: %s", e)

        # Execute and collect results
        results = list(model_query.dicts())

        # Debug: summarize results
        try:
            count = len(results)
            sample = results[:3]
            sample_rollout_ids = []
            sample_updated = []
            for r in sample:
                # r is a row dict with keys: rollout_id, data, updated_at
                rid = r.get("rollout_id")
                # updated_at may be missing on very old rows; guard accordingly
                up_at = r.get("updated_at", None)
                # Prefer rollout_id from nested data if available
                try:
                    rid_nested = r.get("data", {}).get("execution_metadata", {}).get("rollout_id")
                    if rid_nested:
                        rid = rid_nested
                except Exception:
                    pass
                sample_rollout_ids.append(str(rid))
                sample_updated.append(up_at)
            self._logger.debug(
                "[SQLITE_READ_ROWS] fetched_rows=%d sample_rollout_ids=%s sample_updated_at=%s",
                count,
                sample_rollout_ids,
                sample_updated,
            )
        except Exception as e:
            self._logger.debug("[SQLITE_READ_ROWS] Failed to summarize results for debug: %s", e)

        return [result["data"] for result in results]

    def delete_row(self, rollout_id: str) -> int:
        return self._EvaluationRow.delete().where(self._EvaluationRow.rollout_id == rollout_id).execute()

    def delete_all_rows(self) -> int:
        return self._EvaluationRow.delete().execute()
