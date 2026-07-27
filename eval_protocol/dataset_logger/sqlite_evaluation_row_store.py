import os
from typing import List, Optional

from peewee import CharField, Model, SqliteDatabase, fn, SQL
from playhouse.sqlite_ext import JSONField

from eval_protocol.event_bus.sqlite_event_bus_database import (
    SQLITE_HARDENED_PRAGMAS,
    check_and_repair_database,
    connect_with_retry,
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

        # Connect with retry logic that properly handles pragma execution failures
        connect_with_retry(self._db)
        # Use safe=True to avoid errors when tables/indexes already exist
        execute_with_sqlite_retry(lambda: self._db.create_tables([EvaluationRow], safe=True))

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

    def read_rows(
        self,
        rollout_id: Optional[str] = None,
        invocation_ids: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[dict]:
        """
        Read evaluation rows from the database with optional filtering.

        Args:
            rollout_id: Filter by a specific rollout_id (exact match)
            invocation_ids: Filter by a list of invocation_ids (rows matching any)
            limit: Maximum number of rows to return (most recent first)

        Returns:
            List of evaluation row data dictionaries
        """
        query = self._EvaluationRow.select()

        if rollout_id is not None:
            query = query.where(self._EvaluationRow.rollout_id == rollout_id)

        # Apply invocation_ids filter using JSON extraction
        # Note: This filters rows where data->'execution_metadata'->>'invocation_id' matches any of the provided IDs
        if invocation_ids is not None and len(invocation_ids) > 0:
            # Build a condition that matches any of the invocation_ids
            # Using SQLite JSON extraction: json_extract(data, '$.execution_metadata.invocation_id')
            invocation_conditions = []
            for inv_id in invocation_ids:
                invocation_conditions.append(
                    fn.json_extract(self._EvaluationRow.data, "$.execution_metadata.invocation_id") == inv_id
                )
            # Combine with OR
            if len(invocation_conditions) == 1:
                query = query.where(invocation_conditions[0])
            else:
                from functools import reduce
                from operator import or_

                combined_condition = reduce(or_, invocation_conditions)
                query = query.where(combined_condition)

        # Order by rowid descending to get most recent rows first
        query = query.order_by(SQL("rowid DESC"))

        if limit is not None:
            query = query.limit(limit)

        results = list(query.dicts())
        return [result["data"] for result in results]

    def delete_row(self, rollout_id: str) -> int:
        return self._EvaluationRow.delete().where(self._EvaluationRow.rollout_id == rollout_id).execute()

    def delete_all_rows(self) -> int:
        return self._EvaluationRow.delete().execute()
