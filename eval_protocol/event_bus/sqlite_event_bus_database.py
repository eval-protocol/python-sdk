import os
import time
from typing import Any, Callable, List, TypeVar
from uuid import uuid4

import backoff
from peewee import BooleanField, CharField, DatabaseError, DateTimeField, Model, OperationalError, SqliteDatabase
from playhouse.sqlite_ext import JSONField

from eval_protocol.event_bus.logger import logger


# Retry configuration for database operations
SQLITE_RETRY_MAX_TRIES = 5
SQLITE_RETRY_MAX_TIME = 30  # seconds


def _is_database_locked_error(e: Exception) -> bool:
    """Check if an exception is a database locked error."""
    error_str = str(e).lower()
    return "database is locked" in error_str or "locked" in error_str


T = TypeVar("T")


def execute_with_sqlite_retry(operation: Callable[[], T]) -> T:
    """
    Execute a database operation with exponential backoff retry on lock errors.

    Uses the backoff library for consistent retry behavior across the codebase.
    Retries only on OperationalError with "database is locked" message.

    Args:
        operation: A callable that performs the database operation

    Returns:
        The result of the operation

    Raises:
        OperationalError: If the operation fails after all retries
    """

    @backoff.on_exception(
        backoff.expo,
        OperationalError,
        max_tries=SQLITE_RETRY_MAX_TRIES,
        max_time=SQLITE_RETRY_MAX_TIME,
        giveup=lambda e: not _is_database_locked_error(e),
        jitter=backoff.full_jitter,
    )
    def _execute() -> T:
        return operation()

    return _execute()


# SQLite pragmas for hardened concurrency safety
SQLITE_HARDENED_PRAGMAS = {
    "journal_mode": "wal",  # Write-Ahead Logging for concurrent reads/writes
    "synchronous": "normal",  # Balance between safety and performance
    "busy_timeout": 30000,  # 30 second timeout for locked database
    "wal_autocheckpoint": 1000,  # Checkpoint every 1000 pages
    "cache_size": -64000,  # 64MB cache (negative = KB)
    "foreign_keys": 1,  # Enable foreign key constraints
    "temp_store": "memory",  # Store temp tables in memory
}


class DatabaseCorruptedError(Exception):
    """Raised when the database file is corrupted or not a valid SQLite database."""

    def __init__(self, db_path: str, original_error: Exception):
        self.db_path = db_path
        self.original_error = original_error
        super().__init__(f"Database file is corrupted: {db_path}. Original error: {original_error}")


def check_and_repair_database(db_path: str, auto_repair: bool = False) -> bool:
    """
    Check if a database file is valid and optionally repair it.

    Args:
        db_path: Path to the database file
        auto_repair: If True, automatically delete and recreate corrupted database

    Returns:
        True if database is valid or was repaired, False otherwise

    Raises:
        DatabaseCorruptedError: If database is corrupted and auto_repair is False
    """
    if not os.path.exists(db_path):
        return True  # New database, nothing to check

    try:
        # Try to open the database and run an integrity check
        test_db = SqliteDatabase(db_path, pragmas={"busy_timeout": 5000})
        test_db.connect()
        cursor = test_db.execute_sql("PRAGMA integrity_check")
        result = cursor.fetchone()
        test_db.close()

        if result and result[0] == "ok":
            return True
        else:
            logger.warning(f"Database integrity check failed for {db_path}: {result}")
            if auto_repair:
                _backup_and_remove_database(db_path)
                return True
            raise DatabaseCorruptedError(db_path, Exception(f"Integrity check failed: {result}"))

    except DatabaseError as e:
        error_str = str(e).lower()
        # Only treat specific SQLite corruption errors as corruption
        corruption_indicators = [
            "file is not a database",
            "database disk image is malformed",
            "file is encrypted or is not a database",
        ]
        if any(indicator in error_str for indicator in corruption_indicators):
            logger.warning(f"Database file is corrupted: {db_path}")
            if auto_repair:
                _backup_and_remove_database(db_path)
                return True
            raise DatabaseCorruptedError(db_path, e)
        # For other DatabaseErrors (locks, busy, etc.), re-raise without deleting
        raise


def _backup_and_remove_database(db_path: str) -> None:
    """Backup a corrupted database file and remove it."""
    backup_path = f"{db_path}.corrupted.{int(time.time())}"
    try:
        os.rename(db_path, backup_path)
        logger.info(f"Backed up corrupted database to: {backup_path}")
    except OSError as e:
        logger.warning(f"Failed to backup corrupted database, removing: {e}")
        try:
            os.remove(db_path)
        except OSError:
            pass

    # Also try to remove WAL and SHM files if they exist
    for suffix in ["-wal", "-shm"]:
        wal_file = f"{db_path}{suffix}"
        if os.path.exists(wal_file):
            try:
                os.remove(wal_file)
            except OSError:
                pass


class SqliteEventBusDatabase:
    """SQLite database for cross-process event communication."""

    def __init__(self, db_path: str, auto_repair: bool = True):
        self._db_path = db_path

        # Ensure directory exists
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        # Check and optionally repair corrupted database
        check_and_repair_database(db_path, auto_repair=auto_repair)

        # Initialize database with hardened concurrency settings
        self._db = SqliteDatabase(db_path, pragmas=SQLITE_HARDENED_PRAGMAS)

        class BaseModel(Model):
            class Meta:
                database = self._db

        class Event(BaseModel):  # type: ignore
            event_id = CharField(unique=True)
            event_type = CharField()
            data = JSONField()
            timestamp = DateTimeField()
            process_id = CharField()
            processed = BooleanField(default=False)  # Track if event has been processed

        self._Event = Event
        self._db.connect()
        # Use safe=True to avoid errors when tables already exist
        self._db.create_tables([Event], safe=True)

    def publish_event(self, event_type: str, data: Any, process_id: str) -> None:
        """Publish an event to the database."""
        try:
            # Serialize data, handling pydantic models
            if hasattr(data, "model_dump"):
                serialized_data = data.model_dump(mode="json", exclude_none=True)
            else:
                serialized_data = data

            execute_with_sqlite_retry(
                lambda: self._Event.create(
                    event_id=str(uuid4()),
                    event_type=event_type,
                    data=serialized_data,
                    timestamp=time.time(),
                    process_id=process_id,
                    processed=False,
                )
            )
        except Exception as e:
            logger.warning(f"Failed to publish event to database: {e}")

    def get_unprocessed_events(self, process_id: str) -> List[dict]:
        """Get unprocessed events from other processes."""
        try:
            query = (
                self._Event.select()
                .where((self._Event.process_id != process_id) & (~self._Event.processed))
                .order_by(self._Event.timestamp)
            )

            events = []
            for event in query:
                events.append(
                    {
                        "event_id": event.event_id,
                        "event_type": event.event_type,
                        "data": event.data,
                        "timestamp": event.timestamp,
                        "process_id": event.process_id,
                    }
                )

            return events
        except Exception as e:
            logger.warning(f"Failed to get unprocessed events: {e}")
            return []

    def mark_event_processed(self, event_id: str) -> None:
        """Mark an event as processed."""
        try:
            execute_with_sqlite_retry(
                lambda: self._Event.update(processed=True).where(self._Event.event_id == event_id).execute()
            )
        except Exception as e:
            logger.debug(f"Failed to mark event as processed: {e}")

    def cleanup_old_events(self, max_age_hours: int = 24) -> None:
        """Clean up old processed events."""
        try:
            cutoff_time = time.time() - (max_age_hours * 3600)
            execute_with_sqlite_retry(
                lambda: self._Event.delete()
                .where((self._Event.processed) & (self._Event.timestamp < cutoff_time))
                .execute()
            )
        except Exception as e:
            logger.debug(f"Failed to cleanup old events: {e}")
