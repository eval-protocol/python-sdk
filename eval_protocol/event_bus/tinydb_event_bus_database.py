import json
import os
import time
from typing import Any, List
from uuid import uuid4

from tinydb import Query, TinyDB
from tinyrecord.transaction import transaction

from eval_protocol.event_bus.event_bus_database import EventBusDatabase
from eval_protocol.event_bus.logger import logger


class TinyDBEventBusDatabase(EventBusDatabase):
    """
    TinyDB-based event bus database for cross-process event communication.

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
        self._db = self._open_db_with_retry()
        self._table = self._db.table("events")

    def _open_db_with_retry(self, max_retries: int = 3) -> TinyDB:
        """Open TinyDB with retry logic to handle transient JSON decode errors."""
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                return TinyDB(self._db_path)
            except json.JSONDecodeError as e:
                last_error = e
                logger.warning(f"TinyDB JSON decode error on attempt {attempt + 1}: {e}")
                # Wait a bit and retry - the file might be mid-write
                time.sleep(0.1 * (attempt + 1))
                # Try to recover by removing the corrupted file
                if attempt == max_retries - 1 and os.path.exists(self._db_path):
                    try:
                        logger.warning(f"Removing corrupted TinyDB file: {self._db_path}")
                        os.remove(self._db_path)
                        return TinyDB(self._db_path)
                    except Exception:
                        pass
        raise last_error if last_error else RuntimeError("Failed to open TinyDB")

    def publish_event(self, event_type: str, data: Any, process_id: str) -> None:
        """Publish an event to the database using atomic transaction."""
        try:
            # Serialize data, handling pydantic models
            if hasattr(data, "model_dump"):
                serialized_data = data.model_dump(mode="json", exclude_none=True)
            else:
                serialized_data = data

            document = {
                "event_id": str(uuid4()),
                "event_type": event_type,
                "data": serialized_data,
                "timestamp": time.time(),
                "process_id": process_id,
                "processed": False,
            }

            # Use tinyrecord transaction for atomic, concurrent-safe insert
            with transaction(self._table) as tr:
                tr.insert(document)
        except Exception as e:
            logger.warning(f"Failed to publish event to database: {e}")

    def get_unprocessed_events(self, process_id: str) -> List[dict]:
        """Get unprocessed events from other processes with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Clear query cache to force fresh read from disk
                # TinyDB caches query results, so we need to clear cache to see
                # events written by other processes. The search() method will
                # automatically call _read_table() on a cache miss.
                self._table.clear_cache()

                Event = Query()
                results = self._table.search((Event.process_id != process_id) & (Event.processed == False))  # noqa: E712

                logger.debug(
                    f"TinyDBEventBusDatabase: Found {len(results)} unprocessed events for process_id: {process_id} in database: {self._db_path}"
                )

                events = []
                # Sort by timestamp
                for event in sorted(results, key=lambda x: x.get("timestamp", 0)):
                    events.append(
                        {
                            "event_id": event["event_id"],
                            "event_type": event["event_type"],
                            "data": event["data"],
                            "timestamp": event["timestamp"],
                            "process_id": event["process_id"],
                        }
                    )

                return events
            except json.JSONDecodeError as e:
                logger.warning(f"TinyDB JSON decode error on get_unprocessed_events attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))
                else:
                    logger.warning("Failed to read events after retries, returning empty list")
                    return []
            except Exception as e:
                logger.warning(f"Failed to get unprocessed events: {e}")
                return []
        return []

    def mark_event_processed(self, event_id: str) -> None:
        """Mark an event as processed using atomic transaction."""
        try:
            Event = Query()
            with transaction(self._table) as tr:
                tr.update({"processed": True}, Event.event_id == event_id)
        except Exception as e:
            logger.debug(f"Failed to mark event as processed: {e}")

    def cleanup_old_events(self, max_age_hours: int = 24) -> None:
        """Clean up old processed events using atomic transaction."""
        try:
            # Clear cache to see latest data before cleanup
            self._table.clear_cache()

            cutoff_time = time.time() - (max_age_hours * 3600)
            Event = Query()
            with transaction(self._table) as tr:
                tr.remove((Event.processed == True) & (Event.timestamp < cutoff_time))  # noqa: E712
        except Exception as e:
            logger.debug(f"Failed to cleanup old events: {e}")
