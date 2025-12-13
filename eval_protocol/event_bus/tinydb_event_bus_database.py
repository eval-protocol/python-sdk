import os
import time
from typing import Any, List
from uuid import uuid4

from tinydb import Query, TinyDB

from eval_protocol.event_bus.event_bus_database import EventBusDatabase
from eval_protocol.event_bus.logger import logger


class TinyDBEventBusDatabase(EventBusDatabase):
    """
    TinyDB-based event bus database for cross-process event communication.

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
        self._table = self._db.table("events")

    def publish_event(self, event_type: str, data: Any, process_id: str) -> None:
        """Publish an event to the database."""
        try:
            # Serialize data, handling pydantic models
            if hasattr(data, "model_dump"):
                serialized_data = data.model_dump(mode="json", exclude_none=True)
            else:
                serialized_data = data

            self._table.insert(
                {
                    "event_id": str(uuid4()),
                    "event_type": event_type,
                    "data": serialized_data,
                    "timestamp": time.time(),
                    "process_id": process_id,
                    "processed": False,
                }
            )
        except Exception as e:
            logger.warning(f"Failed to publish event to database: {e}")

    def get_unprocessed_events(self, process_id: str) -> List[dict]:
        """Get unprocessed events from other processes."""
        try:
            Event = Query()
            results = self._table.search((Event.process_id != process_id) & (Event.processed == False))  # noqa: E712

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
        except Exception as e:
            logger.warning(f"Failed to get unprocessed events: {e}")
            return []

    def mark_event_processed(self, event_id: str) -> None:
        """Mark an event as processed."""
        try:
            Event = Query()
            self._table.update({"processed": True}, Event.event_id == event_id)
        except Exception as e:
            logger.debug(f"Failed to mark event as processed: {e}")

    def cleanup_old_events(self, max_age_hours: int = 24) -> None:
        """Clean up old processed events."""
        try:
            cutoff_time = time.time() - (max_age_hours * 3600)
            Event = Query()
            self._table.remove((Event.processed == True) & (Event.timestamp < cutoff_time))  # noqa: E712
        except Exception as e:
            logger.debug(f"Failed to cleanup old events: {e}")
