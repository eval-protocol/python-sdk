import time
from typing import Any, List
from uuid import uuid4

from peewee import BooleanField, CharField, DateTimeField, Model, SqliteDatabase
from playhouse.sqlite_ext import JSONField

from eval_protocol.event_bus.logger import logger


class SqliteEventBusDatabase:
    """SQLite database for cross-process event communication."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._db = SqliteDatabase(db_path)

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
        self._db.create_tables([Event])

    def publish_event(self, event_type: str, data: Any, process_id: str) -> None:
        """Publish an event to the database."""
        try:
            # Serialize data, handling pydantic models
            if hasattr(data, "model_dump"):
                serialized_data = data.model_dump(mode="json", exclude_none=True)
            else:
                serialized_data = data

            self._Event.create(
                event_id=str(uuid4()),
                event_type=event_type,
                data=serialized_data,
                timestamp=time.time(),
                process_id=process_id,
                processed=False,
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
            self._Event.update(processed=True).where(self._Event.event_id == event_id).execute()
        except Exception as e:
            logger.debug(f"Failed to mark event as processed: {e}")

    def cleanup_old_events(self, max_age_hours: int = 24) -> None:
        """Clean up old processed events."""
        try:
            cutoff_time = time.time() - (max_age_hours * 3600)
            self._Event.delete().where((self._Event.processed) & (self._Event.timestamp < cutoff_time)).execute()
        except Exception as e:
            logger.debug(f"Failed to cleanup old events: {e}")
