# Global event bus instance - uses configured storage backend for cross-process functionality
import os
from typing import Any, Callable

from eval_protocol.event_bus.event_bus import EventBus
from eval_protocol.event_bus.event_bus_database import EventBusDatabase


def get_event_bus_database(db_path: str) -> EventBusDatabase:
    """
    Factory to get the configured event bus database backend.

    Uses EP_STORAGE environment variable to select backend:
    - "tinydb" (default): Uses TinyDB with JSON file storage
    - "sqlite": Uses SQLite with peewee ORM

    Args:
        db_path: Path to the database file

    Returns:
        EventBusDatabase implementation
    """
    storage_type = os.getenv("EP_STORAGE", "tinydb").lower()

    if storage_type == "sqlite":
        from eval_protocol.event_bus.sqlite_event_bus_database import SqliteEventBusDatabase

        return SqliteEventBusDatabase(db_path)
    else:
        from eval_protocol.event_bus.tinydb_event_bus_database import TinyDBEventBusDatabase

        return TinyDBEventBusDatabase(db_path)


def _get_default_db_filename() -> str:
    """Get the default database filename based on storage backend."""
    storage_type = os.getenv("EP_STORAGE", "tinydb").lower()
    return "logs.db" if storage_type == "sqlite" else "logs.json"


def _get_default_event_bus():
    from eval_protocol.event_bus.cross_process_event_bus import CrossProcessEventBus

    return CrossProcessEventBus()


# Lazy property that creates the event bus only when accessed
class _LazyEventBus(EventBus):
    def __init__(self):
        self._event_bus: EventBus | None = None

    def _get_event_bus(self):
        if self._event_bus is None:
            self._event_bus = _get_default_event_bus()
        return self._event_bus

    def subscribe(self, callback: Callable[[str, Any], None]) -> None:
        return self._get_event_bus().subscribe(callback)

    def unsubscribe(self, callback: Callable[[str, Any], None]) -> None:
        return self._get_event_bus().unsubscribe(callback)

    def emit(self, event_type: str, data: Any) -> None:
        return self._get_event_bus().emit(event_type, data)

    def start_listening(self) -> None:
        return self._get_event_bus().start_listening()

    def stop_listening(self) -> None:
        return self._get_event_bus().stop_listening()


event_bus: EventBus = _LazyEventBus()
