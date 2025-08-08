import threading
import time
from typing import Any, Optional
from uuid import uuid4

from eval_protocol.event_bus.event_bus import EventBus
from eval_protocol.event_bus.logger import logger
from eval_protocol.event_bus.sqlite_event_bus_database import SqliteEventBusDatabase


class SqliteEventBus(EventBus):
    """SQLite-based event bus implementation that supports cross-process communication."""

    def __init__(self, db_path: Optional[str] = None):
        super().__init__()

        # Use the same database as the evaluation row store
        if db_path is None:
            import os

            from eval_protocol.directory_utils import find_eval_protocol_dir

            eval_protocol_dir = find_eval_protocol_dir()
            db_path = os.path.join(eval_protocol_dir, "logs.db")

        self._db = SqliteEventBusDatabase(db_path)
        self._running = False
        self._listener_thread: Optional[threading.Thread] = None
        self._process_id = str(uuid4())

    def emit(self, event_type: str, data: Any) -> None:
        """Emit an event to all subscribers.

        Args:
            event_type: Type of event (e.g., "log")
            data: Event data
        """
        # Call local listeners immediately
        super().emit(event_type, data)

        # Publish to cross-process subscribers
        self._publish_cross_process(event_type, data)

    def _publish_cross_process(self, event_type: str, data: Any) -> None:
        """Publish event to cross-process subscribers via database."""
        self._db.publish_event(event_type, data, self._process_id)

    def start_listening(self) -> None:
        """Start listening for cross-process events."""
        if self._running:
            return

        self._running = True
        self._start_database_listener()

    def stop_listening(self) -> None:
        """Stop listening for cross-process events."""
        self._running = False
        if self._listener_thread and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=1)

    def _start_database_listener(self) -> None:
        """Start database-based event listener."""

        def database_listener():
            last_cleanup = time.time()

            while self._running:
                try:
                    # Get unprocessed events from other processes
                    events = self._db.get_unprocessed_events(self._process_id)

                    for event in events:
                        if not self._running:
                            break

                        try:
                            # Handle the event
                            self._handle_cross_process_event(event["event_type"], event["data"])

                            # Mark as processed
                            self._db.mark_event_processed(event["event_id"])

                        except Exception as e:
                            logger.debug(f"Failed to process event {event['event_id']}: {e}")

                    # Clean up old events every hour
                    current_time = time.time()
                    if current_time - last_cleanup >= 3600:
                        self._db.cleanup_old_events()
                        last_cleanup = current_time

                    # Small sleep to prevent busy waiting
                    time.sleep(0.1)

                except Exception as e:
                    logger.debug(f"Database listener error: {e}")
                    time.sleep(1)

        self._listener_thread = threading.Thread(target=database_listener, daemon=True)
        self._listener_thread.start()

    def _handle_cross_process_event(self, event_type: str, data: Any) -> None:
        """Handle events received from other processes."""
        for listener in self._listeners:
            try:
                listener(event_type, data)
            except Exception as e:
                logger.debug(f"Cross-process event listener failed for {event_type}: {e}")
