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
        logger.debug(f"[CROSS_PROCESS_EMIT] Emitting event type: {event_type}")

        # Call local listeners immediately
        logger.debug(f"[CROSS_PROCESS_EMIT] Calling {len(self._listeners)} local listeners")
        super().emit(event_type, data)
        logger.debug("[CROSS_PROCESS_EMIT] Completed local listener calls")

        # Publish to cross-process subscribers
        logger.debug("[CROSS_PROCESS_EMIT] Publishing to cross-process subscribers")
        self._publish_cross_process(event_type, data)
        logger.debug("[CROSS_PROCESS_EMIT] Completed cross-process publish")

    def _publish_cross_process(self, event_type: str, data: Any) -> None:
        """Publish event to cross-process subscribers via database."""
        logger.debug(f"[CROSS_PROCESS_PUBLISH] Publishing event {event_type} to database")
        try:
            self._db.publish_event(event_type, data, self._process_id)
            logger.debug(f"[CROSS_PROCESS_PUBLISH] Successfully published event {event_type} to database")
        except Exception as e:
            logger.error(f"[CROSS_PROCESS_PUBLISH] Failed to publish event {event_type} to database: {e}")

    def start_listening(self) -> None:
        """Start listening for cross-process events."""
        if self._running:
            logger.debug("[CROSS_PROCESS_LISTEN] Already listening, skipping start")
            return

        logger.debug("[CROSS_PROCESS_LISTEN] Starting cross-process event listening")
        self._running = True
        self._start_database_listener()
        logger.debug("[CROSS_PROCESS_LISTEN] Started database listener thread")

    def stop_listening(self) -> None:
        """Stop listening for cross-process events."""
        logger.debug("[CROSS_PROCESS_LISTEN] Stopping cross-process event listening")
        self._running = False
        if self._listener_thread and self._listener_thread.is_alive():
            logger.debug("[CROSS_PROCESS_LISTEN] Waiting for listener thread to stop")
            self._listener_thread.join(timeout=1)
            logger.debug("[CROSS_PROCESS_LISTEN] Listener thread stopped")

    def _start_database_listener(self) -> None:
        """Start database-based event listener."""

        def database_listener():
            logger.debug("[CROSS_PROCESS_LISTENER] Starting database listener loop")
            last_cleanup = time.time()

            while self._running:
                try:
                    # Get unprocessed events from other processes
                    events = self._db.get_unprocessed_events(self._process_id)
                    if events:
                        logger.debug(f"[CROSS_PROCESS_LISTENER] Found {len(events)} unprocessed events")

                    for event in events:
                        if not self._running:
                            break

                        try:
                            logger.debug(
                                f"[CROSS_PROCESS_LISTENER] Processing event {event['event_id']} of type {event['event_type']}"
                            )
                            # Handle the event
                            self._handle_cross_process_event(event["event_type"], event["data"])
                            logger.debug(f"[CROSS_PROCESS_LISTENER] Successfully processed event {event['event_id']}")

                            # Mark as processed
                            self._db.mark_event_processed(event["event_id"])
                            logger.debug(f"[CROSS_PROCESS_LISTENER] Marked event {event['event_id']} as processed")

                        except Exception as e:
                            logger.debug(f"[CROSS_PROCESS_LISTENER] Failed to process event {event['event_id']}: {e}")

                    # Clean up old events every hour
                    current_time = time.time()
                    if current_time - last_cleanup >= 3600:
                        logger.debug("[CROSS_PROCESS_LISTENER] Cleaning up old events")
                        self._db.cleanup_old_events()
                        last_cleanup = current_time

                    # Small sleep to prevent busy waiting
                    time.sleep(0.1)

                except Exception as e:
                    logger.debug(f"[CROSS_PROCESS_LISTENER] Database listener error: {e}")
                    time.sleep(1)

        self._listener_thread = threading.Thread(target=database_listener, daemon=True)
        self._listener_thread.start()

    def _handle_cross_process_event(self, event_type: str, data: Any) -> None:
        """Handle events received from other processes."""
        logger.debug(f"[CROSS_PROCESS_HANDLE] Handling cross-process event type: {event_type}")
        logger.debug(f"[CROSS_PROCESS_HANDLE] Calling {len(self._listeners)} listeners")

        for i, listener in enumerate(self._listeners):
            try:
                logger.debug(f"[CROSS_PROCESS_HANDLE] Calling listener {i}")
                listener(event_type, data)
                logger.debug(f"[CROSS_PROCESS_HANDLE] Successfully called listener {i}")
            except Exception as e:
                logger.debug(f"[CROSS_PROCESS_HANDLE] Cross-process event listener {i} failed for {event_type}: {e}")
