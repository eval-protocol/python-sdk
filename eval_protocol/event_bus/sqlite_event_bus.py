import asyncio
import os
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
            from eval_protocol.directory_utils import find_eval_protocol_dir

            eval_protocol_dir = find_eval_protocol_dir()
            db_path = os.path.join(eval_protocol_dir, "logs.db")

        self._db: SqliteEventBusDatabase = SqliteEventBusDatabase(db_path)
        self._running = False
        self._process_id = str(os.getpid())

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
        loop = asyncio.get_running_loop()
        loop.create_task(self._database_listener_task())
        logger.debug("[CROSS_PROCESS_LISTEN] Started async database listener task")

    def stop_listening(self) -> None:
        """Stop listening for cross-process events."""
        logger.debug("[CROSS_PROCESS_LISTEN] Stopping cross-process event listening")
        self._running = False

    async def _database_listener_task(self) -> None:
        """Single database listener task that processes events and recreates itself."""
        if not self._running:
            # this should end the task loop
            logger.debug("[CROSS_PROCESS_LISTENER] Stopping database listener task")
            return

        # Get unprocessed events from other processes
        events = self._db.get_unprocessed_events(str(self._process_id))
        if events:
            logger.debug(f"[CROSS_PROCESS_LISTENER] Found {len(events)} unprocessed events")
        else:
            logger.debug(f"[CROSS_PROCESS_LISTENER] No unprocessed events found for process {self._process_id}")

        for event in events:
            logger.debug(
                f"[CROSS_PROCESS_LISTENER] Processing event {event['event_id']} of type {event['event_type']}"
            )
            # Handle the event
            self._handle_cross_process_event(event["event_type"], event["data"])
            logger.debug(f"[CROSS_PROCESS_LISTENER] Successfully processed event {event['event_id']}")

            # Mark as processed
            self._db.mark_event_processed(event["event_id"])
            logger.debug(f"[CROSS_PROCESS_LISTENER] Marked event {event['event_id']} as processed")

        # Clean up old events every hour
        current_time = time.time()
        if not hasattr(self, "_last_cleanup"):
            self._last_cleanup = current_time
        elif current_time - self._last_cleanup >= 3600:
            logger.debug("[CROSS_PROCESS_LISTENER] Cleaning up old events")
            self._db.cleanup_old_events()
            self._last_cleanup = current_time

        # Schedule the next task if still running
        await asyncio.sleep(1.0)
        loop = asyncio.get_running_loop()
        loop.create_task(self._database_listener_task())

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
