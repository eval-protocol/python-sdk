import logging
from typing import Any, Callable, List

logger = logging.getLogger(__name__)


class EventBus:
    """Simple event bus for decoupling components in the evaluation system."""

    def __init__(self):
        self._listeners: List[Callable[[str, Any], None]] = []

    def subscribe(self, callback: Callable[[str, Any], None]) -> None:
        """Subscribe to events.

        Args:
            callback: Function that takes (event_type, data) parameters
        """
        self._listeners.append(callback)

    def unsubscribe(self, callback: Callable[[str, Any], None]) -> None:
        """Unsubscribe from events.

        Args:
            callback: The callback function to remove
        """
        try:
            self._listeners.remove(callback)
        except ValueError:
            pass  # Callback wasn't subscribed

    def emit(self, event_type: str, data: Any) -> None:
        """Emit an event to all subscribers.

        Args:
            event_type: Type of event (e.g., "row_upserted")
            data: Event data
        """
        for listener in self._listeners:
            try:
                listener(event_type, data)
            except Exception as e:
                logger.debug(f"Event listener failed for {event_type}: {e}")


# Global event bus instance
event_bus = EventBus()
