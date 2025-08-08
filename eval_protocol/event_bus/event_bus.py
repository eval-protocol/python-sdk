from typing import Any, Callable, List

from eval_protocol.event_bus.logger import logger


class EventBus:
    """Core event bus interface for decoupling components in the evaluation system."""

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

    def start_listening(self) -> None:
        """Start listening for cross-process events. Override in subclasses."""
        pass

    def stop_listening(self) -> None:
        """Stop listening for cross-process events. Override in subclasses."""
        pass
