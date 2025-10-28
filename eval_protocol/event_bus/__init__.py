# Global event bus instance - uses SqliteEventBus for cross-process functionality
from typing import Any, Callable
from eval_protocol.event_bus.event_bus import EventBus


def _get_default_event_bus():
    from eval_protocol.event_bus.sqlite_event_bus import SqliteEventBus

    return SqliteEventBus()


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
