# Global event bus instance - uses SqliteEventBus for cross-process functionality
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

    def __getattr__(self, name):
        return getattr(self._get_event_bus(), name)


event_bus: EventBus = _LazyEventBus()
