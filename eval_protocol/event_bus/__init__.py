# Global event bus instance - uses SqliteEventBus for cross-process functionality
from eval_protocol.event_bus.event_bus import EventBus
from eval_protocol.event_bus.sqlite_event_bus import SqliteEventBus

event_bus: EventBus = SqliteEventBus()
