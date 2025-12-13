"""
Backwards-compatible alias for CrossProcessEventBus.

This module is kept for backwards compatibility. New code should use
CrossProcessEventBus from cross_process_event_bus.py instead.
"""

from eval_protocol.event_bus.cross_process_event_bus import CrossProcessEventBus

# Backwards-compatible alias
SqliteEventBus = CrossProcessEventBus

__all__ = ["SqliteEventBus"]
