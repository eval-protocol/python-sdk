#!/usr/bin/env python3
"""Helper script for testing event bus cross-process communication."""

import asyncio
import sys
import json
from eval_protocol.event_bus.sqlite_event_bus import SqliteEventBus


async def listener_process(db_path: str):
    """Run an event bus listener in a separate process."""
    try:
        event_bus = SqliteEventBus(db_path=db_path)

        received_events = []

        def test_listener(event_type: str, data):
            received_events.append((event_type, data))

        event_bus.subscribe(test_listener)
        event_bus.start_listening()

        # Wait for events for up to 5 seconds
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < 5.0:
            await asyncio.sleep(0.1)
            if received_events:
                break

        # Output results to stdout
        print(json.dumps(received_events))
        event_bus.stop_listening()

    except Exception as e:
        print(f"Error in listener process: {e}", file=sys.stderr)
        sys.exit(1)


async def emitter_process(db_path: str, event_type: str, data_json: str):
    """Emit an event from a separate process."""
    try:
        event_bus = SqliteEventBus(db_path=db_path)

        # Parse the data
        if data_json:
            data = json.loads(data_json)
        else:
            data = None

        event_bus.emit(event_type, data)

    except Exception as e:
        print(f"Error in emitter process: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_event_bus_helper.py <mode> <db_path> [event_type] [data_json]", file=sys.stderr)
        sys.exit(1)

    mode = sys.argv[1]
    db_path = sys.argv[2]

    if mode == "listener":
        asyncio.run(listener_process(db_path))
    elif mode == "emitter":
        event_type = sys.argv[3]
        data_json = sys.argv[4] if len(sys.argv) > 4 else ""
        asyncio.run(emitter_process(db_path, event_type, data_json))
    else:
        print(f"Unknown mode: {mode}", file=sys.stderr)
        sys.exit(1)
