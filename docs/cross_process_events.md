# Event Bus System

The eval protocol includes a flexible event bus system that supports both in-process and cross-process event communication. This is particularly useful for scenarios where you have:

- An evaluation test running in one process
- A logs server running in another process
- Real-time updates between processes

## Architecture

The event bus system consists of:

1. **EventBus**: The core interface for event communication
2. **SqliteEventBus**: An implementation that adds cross-process capabilities using SQLite

### Core EventBus Interface

The `EventBus` class provides the basic event bus functionality:

```python
from eval_protocol.event_bus import EventBus

event_bus = EventBus()

def handle_event(event_type: str, data):
    print(f"Received {event_type}: {data}")

event_bus.subscribe(handle_event)
event_bus.emit("test_event", {"data": "value"})
```

### SqliteEventBus Implementation

The `SqliteEventBus` extends `EventBus` to add cross-process communication capabilities using the existing SQLite database infrastructure. Events are stored in the same database as evaluation rows, providing:

- **No additional dependencies** - Uses existing peewee/SQLite infrastructure
- **Reliable delivery** - Database transactions ensure event persistence
- **Automatic cleanup** - Old events are automatically cleaned up
- **Process isolation** - Each process has a unique ID to avoid processing its own events

### Database Schema

Events are stored in a new `Event` table with the following structure:

- `event_id`: Unique identifier for each event
- `event_type`: Type of event (e.g., "row_upserted")
- `data`: JSON data payload
- `timestamp`: When the event was created
- `process_id`: ID of the process that created the event
- `processed`: Whether the event has been processed by other processes

## Usage

### Basic Usage (In-Process)

```python
from eval_protocol.event_bus import EventBus

# Create a basic event bus for in-process communication
event_bus = EventBus()

# Subscribe to events
def handle_event(event_type: str, data):
    print(f"Received {event_type}: {data}")

event_bus.subscribe(handle_event)

# Emit events
event_bus.emit("test_event", {"data": "value"})
```

### Cross-Process Usage

```python
from eval_protocol.event_bus import SqliteEventBus

# Create a cross-process event bus
event_bus = SqliteEventBus()

# Subscribe to events
def handle_event(event_type: str, data):
    print(f"Received {event_type}: {data}")

event_bus.subscribe(handle_event)

# Start listening for cross-process events
event_bus.start_listening()

# Emit events (will be broadcast to other processes)
event_bus.emit("row_upserted", evaluation_row)
```

### Using the Global Event Bus

The global `event_bus` instance is a `SqliteEventBus` that provides cross-process functionality:

```python
from eval_protocol.event_bus import event_bus

# Subscribe to events
def handle_event(event_type: str, data):
    print(f"Received {event_type}: {data}")

event_bus.subscribe(handle_event)

# Start listening for cross-process events
event_bus.start_listening()

# Emit events
event_bus.emit("row_upserted", evaluation_row)
```

### In Evaluation Tests

The event bus is automatically used by the dataset logger. When you log evaluation rows, they are automatically broadcast to all listening processes:

```python
from eval_protocol.dataset_logger import default_logger

# This will automatically emit a "row_upserted" event
default_logger.log(evaluation_row)
```

### In Logs Server

The logs server automatically starts listening for cross-process events and broadcasts them to connected WebSocket clients:

```python
from eval_protocol.utils.logs_server import serve_logs

# This will start the server and listen for cross-process events
serve_logs()
```

## Configuration

### EventBus Configuration

The basic `EventBus` requires no configuration - it works entirely in-memory.

### SqliteEventBus Configuration

The `SqliteEventBus` automatically uses the same SQLite database as the evaluation row store, so no additional configuration is required. The database is located at:

- Default: `~/.eval_protocol/logs.db`
- Custom: Can be specified when creating the event bus

#### Custom Database Path

```python
from eval_protocol.event_bus import SqliteEventBus

# Use a custom database path
event_bus = SqliteEventBus(db_path="/path/to/custom.db")
```

## Performance Considerations

### EventBus Performance

- **In-memory**: Events are processed immediately with no latency
- **Memory usage**: Events are not persisted, so memory usage is minimal
- **Scalability**: Suitable for high-frequency events within a single process

### SqliteEventBus Performance

- **Database-based**: Events are stored in SQLite with proper indexing
- **Polling frequency**: Events are checked every 100ms by default
- **Memory usage**: Events are automatically cleaned up after 24 hours
- **Latency**: ~100ms latency due to polling interval
- **Scalability**: Suitable for moderate event volumes (< 1000 events/second)

## Event Types

The following event types are currently supported:

- `row_upserted`: Emitted when an evaluation row is logged
- `log`: Legacy event type (handled the same as `row_upserted`)

## Testing

You can test the cross-process event bus using the provided example:

1. Start the logs server in one terminal:
   ```bash
   python examples/cross_process_events_example.py server
   ```

2. Run the evaluation in another terminal:
   ```bash
   python examples/cross_process_events_example.py eval
   ```

## Troubleshooting

### Events Not Received

1. Check that the event bus is started listening: `event_bus.start_listening()`
2. Verify the database is accessible and writable
3. Check for database lock issues (multiple processes accessing the same database)
4. Ensure both processes are using the same database path

### Database Lock Issues

SQLite has limitations with concurrent access. If you experience database locks:

1. Ensure processes are not writing to the database simultaneously
2. Consider using a different database backend for high-concurrency scenarios
3. The event bus automatically handles some concurrency issues

### High Database Size

The system automatically cleans up old processed events after 24 hours. If you're seeing high database size:

1. Check the database file size: `~/.eval_protocol/logs.db`
2. Manually clean up old events if needed
3. Adjust the cleanup interval in the code if necessary

### Performance Issues

If you're experiencing performance issues:

1. Check the polling interval (currently 100ms)
2. Monitor database size and cleanup frequency
3. Consider reducing the number of events emitted
4. Profile the database queries for bottlenecks
