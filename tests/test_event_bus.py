import tempfile
import time

from eval_protocol.event_bus import SqliteEventBus
from eval_protocol.event_bus.event_bus import EventBus
from eval_protocol.models import EvaluationRow, InputMetadata


class TestSqliteEventBus:
    def test_basic_event_emission(self):
        """Test basic event emission and subscription."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            event_bus = SqliteEventBus(db_path=db_path)

            # Test event listener
            received_events = []

            def test_listener(event_type: str, data):
                received_events.append((event_type, data))

            event_bus.subscribe(test_listener)

            # Emit an event
            test_data = {"test": "data"}
            event_bus.emit("test_event", test_data)

            # Check that local listener received the event
            assert len(received_events) == 1
            assert received_events[0][0] == "test_event"
            assert received_events[0][1] == test_data

        finally:
            import os

            os.unlink(db_path)

    def test_cross_process_events(self):
        """Test cross-process event communication."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            # Create two event buses (simulating different processes)
            event_bus1 = SqliteEventBus(db_path=db_path)
            event_bus2 = SqliteEventBus(db_path=db_path)

            # Set up listener on event_bus2
            received_events = []

            def test_listener(event_type: str, data):
                received_events.append((event_type, data))

            event_bus2.subscribe(test_listener)
            event_bus2.start_listening()

            # Emit event from event_bus1
            test_data = {"test": "cross_process"}
            event_bus1.emit("cross_process_event", test_data)

            # Wait a bit for the event to be processed
            time.sleep(0.2)

            # Check that event_bus2 received the event
            assert len(received_events) == 1
            assert received_events[0][0] == "cross_process_event"
            assert received_events[0][1] == test_data

            event_bus2.stop_listening()

        finally:
            import os

            os.unlink(db_path)

    def test_evaluation_row_events(self):
        """Test that EvaluationRow objects can be emitted and received."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            event_bus1 = SqliteEventBus(db_path=db_path)
            event_bus2 = SqliteEventBus(db_path=db_path)

            received_events = []

            def test_listener(event_type: str, data):
                received_events.append((event_type, data))

            event_bus2.subscribe(test_listener)
            event_bus2.start_listening()

            # Create and emit an EvaluationRow
            test_row = EvaluationRow(
                messages=[{"role": "user", "content": "test"}], input_metadata=InputMetadata(row_id="test-123")
            )

            event_bus1.emit("row_upserted", test_row)

            # Wait for processing
            time.sleep(0.2)

            # Check that the event was received
            assert len(received_events) == 1
            assert received_events[0][0] == "row_upserted"
            # The event data should be a dict, but it should be deserializable by EvaluationRow
            assert isinstance(received_events[0][1], dict)
            # Try to deserialize to EvaluationRow to ensure it's compatible
            event = EvaluationRow(**received_events[0][1])
            assert event.input_metadata.row_id == "test-123"

            event_bus2.stop_listening()

        finally:
            import os

            os.unlink(db_path)

    def test_process_isolation(self):
        """Test that processes receive their own events locally but not via cross-process mechanism."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            event_bus = SqliteEventBus(db_path=db_path)

            received_events = []

            def test_listener(event_type: str, data):
                received_events.append((event_type, data))

            event_bus.subscribe(test_listener)
            event_bus.start_listening()

            # Emit an event
            event_bus.emit("self_event", {"test": "data"})

            # Wait for processing
            time.sleep(0.2)

            # Should receive the event from its own process via local delivery
            assert len(received_events) == 1
            assert received_events[0] == ("self_event", {"test": "data"})

            event_bus.stop_listening()

        finally:
            import os

            os.unlink(db_path)

    def test_multiple_listeners(self):
        """Test that multiple listeners can subscribe to events."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            event_bus = SqliteEventBus(db_path=db_path)

            listener1_events = []
            listener2_events = []

            def listener1(event_type: str, data):
                listener1_events.append((event_type, data))

            def listener2(event_type: str, data):
                listener2_events.append((event_type, data))

            event_bus.subscribe(listener1)
            event_bus.subscribe(listener2)

            # Emit an event
            test_data = {"test": "multiple_listeners"}
            event_bus.emit("multi_event", test_data)

            # Check that both listeners received the event
            assert len(listener1_events) == 1
            assert len(listener2_events) == 1
            assert listener1_events[0] == ("multi_event", test_data)
            assert listener2_events[0] == ("multi_event", test_data)

        finally:
            import os

            os.unlink(db_path)


class TestEventBus:
    def test_basic_event_bus(self):
        """Test the core EventBus interface."""
        event_bus = EventBus()

        received_events = []

        def test_listener(event_type: str, data):
            received_events.append((event_type, data))

        event_bus.subscribe(test_listener)

        # Test local event emission
        test_data = {"test": "basic_event_bus"}
        event_bus.emit("test_event", test_data)

        assert len(received_events) == 1
        assert received_events[0] == ("test_event", test_data)

    def test_multiple_listeners(self):
        """Test that multiple listeners can subscribe to events."""
        event_bus = EventBus()

        listener1_events = []
        listener2_events = []

        def listener1(event_type: str, data):
            listener1_events.append((event_type, data))

        def listener2(event_type: str, data):
            listener2_events.append((event_type, data))

        event_bus.subscribe(listener1)
        event_bus.subscribe(listener2)

        # Emit an event
        test_data = {"test": "multiple_listeners"}
        event_bus.emit("multi_event", test_data)

        # Check that both listeners received the event
        assert len(listener1_events) == 1
        assert len(listener2_events) == 1
        assert listener1_events[0] == ("multi_event", test_data)
        assert listener2_events[0] == ("multi_event", test_data)

    def test_unsubscribe(self):
        """Test that unsubscribing works correctly."""
        event_bus = EventBus()

        received_events = []

        def test_listener(event_type: str, data):
            received_events.append((event_type, data))

        event_bus.subscribe(test_listener)

        # Emit an event
        event_bus.emit("test_event", {"test": "data"})
        assert len(received_events) == 1

        # Unsubscribe
        event_bus.unsubscribe(test_listener)

        # Emit another event
        event_bus.emit("test_event2", {"test": "data2"})
        assert len(received_events) == 1  # Should not receive the second event

    def test_start_stop_listening_noop(self):
        """Test that start_listening and stop_listening are no-ops in base EventBus."""
        event_bus = EventBus()

        # Should not raise any exceptions (these are no-ops in base class)
        event_bus.start_listening()
        event_bus.stop_listening()
        event_bus.start_listening()
        event_bus.stop_listening()
