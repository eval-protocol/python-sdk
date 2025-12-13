"""
Tests for the storage backend abstraction and implementations.

Tests both TinyDB (default) and SQLite backends to ensure they work correctly
and can be selected via the EP_STORAGE environment variable.
"""

import os
import tempfile
from typing import Generator

import pytest

from eval_protocol.dataset_logger.evaluation_row_store import EvaluationRowStore
from eval_protocol.dataset_logger.sqlite_evaluation_row_store import SqliteEvaluationRowStore
from eval_protocol.dataset_logger.tinydb_evaluation_row_store import TinyDBEvaluationRowStore
from eval_protocol.event_bus.event_bus_database import EventBusDatabase
from eval_protocol.event_bus.sqlite_event_bus_database import SqliteEventBusDatabase
from eval_protocol.event_bus.tinydb_event_bus_database import TinyDBEventBusDatabase


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestEvaluationRowStoreABC:
    """Tests that both implementations correctly implement the ABC."""

    def test_sqlite_implements_abc(self, temp_dir: str):
        """SqliteEvaluationRowStore should implement EvaluationRowStore."""
        db_path = os.path.join(temp_dir, "test.db")
        store = SqliteEvaluationRowStore(db_path)
        assert isinstance(store, EvaluationRowStore)

    def test_tinydb_implements_abc(self, temp_dir: str):
        """TinyDBEvaluationRowStore should implement EvaluationRowStore."""
        db_path = os.path.join(temp_dir, "test.json")
        store = TinyDBEvaluationRowStore(db_path)
        assert isinstance(store, EvaluationRowStore)


class TestTinyDBEvaluationRowStore:
    """Tests for TinyDBEvaluationRowStore."""

    def test_upsert_and_read_row(self, temp_dir: str):
        """Test basic upsert and read operations."""
        db_path = os.path.join(temp_dir, "test.json")
        store = TinyDBEvaluationRowStore(db_path)

        data = {
            "execution_metadata": {"rollout_id": "test-rollout-1"},
            "input_metadata": {"row_id": "row-1"},
            "messages": [{"role": "user", "content": "Hello"}],
        }

        store.upsert_row(data)
        rows = store.read_rows(rollout_id="test-rollout-1")

        assert len(rows) == 1
        assert rows[0]["execution_metadata"]["rollout_id"] == "test-rollout-1"
        assert rows[0]["input_metadata"]["row_id"] == "row-1"

    def test_upsert_updates_existing_row(self, temp_dir: str):
        """Test that upsert updates an existing row."""
        db_path = os.path.join(temp_dir, "test.json")
        store = TinyDBEvaluationRowStore(db_path)

        data1 = {
            "execution_metadata": {"rollout_id": "test-rollout-1"},
            "input_metadata": {"row_id": "row-1"},
            "messages": [{"role": "user", "content": "Hello"}],
        }
        data2 = {
            "execution_metadata": {"rollout_id": "test-rollout-1"},
            "input_metadata": {"row_id": "row-1-updated"},
            "messages": [{"role": "user", "content": "Updated"}],
        }

        store.upsert_row(data1)
        store.upsert_row(data2)

        rows = store.read_rows()
        assert len(rows) == 1
        assert rows[0]["input_metadata"]["row_id"] == "row-1-updated"

    def test_read_all_rows(self, temp_dir: str):
        """Test reading all rows without filter."""
        db_path = os.path.join(temp_dir, "test.json")
        store = TinyDBEvaluationRowStore(db_path)

        for i in range(3):
            data = {
                "execution_metadata": {"rollout_id": f"test-rollout-{i}"},
                "input_metadata": {"row_id": f"row-{i}"},
            }
            store.upsert_row(data)

        rows = store.read_rows()
        assert len(rows) == 3

    def test_delete_row(self, temp_dir: str):
        """Test deleting a specific row."""
        db_path = os.path.join(temp_dir, "test.json")
        store = TinyDBEvaluationRowStore(db_path)

        data = {
            "execution_metadata": {"rollout_id": "test-rollout-1"},
            "input_metadata": {"row_id": "row-1"},
        }
        store.upsert_row(data)

        deleted = store.delete_row("test-rollout-1")
        assert deleted == 1

        rows = store.read_rows()
        assert len(rows) == 0

    def test_delete_row_nonexistent(self, temp_dir: str):
        """Test deleting a row that doesn't exist returns 0."""
        db_path = os.path.join(temp_dir, "test.json")
        store = TinyDBEvaluationRowStore(db_path)

        # Try to delete a row that doesn't exist
        deleted = store.delete_row("nonexistent-rollout")
        assert deleted == 0

        # Verify store is still empty
        rows = store.read_rows()
        assert len(rows) == 0

    def test_delete_all_rows(self, temp_dir: str):
        """Test deleting all rows."""
        db_path = os.path.join(temp_dir, "test.json")
        store = TinyDBEvaluationRowStore(db_path)

        for i in range(3):
            data = {
                "execution_metadata": {"rollout_id": f"test-rollout-{i}"},
                "input_metadata": {"row_id": f"row-{i}"},
            }
            store.upsert_row(data)

        deleted = store.delete_all_rows()
        assert deleted == 3

        rows = store.read_rows()
        assert len(rows) == 0

    def test_db_path_property(self, temp_dir: str):
        """Test that db_path property returns correct path."""
        db_path = os.path.join(temp_dir, "test.json")
        store = TinyDBEvaluationRowStore(db_path)
        assert store.db_path == db_path

    def test_raises_on_missing_rollout_id(self, temp_dir: str):
        """Test that upsert raises when rollout_id is None."""
        db_path = os.path.join(temp_dir, "test.json")
        store = TinyDBEvaluationRowStore(db_path)

        data = {
            "execution_metadata": {"rollout_id": None},
            "input_metadata": {"row_id": "row-1"},
        }

        with pytest.raises(ValueError, match="rollout_id is required"):
            store.upsert_row(data)


class TestTinyDBEventBusDatabase:
    """Tests for TinyDBEventBusDatabase."""

    def test_publish_and_get_events(self, temp_dir: str):
        """Test publishing and retrieving events."""
        db_path = os.path.join(temp_dir, "events.json")
        db = TinyDBEventBusDatabase(db_path)

        db.publish_event("test_event", {"key": "value"}, "process-1")

        # Get events from a different process
        events = db.get_unprocessed_events("process-2")

        assert len(events) == 1
        assert events[0]["event_type"] == "test_event"
        assert events[0]["data"] == {"key": "value"}
        assert events[0]["process_id"] == "process-1"

    def test_events_filtered_by_process_id(self, temp_dir: str):
        """Test that events from same process are not returned."""
        db_path = os.path.join(temp_dir, "events.json")
        db = TinyDBEventBusDatabase(db_path)

        db.publish_event("test_event", {"key": "value"}, "process-1")

        # Get events from the same process - should be empty
        events = db.get_unprocessed_events("process-1")
        assert len(events) == 0

    def test_mark_event_processed(self, temp_dir: str):
        """Test marking events as processed."""
        db_path = os.path.join(temp_dir, "events.json")
        db = TinyDBEventBusDatabase(db_path)

        db.publish_event("test_event", {"key": "value"}, "process-1")

        events = db.get_unprocessed_events("process-2")
        assert len(events) == 1

        db.mark_event_processed(events[0]["event_id"])

        # Should no longer be returned
        events = db.get_unprocessed_events("process-2")
        assert len(events) == 0

    def test_cleanup_old_events(self, temp_dir: str):
        """Test cleaning up old processed events."""
        db_path = os.path.join(temp_dir, "events.json")
        db = TinyDBEventBusDatabase(db_path)

        db.publish_event("test_event", {"key": "value"}, "process-1")
        events = db.get_unprocessed_events("process-2")
        db.mark_event_processed(events[0]["event_id"])

        # Cleanup with 0 hours should remove all processed events
        db.cleanup_old_events(max_age_hours=0)

        # The event should still be gone (processed)
        events = db.get_unprocessed_events("process-2")
        assert len(events) == 0


class TestEventBusDatabaseABC:
    """Tests that both implementations correctly implement the ABC."""

    def test_sqlite_implements_abc(self, temp_dir: str):
        """SqliteEventBusDatabase should implement EventBusDatabase."""
        db_path = os.path.join(temp_dir, "events.db")
        db = SqliteEventBusDatabase(db_path)
        assert isinstance(db, EventBusDatabase)

    def test_tinydb_implements_abc(self, temp_dir: str):
        """TinyDBEventBusDatabase should implement EventBusDatabase."""
        db_path = os.path.join(temp_dir, "events.json")
        db = TinyDBEventBusDatabase(db_path)
        assert isinstance(db, EventBusDatabase)


class TestFactoryFunctions:
    """Tests for factory functions that select storage backends."""

    def test_get_evaluation_row_store_default_tinydb(self, temp_dir: str, monkeypatch):
        """Default should be TinyDB."""
        monkeypatch.delenv("EP_STORAGE", raising=False)

        from eval_protocol.dataset_logger import get_evaluation_row_store

        db_path = os.path.join(temp_dir, "test.json")
        store = get_evaluation_row_store(db_path)

        assert isinstance(store, TinyDBEvaluationRowStore)

    def test_get_evaluation_row_store_sqlite(self, temp_dir: str, monkeypatch):
        """EP_STORAGE=sqlite should use SQLite."""
        monkeypatch.setenv("EP_STORAGE", "sqlite")

        from eval_protocol.dataset_logger import get_evaluation_row_store

        db_path = os.path.join(temp_dir, "test.db")
        store = get_evaluation_row_store(db_path)

        assert isinstance(store, SqliteEvaluationRowStore)

    def test_get_event_bus_database_default_tinydb(self, temp_dir: str, monkeypatch):
        """Default should be TinyDB."""
        monkeypatch.delenv("EP_STORAGE", raising=False)

        from eval_protocol.event_bus import get_event_bus_database

        db_path = os.path.join(temp_dir, "events.json")
        db = get_event_bus_database(db_path)

        assert isinstance(db, TinyDBEventBusDatabase)

    def test_get_event_bus_database_sqlite(self, temp_dir: str, monkeypatch):
        """EP_STORAGE=sqlite should use SQLite."""
        monkeypatch.setenv("EP_STORAGE", "sqlite")

        from eval_protocol.event_bus import get_event_bus_database

        db_path = os.path.join(temp_dir, "events.db")
        db = get_event_bus_database(db_path)

        assert isinstance(db, SqliteEventBusDatabase)


class TestCrossProcessCacheInvalidation:
    """
    Tests that query cache is properly invalidated when another process modifies the database.

    This simulates cross-process scenarios by creating separate store instances
    pointing to the same database file. Each instance represents a different "process"
    that might have cached query results.
    """

    @pytest.mark.parametrize(
        "store_class,file_ext",
        [
            (TinyDBEvaluationRowStore, ".json"),
            (SqliteEvaluationRowStore, ".db"),
        ],
    )
    def test_evaluation_row_store_sees_writes_from_other_process(self, temp_dir: str, store_class, file_ext: str):
        """
        Ensure a store instance can read fresh data written by another instance.

        This verifies that cached query results don't prevent seeing new data
        written by a separate process.
        """
        db_path = os.path.join(temp_dir, f"test{file_ext}")

        # Simulate two processes with separate store instances
        process1_store = store_class(db_path)
        process2_store = store_class(db_path)

        # Process 1 reads initially (may cache empty result)
        initial_rows = process1_store.read_rows()
        assert len(initial_rows) == 0

        # Process 2 writes new data
        data = {
            "execution_metadata": {"rollout_id": "cross-process-test"},
            "input_metadata": {"row_id": "row-from-process-2"},
            "messages": [{"role": "user", "content": "Hello from process 2"}],
        }
        process2_store.upsert_row(data)

        # Process 1 should see the new data (cache should be invalidated/bypassed)
        rows = process1_store.read_rows()
        assert len(rows) == 1
        assert rows[0]["execution_metadata"]["rollout_id"] == "cross-process-test"
        assert rows[0]["input_metadata"]["row_id"] == "row-from-process-2"

    @pytest.mark.parametrize(
        "store_class,file_ext",
        [
            (TinyDBEvaluationRowStore, ".json"),
            (SqliteEvaluationRowStore, ".db"),
        ],
    )
    def test_evaluation_row_store_sees_updates_from_other_process(self, temp_dir: str, store_class, file_ext: str):
        """
        Ensure a store instance sees updates made by another instance.

        This verifies that cached query results are properly invalidated
        when another process updates existing data.
        """
        db_path = os.path.join(temp_dir, f"test{file_ext}")

        # Both processes start with the same initial data
        process1_store = store_class(db_path)
        initial_data = {
            "execution_metadata": {"rollout_id": "shared-rollout"},
            "input_metadata": {"row_id": "initial-row"},
            "value": "initial",
        }
        process1_store.upsert_row(initial_data)

        # Process 2 opens the same database
        process2_store = store_class(db_path)

        # Process 1 reads and potentially caches the result
        rows = process1_store.read_rows(rollout_id="shared-rollout")
        assert len(rows) == 1
        assert rows[0]["value"] == "initial"

        # Process 2 updates the data
        updated_data = {
            "execution_metadata": {"rollout_id": "shared-rollout"},
            "input_metadata": {"row_id": "updated-row"},
            "value": "updated-by-process-2",
        }
        process2_store.upsert_row(updated_data)

        # Process 1 should see the updated data
        rows = process1_store.read_rows(rollout_id="shared-rollout")
        assert len(rows) == 1
        assert rows[0]["value"] == "updated-by-process-2"
        assert rows[0]["input_metadata"]["row_id"] == "updated-row"

    @pytest.mark.parametrize(
        "db_class,file_ext",
        [
            (TinyDBEventBusDatabase, ".json"),
            (SqliteEventBusDatabase, ".db"),
        ],
    )
    def test_event_bus_database_sees_events_from_other_process(self, temp_dir: str, db_class, file_ext: str):
        """
        Ensure an event bus instance can read events published by another instance.

        This verifies that cached query results don't prevent seeing new events
        written by a separate process.
        """
        db_path = os.path.join(temp_dir, f"events{file_ext}")

        # Simulate two processes with separate event bus instances
        process1_db = db_class(db_path)
        process2_db = db_class(db_path)

        # Process 1 checks for events initially (may cache empty result)
        initial_events = process1_db.get_unprocessed_events("process-1")
        assert len(initial_events) == 0

        # Process 2 publishes an event
        process2_db.publish_event("test_event", {"key": "value"}, "process-2")

        # Process 1 should see the new event (cache should be invalidated/bypassed)
        events = process1_db.get_unprocessed_events("process-1")
        assert len(events) == 1
        assert events[0]["event_type"] == "test_event"
        assert events[0]["data"] == {"key": "value"}
        assert events[0]["process_id"] == "process-2"

    @pytest.mark.parametrize(
        "db_class,file_ext",
        [
            (TinyDBEventBusDatabase, ".json"),
            (SqliteEventBusDatabase, ".db"),
        ],
    )
    def test_event_bus_database_sees_processed_status_from_other_process(self, temp_dir: str, db_class, file_ext: str):
        """
        Ensure an event bus instance sees when another instance marks events as processed.

        This verifies that cached query results are properly invalidated
        when another process updates event status.
        """
        db_path = os.path.join(temp_dir, f"events{file_ext}")

        # Process 1 publishes an event
        process1_db = db_class(db_path)
        process1_db.publish_event("test_event", {"key": "value"}, "process-1")

        # Process 2 opens the same database and sees the event
        process2_db = db_class(db_path)
        events = process2_db.get_unprocessed_events("process-2")
        assert len(events) == 1

        # Process 3 opens and marks the event as processed
        process3_db = db_class(db_path)
        events = process3_db.get_unprocessed_events("process-3")
        assert len(events) == 1
        process3_db.mark_event_processed(events[0]["event_id"])

        # Process 2 should no longer see the event (it's been processed)
        events = process2_db.get_unprocessed_events("process-2")
        assert len(events) == 0


class TestBackwardsCompatibility:
    """Tests for backwards compatibility aliases."""

    def test_sqlite_dataset_logger_adapter_alias(self):
        """SqliteDatasetLoggerAdapter should be an alias for DatasetLoggerAdapter."""
        from eval_protocol.dataset_logger.dataset_logger_adapter import DatasetLoggerAdapter
        from eval_protocol.dataset_logger.sqlite_dataset_logger_adapter import SqliteDatasetLoggerAdapter

        assert SqliteDatasetLoggerAdapter is DatasetLoggerAdapter

    def test_sqlite_event_bus_alias(self):
        """SqliteEventBus should be an alias for CrossProcessEventBus."""
        from eval_protocol.event_bus.cross_process_event_bus import CrossProcessEventBus
        from eval_protocol.event_bus.sqlite_event_bus import SqliteEventBus

        assert SqliteEventBus is CrossProcessEventBus
