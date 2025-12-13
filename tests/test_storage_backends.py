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
