"""
Tests for SQLite hardening and concurrency safety.

These tests verify that:
1. WAL mode and other concurrency pragmas are correctly applied
2. Database corruption detection works
3. Auto-repair functionality works
4. Multiple concurrent operations don't corrupt the database
"""

import os
import sqlite3
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

import pytest

from eval_protocol.event_bus.sqlite_event_bus_database import (
    SQLITE_HARDENED_PRAGMAS,
    DatabaseCorruptedError,
    SqliteEventBusDatabase,
    _backup_and_remove_database,
    check_and_repair_database,
)
from eval_protocol.dataset_logger.sqlite_evaluation_row_store import SqliteEvaluationRowStore


class TestSqliteHardenedPragmas:
    """Test that hardened pragmas are correctly defined and applied."""

    def test_pragmas_are_defined(self):
        """Test that all required pragmas are defined."""
        required_pragmas = [
            "journal_mode",
            "synchronous",
            "busy_timeout",
            "wal_autocheckpoint",
            "cache_size",
            "foreign_keys",
            "temp_store",
        ]
        for pragma in required_pragmas:
            assert pragma in SQLITE_HARDENED_PRAGMAS, f"Missing pragma: {pragma}"

    def test_wal_mode_is_enabled(self):
        """Test that WAL mode is set in pragmas."""
        assert SQLITE_HARDENED_PRAGMAS["journal_mode"] == "wal"

    def test_busy_timeout_is_set(self):
        """Test that busy_timeout is set to a reasonable value."""
        # Should be at least 10 seconds (10000ms)
        assert SQLITE_HARDENED_PRAGMAS["busy_timeout"] >= 10000


class TestSqliteEventBusDatabaseHardening:
    """Test SqliteEventBusDatabase hardening features."""

    def test_creates_database_with_wal_mode(self):
        """Test that database is created with WAL journal mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = SqliteEventBusDatabase(db_path)

            cursor = db._db.execute_sql("PRAGMA journal_mode")
            journal_mode = cursor.fetchone()[0]
            assert journal_mode == "wal", f"Expected WAL mode, got {journal_mode}"

    def test_creates_database_with_busy_timeout(self):
        """Test that database has busy_timeout set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = SqliteEventBusDatabase(db_path)

            cursor = db._db.execute_sql("PRAGMA busy_timeout")
            timeout = cursor.fetchone()[0]
            assert timeout == SQLITE_HARDENED_PRAGMAS["busy_timeout"]

    def test_creates_database_with_synchronous_normal(self):
        """Test that synchronous mode is set to normal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = SqliteEventBusDatabase(db_path)

            cursor = db._db.execute_sql("PRAGMA synchronous")
            sync_mode = cursor.fetchone()[0]
            # 1 = NORMAL in SQLite
            assert sync_mode == 1, f"Expected synchronous=1 (NORMAL), got {sync_mode}"

    def test_creates_directory_if_not_exists(self):
        """Test that parent directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "subdir", "nested", "test.db")
            db = SqliteEventBusDatabase(db_path)

            assert os.path.exists(db_path)
            assert os.path.isfile(db_path)


class TestSqliteEvaluationRowStoreHardening:
    """Test SqliteEvaluationRowStore hardening features."""

    def test_creates_database_with_wal_mode(self):
        """Test that database is created with WAL journal mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "eval.db")
            store = SqliteEvaluationRowStore(db_path)

            cursor = store._db.execute_sql("PRAGMA journal_mode")
            journal_mode = cursor.fetchone()[0]
            assert journal_mode == "wal", f"Expected WAL mode, got {journal_mode}"

    def test_creates_database_with_busy_timeout(self):
        """Test that database has busy_timeout set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "eval.db")
            store = SqliteEvaluationRowStore(db_path)

            cursor = store._db.execute_sql("PRAGMA busy_timeout")
            timeout = cursor.fetchone()[0]
            assert timeout == SQLITE_HARDENED_PRAGMAS["busy_timeout"]


class TestDatabaseCorruptionDetection:
    """Test database corruption detection functionality."""

    def test_nonexistent_database_passes_check(self):
        """Test that check passes for a non-existent database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "nonexistent.db")
            result = check_and_repair_database(db_path)
            assert result is True

    def test_valid_database_passes_check(self):
        """Test that a valid database passes the integrity check."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "valid.db")

            # Create a valid database
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, data TEXT)")
            conn.execute("INSERT INTO test VALUES (1, 'test')")
            conn.commit()
            conn.close()

            result = check_and_repair_database(db_path)
            assert result is True

    def test_corrupted_file_raises_error_without_auto_repair(self):
        """Test that a corrupted file raises DatabaseCorruptedError when auto_repair=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "corrupted.db")

            # Create a corrupted file (not a valid SQLite database)
            with open(db_path, "w") as f:
                f.write("This is not a valid SQLite database!")

            with pytest.raises(DatabaseCorruptedError) as exc_info:
                check_and_repair_database(db_path, auto_repair=False)

            assert exc_info.value.db_path == db_path

    def test_corrupted_file_auto_repaired(self):
        """Test that a corrupted file is auto-repaired when auto_repair=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "corrupted.db")

            # Create a corrupted file
            with open(db_path, "w") as f:
                f.write("This is not a valid SQLite database!")

            result = check_and_repair_database(db_path, auto_repair=True)
            assert result is True

            # Original file should be removed (or renamed to backup)
            assert not os.path.exists(db_path)

    def test_corrupted_file_backup_created(self):
        """Test that a backup is created when auto-repairing a corrupted file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "corrupted.db")

            # Create a corrupted file
            with open(db_path, "w") as f:
                f.write("This is not a valid SQLite database!")

            check_and_repair_database(db_path, auto_repair=True)

            # Check for backup file
            files = os.listdir(tmpdir)
            backup_files = [f for f in files if "corrupted" in f and f != "corrupted.db"]
            assert len(backup_files) == 1
            assert "corrupted" in backup_files[0]

    def test_transient_errors_do_not_delete_database(self):
        """Test that transient I/O errors (like PermissionError) don't trigger database deletion.

        This is a regression test for a bug where the catch-all Exception handler
        would delete valid databases on transient errors like PermissionError,
        OSError, or temporary lock conflicts.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "valid.db")

            # Create a valid database
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, data TEXT)")
            conn.execute("INSERT INTO test VALUES (1, 'important data')")
            conn.commit()
            conn.close()

            # Verify the database is valid first
            result = check_and_repair_database(db_path)
            assert result is True
            assert os.path.exists(db_path)

            # The database should still exist and be valid
            conn = sqlite3.connect(db_path)
            cursor = conn.execute("SELECT data FROM test WHERE id=1")
            row = cursor.fetchone()
            conn.close()
            assert row[0] == "important data"

    def test_database_error_without_corruption_indicator_is_not_auto_repaired(self):
        """Test that DatabaseError without corruption indicators is re-raised, not auto-repaired."""
        from unittest.mock import patch, MagicMock
        from peewee import DatabaseError

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "locked.db")

            # Create a valid database first
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE test (id INTEGER)")
            conn.close()

            # Mock SqliteDatabase to raise a non-corruption DatabaseError (e.g., database locked)
            with patch("eval_protocol.event_bus.sqlite_event_bus_database.SqliteDatabase") as mock_db_class:
                mock_db = MagicMock()
                mock_db_class.return_value = mock_db
                mock_db.connect.side_effect = DatabaseError("database is locked")

                # Should re-raise the error, not delete the database
                with pytest.raises(DatabaseError) as exc_info:
                    check_and_repair_database(db_path, auto_repair=True)

                assert "locked" in str(exc_info.value)

            # Database file should still exist (not deleted)
            assert os.path.exists(db_path)


class TestBackupAndRemoveDatabase:
    """Test the backup and remove database functionality."""

    def test_backup_creates_timestamped_file(self):
        """Test that backup creates a timestamped backup file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")

            # Create a file
            with open(db_path, "w") as f:
                f.write("test content")

            _backup_and_remove_database(db_path)

            # Original should be gone
            assert not os.path.exists(db_path)

            # Backup should exist with timestamp
            files = os.listdir(tmpdir)
            assert len(files) == 1
            assert files[0].startswith("test.db.corrupted.")

    def test_removes_wal_and_shm_files(self):
        """Test that WAL and SHM files are also removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")

            # Create main db and WAL/SHM files
            for suffix in ["", "-wal", "-shm"]:
                with open(f"{db_path}{suffix}", "w") as f:
                    f.write("test")

            _backup_and_remove_database(db_path)

            # WAL and SHM should be removed
            assert not os.path.exists(f"{db_path}-wal")
            assert not os.path.exists(f"{db_path}-shm")


class TestDatabaseAutoRepairOnInit:
    """Test that databases are auto-repaired on initialization."""

    def test_event_bus_database_auto_repairs(self):
        """Test that SqliteEventBusDatabase auto-repairs corrupted database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "events.db")

            # Create a corrupted file
            with open(db_path, "w") as f:
                f.write("corrupted!")

            # Should not raise, should auto-repair
            db = SqliteEventBusDatabase(db_path, auto_repair=True)

            # Should be usable
            db.publish_event("test", {"data": "test"}, "test-process")

    def test_evaluation_row_store_auto_repairs(self):
        """Test that SqliteEvaluationRowStore auto-repairs corrupted database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "eval.db")

            # Create a corrupted file
            with open(db_path, "w") as f:
                f.write("corrupted!")

            # Should not raise, should auto-repair
            store = SqliteEvaluationRowStore(db_path, auto_repair=True)

            # Should be usable - verify by checking that it has the expected table
            cursor = store._db.execute_sql(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='evaluationrow'"
            )
            result = cursor.fetchone()
            assert result is not None


class TestConcurrencySafety:
    """Test concurrent access to SQLite databases."""

    def test_concurrent_writes_to_event_bus(self):
        """Test that concurrent writes to event bus don't fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "concurrent.db")
            db = SqliteEventBusDatabase(db_path)

            errors: List[Exception] = []
            num_threads = 10
            events_per_thread = 50

            def write_events(thread_id: int):
                try:
                    for i in range(events_per_thread):
                        db.publish_event(
                            f"event_{thread_id}_{i}",
                            {"thread": thread_id, "index": i},
                            f"process_{thread_id}",
                        )
                except Exception as e:
                    errors.append(e)

            # Run concurrent writes
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(write_events, i) for i in range(num_threads)]
                for future in futures:
                    future.result()

            # Should have no errors
            assert len(errors) == 0, f"Concurrent write errors: {errors}"

    def test_concurrent_upserts_to_evaluation_store(self):
        """Test that concurrent upserts to evaluation store don't fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "concurrent_eval.db")
            store = SqliteEvaluationRowStore(db_path)

            errors: List[Exception] = []
            num_threads = 10
            upserts_per_thread = 20

            def upsert_rows(thread_id: int):
                try:
                    for i in range(upserts_per_thread):
                        rollout_id = f"rollout_{thread_id}_{i}"
                        data = {
                            "execution_metadata": {"rollout_id": rollout_id},
                            "data": {"thread": thread_id, "index": i},
                        }
                        store.upsert_row(data)
                except Exception as e:
                    errors.append(e)

            # Run concurrent upserts
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(upsert_rows, i) for i in range(num_threads)]
                for future in futures:
                    future.result()

            # Should have no errors
            assert len(errors) == 0, f"Concurrent upsert errors: {errors}"

            # Verify all rows were written
            all_rows = store.read_rows()
            assert len(all_rows) == num_threads * upserts_per_thread

    def test_concurrent_reads_and_writes(self):
        """Test that concurrent reads and writes work correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "rw_concurrent.db")
            store = SqliteEvaluationRowStore(db_path)

            # Pre-populate with some data
            for i in range(10):
                store.upsert_row(
                    {
                        "execution_metadata": {"rollout_id": f"initial_{i}"},
                        "data": {"initial": True},
                    }
                )

            errors: List[Exception] = []
            read_counts: List[int] = []

            def writer():
                try:
                    for i in range(50):
                        store.upsert_row(
                            {
                                "execution_metadata": {"rollout_id": f"write_{i}"},
                                "data": {"written": True},
                            }
                        )
                        time.sleep(0.001)
                except Exception as e:
                    errors.append(e)

            def reader():
                try:
                    for _ in range(100):
                        rows = store.read_rows()
                        read_counts.append(len(rows))
                        time.sleep(0.001)
                except Exception as e:
                    errors.append(e)

            # Run concurrent reads and writes
            threads = [
                threading.Thread(target=writer),
                threading.Thread(target=reader),
                threading.Thread(target=reader),
            ]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Should have no errors
            assert len(errors) == 0, f"Concurrent read/write errors: {errors}"

            # All reads should have returned valid counts
            assert all(count >= 10 for count in read_counts), "Reads should return at least initial rows"


class TestDatabaseCorruptedErrorClass:
    """Test the DatabaseCorruptedError exception class."""

    def test_error_contains_db_path(self):
        """Test that error contains the database path."""
        original_error = Exception("original error")
        error = DatabaseCorruptedError("/path/to/db.sqlite", original_error)

        assert error.db_path == "/path/to/db.sqlite"
        assert error.original_error == original_error
        assert "/path/to/db.sqlite" in str(error)

    def test_error_is_exception(self):
        """Test that DatabaseCorruptedError is an Exception."""
        error = DatabaseCorruptedError("/path/to/db", Exception("test"))
        assert isinstance(error, Exception)
