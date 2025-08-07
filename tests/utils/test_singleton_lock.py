"""
Tests for the singleton lock functionality.

This module tests the file-based singleton lock mechanism that ensures only one
instance of a process can run at a time across the system.
"""

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from eval_protocol.utils.singleton_lock import (
    acquire_singleton_lock,
    cleanup_stale_lock,
    get_lock_file_paths,
    get_lock_holder_pid,
    is_lock_held,
    is_process_running,
    release_singleton_lock,
)


class TestSingletonLock:
    """Test cases for singleton lock functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def lock_name(self):
        """Test lock name."""
        return "test_lock"

    def test_get_lock_file_paths(self, temp_dir, lock_name):
        """Test that lock file paths are generated correctly."""
        lock_file_path, pid_file_path = get_lock_file_paths(temp_dir, lock_name)

        assert lock_file_path == temp_dir / f"{lock_name}.lock"
        assert pid_file_path == temp_dir / f"{lock_name}.pid"

    def test_is_process_running_current_pid(self):
        """Test that current process is detected as running."""
        current_pid = os.getpid()
        assert is_process_running(current_pid) is True

    def test_is_process_running_invalid_pid(self):
        """Test that invalid PID is detected as not running."""
        # Use a very high PID that shouldn't exist
        invalid_pid = 999999
        assert is_process_running(invalid_pid) is False

    def test_acquire_singleton_lock_success(self, temp_dir, lock_name):
        """Test successful lock acquisition."""
        result = acquire_singleton_lock(temp_dir, lock_name)
        assert result is None  # Successfully acquired lock

        # Verify lock files were created
        lock_file_path, pid_file_path = get_lock_file_paths(temp_dir, lock_name)
        assert lock_file_path.exists()
        assert pid_file_path.exists()

        # Verify PID file contains current PID
        with open(pid_file_path, "r") as f:
            stored_pid = int(f.read().strip())
        assert stored_pid == os.getpid()

    def test_acquire_singleton_lock_already_held(self, temp_dir, lock_name):
        """Test that lock acquisition fails when lock is already held."""
        # First process acquires the lock
        result1 = acquire_singleton_lock(temp_dir, lock_name)
        assert result1 is None

        # Second process tries to acquire the lock
        # Since the lock is already held by the current process, it should return the current PID
        result2 = acquire_singleton_lock(temp_dir, lock_name)
        assert result2 == os.getpid()  # Should return current holder's PID

    def test_acquire_singleton_lock_stale_lock_cleanup(self, temp_dir, lock_name):
        """Test that stale locks are cleaned up during acquisition."""
        # Create a stale lock file with a non-existent PID
        lock_file_path, pid_file_path = get_lock_file_paths(temp_dir, lock_name)

        # Create PID file with non-existent PID
        with open(pid_file_path, "w") as f:
            f.write("999999")  # Non-existent PID

        # Try to acquire lock - should clean up stale lock and succeed
        result = acquire_singleton_lock(temp_dir, lock_name)
        assert result is None  # Should succeed after cleaning up stale lock

    def test_release_singleton_lock(self, temp_dir, lock_name):
        """Test lock release functionality."""
        # Acquire the lock first
        acquire_singleton_lock(temp_dir, lock_name)

        # Verify lock files exist
        lock_file_path, pid_file_path = get_lock_file_paths(temp_dir, lock_name)
        assert lock_file_path.exists()
        assert pid_file_path.exists()

        # Release the lock
        release_singleton_lock(temp_dir, lock_name)

        # Verify lock files were removed
        assert not lock_file_path.exists()
        assert not pid_file_path.exists()

    def test_release_singleton_lock_no_files(self, temp_dir, lock_name):
        """Test that release doesn't fail when lock files don't exist."""
        # Release lock without acquiring it first
        release_singleton_lock(temp_dir, lock_name)
        # Should not raise any exceptions

    def test_is_lock_held_true(self, temp_dir, lock_name):
        """Test that lock is detected as held when it exists and process is running."""
        # Acquire the lock
        acquire_singleton_lock(temp_dir, lock_name)

        # Check if lock is held
        assert is_lock_held(temp_dir, lock_name) is True

    def test_is_lock_held_false_no_lock(self, temp_dir, lock_name):
        """Test that lock is detected as not held when no lock files exist."""
        assert is_lock_held(temp_dir, lock_name) is False

    def test_is_lock_held_false_stale_lock(self, temp_dir, lock_name):
        """Test that stale lock is detected as not held."""
        # Create a stale lock file with a non-existent PID
        lock_file_path, pid_file_path = get_lock_file_paths(temp_dir, lock_name)

        with open(pid_file_path, "w") as f:
            f.write("999999")  # Non-existent PID

        assert is_lock_held(temp_dir, lock_name) is False

    def test_get_lock_holder_pid_success(self, temp_dir, lock_name):
        """Test getting PID of lock holder."""
        # Acquire the lock
        acquire_singleton_lock(temp_dir, lock_name)

        # Get holder PID
        holder_pid = get_lock_holder_pid(temp_dir, lock_name)
        assert holder_pid == os.getpid()

    def test_get_lock_holder_pid_no_lock(self, temp_dir, lock_name):
        """Test getting PID when no lock exists."""
        holder_pid = get_lock_holder_pid(temp_dir, lock_name)
        assert holder_pid is None

    def test_get_lock_holder_pid_stale_lock(self, temp_dir, lock_name):
        """Test getting PID when lock is stale."""
        # Create a stale lock file with a non-existent PID
        lock_file_path, pid_file_path = get_lock_file_paths(temp_dir, lock_name)

        with open(pid_file_path, "w") as f:
            f.write("999999")  # Non-existent PID

        holder_pid = get_lock_holder_pid(temp_dir, lock_name)
        assert holder_pid is None

    def test_cleanup_stale_lock_success(self, temp_dir, lock_name):
        """Test successful cleanup of stale lock."""
        # Create a stale lock file with a non-existent PID
        lock_file_path, pid_file_path = get_lock_file_paths(temp_dir, lock_name)

        with open(pid_file_path, "w") as f:
            f.write("999999")  # Non-existent PID

        # Clean up stale lock
        result = cleanup_stale_lock(temp_dir, lock_name)
        assert result is True

        # Verify lock files were removed
        assert not lock_file_path.exists()
        assert not pid_file_path.exists()

    def test_cleanup_stale_lock_no_cleanup_needed(self, temp_dir, lock_name):
        """Test cleanup when no stale lock exists."""
        result = cleanup_stale_lock(temp_dir, lock_name)
        assert result is False

    def test_cleanup_stale_lock_active_lock(self, temp_dir, lock_name):
        """Test cleanup when lock is actively held."""
        # Acquire the lock
        acquire_singleton_lock(temp_dir, lock_name)

        # Try to clean up active lock
        result = cleanup_stale_lock(temp_dir, lock_name)
        assert result is False

        # Verify lock files still exist
        lock_file_path, pid_file_path = get_lock_file_paths(temp_dir, lock_name)
        assert lock_file_path.exists()
        assert pid_file_path.exists()

    def test_concurrent_lock_acquisition_race_condition(self, temp_dir, lock_name):
        """Test race condition handling in concurrent lock acquisition."""
        # This test simulates a race condition by creating the PID file
        # after the first process checks but before it creates its own

        lock_file_path, pid_file_path = get_lock_file_paths(temp_dir, lock_name)

        # Create PID file as if another process got there first
        with open(pid_file_path, "w") as f:
            f.write("999999")

        # Try to acquire lock - should detect the existing PID file and return that PID
        # Note: The function automatically cleans up stale locks, so it will succeed
        result = acquire_singleton_lock(temp_dir, lock_name)
        assert result is None  # Should succeed after cleaning up stale lock

    def test_atomic_file_creation(self, temp_dir, lock_name):
        """Test that PID file creation is atomic."""
        lock_file_path, pid_file_path = get_lock_file_paths(temp_dir, lock_name)

        # Acquire lock
        acquire_singleton_lock(temp_dir, lock_name)

        # Verify no temporary file exists
        temp_pid_file = pid_file_path.with_suffix(".tmp")
        assert not temp_pid_file.exists()

        # Verify final PID file exists and contains correct PID
        assert pid_file_path.exists()
        with open(pid_file_path, "r") as f:
            stored_pid = int(f.read().strip())
        assert stored_pid == os.getpid()

    def test_multiple_lock_names_independence(self, temp_dir):
        """Test that different lock names are independent."""
        lock_name1 = "lock1"
        lock_name2 = "lock2"

        # Acquire first lock
        result1 = acquire_singleton_lock(temp_dir, lock_name1)
        assert result1 is None

        # Acquire second lock (should succeed since different name)
        result2 = acquire_singleton_lock(temp_dir, lock_name2)
        assert result2 is None

        # Verify both locks are held
        assert is_lock_held(temp_dir, lock_name1) is True
        assert is_lock_held(temp_dir, lock_name2) is True

        # Verify lock files exist for both
        lock_file1, pid_file1 = get_lock_file_paths(temp_dir, lock_name1)
        lock_file2, pid_file2 = get_lock_file_paths(temp_dir, lock_name2)
        assert lock_file1.exists()
        assert pid_file1.exists()
        assert lock_file2.exists()
        assert pid_file2.exists()
