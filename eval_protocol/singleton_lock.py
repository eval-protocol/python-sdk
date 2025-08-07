"""
Singleton Lock Management

This module provides file-based singleton lock functionality for ensuring only one
instance of a process can run at a time across the system.

The lock mechanism uses two files:
- A PID file that contains the process ID of the lock holder
- A lock file that serves as a marker for the lock

This approach provides atomic lock acquisition and proper cleanup of stale locks
from terminated processes.
"""

import os
from pathlib import Path
from typing import Optional, Tuple


def get_lock_file_paths(base_dir: Path, lock_name: str) -> Tuple[Path, Path]:
    """
    Get the lock file paths for a given lock name.

    Args:
        base_dir: Base directory where lock files should be stored
        lock_name: Name identifier for the lock (e.g., "watcher", "server")

    Returns:
        Tuple of (lock_file_path, pid_file_path)
    """
    lock_file_path = base_dir / f"{lock_name}.lock"
    pid_file_path = base_dir / f"{lock_name}.pid"
    return lock_file_path, pid_file_path


def acquire_singleton_lock(base_dir: Path, lock_name: str) -> Optional[int]:
    """
    Try to acquire the singleton lock. Returns the PID of the current holder if failed.

    Args:
        base_dir: Base directory where lock files should be stored
        lock_name: Name identifier for the lock

    Returns:
        None if lock acquired successfully, otherwise the PID of the current holder
    """
    lock_file_path, pid_file_path = get_lock_file_paths(base_dir, lock_name)

    # First, check if PID file exists and contains a running process
    if pid_file_path.exists():
        try:
            with open(pid_file_path, "r") as pid_file:
                content = pid_file.read().strip()
                if content.isdigit():
                    pid = int(content)
                    # Check if the process is still running
                    try:
                        os.kill(pid, 0)
                        # Process is running, we can't acquire the lock
                        return pid
                    except OSError:
                        # Process is not running, clean up stale files
                        pass
        except (IOError, OSError):
            pass

    # Try to create the PID file atomically
    temp_pid_file = None
    try:
        # Use atomic file creation
        temp_pid_file = pid_file_path.with_suffix(".tmp")
        with open(temp_pid_file, "w") as temp_file:
            temp_file.write(str(os.getpid()))
            temp_file.flush()
            os.fsync(temp_file.fileno())

        # Atomically move the temp file to the final location
        temp_pid_file.rename(pid_file_path)

        # Create the lock file to indicate we have the lock
        with open(lock_file_path, "w") as lock_file:
            lock_file.write(str(os.getpid()))
            lock_file.flush()
            os.fsync(lock_file.fileno())

        return None  # Successfully acquired lock

    except (IOError, OSError) as e:
        # Failed to acquire lock
        try:
            if temp_pid_file and temp_pid_file.exists():
                temp_pid_file.unlink()
        except (IOError, OSError):
            pass

        # Check if someone else got the lock
        if pid_file_path.exists():
            try:
                with open(pid_file_path, "r") as pid_file:
                    content = pid_file.read().strip()
                    if content.isdigit():
                        return int(content)
            except (IOError, OSError):
                pass

        return 999999  # Dummy PID to indicate lock is held


def release_singleton_lock(base_dir: Path, lock_name: str) -> None:
    """
    Release the singleton lock.

    Args:
        base_dir: Base directory where lock files are stored
        lock_name: Name identifier for the lock
    """
    lock_file_path, pid_file_path = get_lock_file_paths(base_dir, lock_name)
    try:
        if pid_file_path.exists():
            pid_file_path.unlink()
        if lock_file_path.exists():
            lock_file_path.unlink()
    except (IOError, OSError):
        pass


def is_process_running(pid: int) -> bool:
    """
    Check if a process is still running.

    Args:
        pid: Process ID to check

    Returns:
        True if the process is running, False otherwise
    """
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def is_lock_held(base_dir: Path, lock_name: str) -> bool:
    """
    Check if a lock is currently held by a running process.

    Args:
        base_dir: Base directory where lock files are stored
        lock_name: Name identifier for the lock

    Returns:
        True if the lock is held by a running process, False otherwise
    """
    _, pid_file_path = get_lock_file_paths(base_dir, lock_name)

    try:
        if pid_file_path.exists():
            with open(pid_file_path, "r") as pid_file:
                content = pid_file.read().strip()
                if content.isdigit():
                    pid = int(content)
                    if is_process_running(pid):
                        return True
    except (IOError, OSError):
        pass

    return False


def get_lock_holder_pid(base_dir: Path, lock_name: str) -> Optional[int]:
    """
    Get the PID of the process currently holding the lock.

    Args:
        base_dir: Base directory where lock files are stored
        lock_name: Name identifier for the lock

    Returns:
        PID of the lock holder if the lock is held by a running process, None otherwise
    """
    _, pid_file_path = get_lock_file_paths(base_dir, lock_name)
    try:
        if pid_file_path.exists():
            with open(pid_file_path, "r") as pid_file:
                content = pid_file.read().strip()
                if content.isdigit():
                    pid = int(content)
                    if is_process_running(pid):
                        return pid
    except (IOError, OSError):
        pass
    return None


def cleanup_stale_lock(base_dir: Path, lock_name: str) -> bool:
    """
    Clean up a stale lock (lock files exist but process is not running).

    Args:
        base_dir: Base directory where lock files are stored
        lock_name: Name identifier for the lock

    Returns:
        True if stale lock was cleaned up, False if no cleanup was needed
    """
    lock_file_path, pid_file_path = get_lock_file_paths(base_dir, lock_name)

    # Check if PID file exists but process is not running
    if pid_file_path.exists():
        try:
            with open(pid_file_path, "r") as pid_file:
                content = pid_file.read().strip()
                if content.isdigit():
                    pid = int(content)
                    if not is_process_running(pid):
                        # Process is not running, clean up stale files
                        release_singleton_lock(base_dir, lock_name)
                        return True
        except (IOError, OSError):
            # If we can't read the PID file, clean it up
            release_singleton_lock(base_dir, lock_name)
            return True

    return False
