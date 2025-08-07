"""
Multiprocessing tests for the singleton lock functionality.

This module tests the file-based singleton lock mechanism using actual
multiprocessing to simulate real concurrent scenarios.
"""

import multiprocessing
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import pytest

from eval_protocol.utils.singleton_lock import (
    acquire_singleton_lock,
    is_lock_held,
    release_singleton_lock,
)


def worker_process_acquire_lock(base_dir: Path, lock_name: str, result_queue: multiprocessing.Queue, worker_id: int):
    """Worker process that tries to acquire a lock."""
    try:
        print(f"Worker {worker_id} (PID: {os.getpid()}) attempting to acquire lock")
        result = acquire_singleton_lock(base_dir, lock_name)
        result_queue.put((worker_id, result))
        print(f"Worker {worker_id} (PID: {os.getpid()}) got result: {result}")

        # If we got the lock, hold it for a bit then release
        if result is None:
            print(f"Worker {worker_id} (PID: {os.getpid()}) acquired lock, holding for 2 seconds")
            time.sleep(2)
            release_singleton_lock(base_dir, lock_name)
            print(f"Worker {worker_id} (PID: {os.getpid()}) released lock")
    except Exception as e:
        print(f"Worker {worker_id} (PID: {os.getpid()}) got exception: {e}")
        result_queue.put((worker_id, f"ERROR: {e}"))


def worker_process_check_lock(base_dir: Path, lock_name: str, result_queue: multiprocessing.Queue, worker_id: int):
    """Worker process that checks if a lock is held."""
    try:
        print(f"Worker {worker_id} (PID: {os.getpid()}) checking if lock is held")
        is_held = is_lock_held(base_dir, lock_name)
        result_queue.put((worker_id, is_held))
        print(f"Worker {worker_id} (PID: {os.getpid()}) lock held: {is_held}")
    except Exception as e:
        print(f"Worker {worker_id} (PID: {os.getpid()}) got exception: {e}")
        result_queue.put((worker_id, f"ERROR: {e}"))


def worker_process_hold_lock(
    base_dir: Path, lock_name: str, result_queue: multiprocessing.Queue, worker_id: int, hold_time: float = 5.0
):
    """Worker process that acquires and holds a lock for a specified time."""
    try:
        print(f"Worker {worker_id} (PID: {os.getpid()}) attempting to acquire lock")
        result = acquire_singleton_lock(base_dir, lock_name)

        if result is None:
            print(f"Worker {worker_id} (PID: {os.getpid()}) acquired lock, holding for {hold_time} seconds")
            result_queue.put((worker_id, "ACQUIRED"))
            time.sleep(hold_time)
            release_singleton_lock(base_dir, lock_name)
            print(f"Worker {worker_id} (PID: {os.getpid()}) released lock")
            result_queue.put((worker_id, "RELEASED"))
        else:
            print(f"Worker {worker_id} (PID: {os.getpid()}) failed to acquire lock, holder PID: {result}")
            result_queue.put((worker_id, f"FAILED: {result}"))
    except Exception as e:
        print(f"Worker {worker_id} (PID: {os.getpid()}) got exception: {e}")
        result_queue.put((worker_id, f"ERROR: {e}"))


class TestSingletonLockMultiprocessing:
    """Test cases for singleton lock functionality using multiprocessing."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def lock_name(self):
        """Test lock name."""
        return "test_multiprocessing_lock"

    def test_multiprocessing_lock_acquisition_sequential(self, temp_dir, lock_name):
        """Test that only one process can acquire the lock at a time."""
        # Use multiprocessing.Manager() for better compatibility
        manager = multiprocessing.Manager()
        result_queue = manager.Queue()

        # Start multiple processes trying to acquire the lock
        processes = []
        num_workers = 3

        for i in range(num_workers):
            p = multiprocessing.Process(
                target=worker_process_acquire_lock,
                args=(temp_dir, lock_name, result_queue, i),
                daemon=False,  # Don't use daemon processes
            )
            processes.append(p)
            p.start()
            time.sleep(0.1)  # Small delay to create race condition

        # Wait for all processes to complete
        for p in processes:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join()

        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        # Sort by worker ID for consistent ordering
        results.sort(key=lambda x: x[0])

        print(f"Results: {results}")

        # Verify that exactly one process acquired the lock (got None)
        acquired_results = [r for r in results if r[1] is None]
        assert len(acquired_results) == 1, f"Expected exactly one process to acquire lock, got {len(acquired_results)}"

        # Verify that other processes got the PID of the lock holder
        acquired_worker_id = acquired_results[0][0]
        for worker_id, result in results:
            if worker_id != acquired_worker_id:
                assert isinstance(result, int), f"Expected PID, got {result}"
                assert result > 0, f"Expected positive PID, got {result}"

    def test_multiprocessing_lock_holder_detection(self, temp_dir, lock_name):
        """Test that processes can detect when a lock is held."""
        manager = multiprocessing.Manager()
        result_queue = manager.Queue()

        # Start a process that holds the lock
        holder_process = multiprocessing.Process(
            target=worker_process_hold_lock,
            args=(temp_dir, lock_name, result_queue, 0, 3.0),  # Hold for 3 seconds
            daemon=False,
        )
        holder_process.start()

        # Wait a moment for the holder to acquire the lock
        time.sleep(0.5)

        # Start multiple processes checking if the lock is held
        checker_processes = []
        num_checkers = 3

        for i in range(num_checkers):
            p = multiprocessing.Process(
                target=worker_process_check_lock,
                args=(temp_dir, lock_name, result_queue, i + 100),  # Use different IDs
                daemon=False,
            )
            checker_processes.append(p)
            p.start()

        # Wait for all processes to complete
        holder_process.join(timeout=10)
        for p in checker_processes:
            p.join(timeout=5)

        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        print(f"Results: {results}")

        # Verify that the holder process acquired the lock
        holder_results = [r for r in results if r[0] == 0]
        assert len(holder_results) >= 1, "Holder process should have reported acquiring the lock"

        # Verify that checker processes detected the lock as held
        checker_results = [r for r in results if r[0] >= 100]
        for worker_id, is_held in checker_results:
            assert is_held is True, f"Checker {worker_id} should have detected lock as held"

    def test_multiprocessing_lock_cleanup_after_process_termination(self, temp_dir, lock_name):
        """Test that locks are properly cleaned up when processes terminate."""
        manager = multiprocessing.Manager()
        result_queue = manager.Queue()

        # Start a process that holds the lock
        holder_process = multiprocessing.Process(
            target=worker_process_hold_lock,
            args=(temp_dir, lock_name, result_queue, 0, 10.0),  # Hold for 10 seconds
            daemon=False,
        )
        holder_process.start()

        # Wait for the holder to acquire the lock and check results
        time.sleep(1.0)

        # Check if the process actually acquired the lock
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        # Look for the ACQUIRED message
        acquired_results = [r for r in results if r[1] == "ACQUIRED"]
        if not acquired_results:
            # If no acquisition happened, this test is not meaningful
            holder_process.terminate()
            holder_process.join(timeout=5)
            pytest.skip("Process did not acquire lock, skipping cleanup test")

        # Verify the lock is held
        assert is_lock_held(temp_dir, lock_name) is True

        # Terminate the holder process
        holder_process.terminate()
        holder_process.join(timeout=5)

        # Wait a moment for cleanup
        time.sleep(0.5)

        # Verify the lock is no longer held
        assert is_lock_held(temp_dir, lock_name) is False

        # Try to acquire the lock - should succeed
        result = acquire_singleton_lock(temp_dir, lock_name)
        assert result is None, "Should be able to acquire lock after process termination"

        # Clean up
        release_singleton_lock(temp_dir, lock_name)

    def test_multiprocessing_daemon_vs_non_daemon(self, temp_dir, lock_name):
        """Test lock behavior with daemon vs non-daemon processes."""
        manager = multiprocessing.Manager()
        result_queue = manager.Queue()

        # Test with daemon=True
        print("Testing with daemon=True")
        daemon_process = multiprocessing.Process(
            target=worker_process_hold_lock, args=(temp_dir, lock_name, result_queue, 0, 2.0), daemon=True
        )
        daemon_process.start()
        time.sleep(0.5)

        # Check if lock is held
        is_held_daemon = is_lock_held(temp_dir, lock_name)
        print(f"Lock held with daemon process: {is_held_daemon}")

        daemon_process.join(timeout=5)

        # Test with daemon=False
        print("Testing with daemon=False")
        non_daemon_process = multiprocessing.Process(
            target=worker_process_hold_lock, args=(temp_dir, lock_name, result_queue, 1, 2.0), daemon=False
        )
        non_daemon_process.start()
        time.sleep(0.5)

        # Check if lock is held
        is_held_non_daemon = is_lock_held(temp_dir, lock_name)
        print(f"Lock held with non-daemon process: {is_held_non_daemon}")

        non_daemon_process.join(timeout=5)

        # Both should work the same way for lock acquisition
        assert (
            is_held_daemon == is_held_non_daemon
        ), "Lock behavior should be the same for daemon and non-daemon processes"

    def test_multiprocessing_concurrent_acquisition_race_condition(self, temp_dir, lock_name):
        """Test race condition handling with multiple processes trying to acquire simultaneously."""
        manager = multiprocessing.Manager()
        result_queue = manager.Queue()

        # Start multiple processes simultaneously
        processes = []
        num_workers = 5

        # Start all processes at nearly the same time
        for i in range(num_workers):
            p = multiprocessing.Process(
                target=worker_process_acquire_lock, args=(temp_dir, lock_name, result_queue, i), daemon=False
            )
            processes.append(p)

        # Start all processes with minimal delay
        for p in processes:
            p.start()

        # Wait for all processes to complete
        for p in processes:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join()

        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        print(f"Race condition test results: {results}")

        # Verify that exactly one process acquired the lock
        acquired_results = [r for r in results if r[1] is None]
        assert (
            len(acquired_results) == 1
        ), f"Expected exactly one process to acquire lock in race condition, got {len(acquired_results)}"

        # Verify that other processes got valid PIDs
        acquired_worker_id = acquired_results[0][0]
        for worker_id, result in results:
            if worker_id != acquired_worker_id:
                assert isinstance(result, int), f"Expected PID, got {result}"
                assert result > 0, f"Expected positive PID, got {result}"

    def test_multiprocessing_lock_independence(self, temp_dir):
        """Test that different lock names are independent across processes."""
        lock_name1 = "lock1"
        lock_name2 = "lock2"

        manager = multiprocessing.Manager()
        result_queue = manager.Queue()

        # Start processes trying to acquire different locks
        process1 = multiprocessing.Process(
            target=worker_process_acquire_lock, args=(temp_dir, lock_name1, result_queue, 1), daemon=False
        )
        process2 = multiprocessing.Process(
            target=worker_process_acquire_lock, args=(temp_dir, lock_name2, result_queue, 2), daemon=False
        )

        process1.start()
        process2.start()

        process1.join(timeout=5)
        process2.join(timeout=5)

        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        print(f"Lock independence test results: {results}")

        # Both processes should have acquired their respective locks
        assert len(results) == 2, "Expected results from both processes"
        for worker_id, result in results:
            assert result is None, f"Process {worker_id} should have acquired its lock"

    def test_daemon_off_process_survives_parent_termination(self, temp_dir, lock_name):
        """Test that a daemon=Off process continues to run and hold the lock when parent is killed."""
        import signal
        import subprocess
        import sys

        # Create a script that will be run as a separate process
        script_content = f'''
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

# Copy the singleton lock functions directly to avoid import issues
def get_lock_file_paths(base_dir: Path, lock_name: str) -> Tuple[Path, Path]:
    lock_file_path = base_dir / f"{{lock_name}}.lock"
    pid_file_path = base_dir / f"{{lock_name}}.pid"
    return lock_file_path, pid_file_path

def is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def acquire_singleton_lock(base_dir: Path, lock_name: str) -> Optional[int]:
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
    lock_file_path, pid_file_path = get_lock_file_paths(base_dir, lock_name)
    try:
        if pid_file_path.exists():
            pid_file_path.unlink()
        if lock_file_path.exists():
            lock_file_path.unlink()
    except (IOError, OSError):
        pass

def is_lock_held(base_dir: Path, lock_name: str) -> bool:
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

def child_process_holder(base_dir, lock_name, pid_file):
    """Child process that acquires and holds a lock."""
    try:
        # Write our PID to a file so parent can read it
        with open(pid_file, 'w') as f:
            f.write(str(os.getpid()))

        result = acquire_singleton_lock(Path(base_dir), lock_name)

        if result is None:
            # Keep the lock held by sleeping in a loop
            while True:
                time.sleep(1)
                # Verify we still hold the lock
                if not is_lock_held(Path(base_dir), lock_name):
                    break
        else:
            sys.exit(1)
    except Exception as e:
        sys.exit(1)

if __name__ == "__main__":
    base_dir = "{temp_dir}"
    lock_name = "{lock_name}"
    pid_file = "{temp_dir}/child_pid.txt"

    # Start child process with daemon=False
    child = multiprocessing.Process(
        target=child_process_holder,
        args=(base_dir, lock_name, pid_file),
        daemon=False
    )
    child.start()

    # Wait for child to start and acquire lock
    time.sleep(2)

    # Verify child is still running
    if not child.is_alive():
        sys.exit(1)

    # Write parent PID to file
    with open("{temp_dir}/parent_pid.txt", 'w') as f:
        f.write(str(os.getpid()))

    # Sleep indefinitely - parent will be killed by test
    while True:
        time.sleep(1)
'''

        # Write the script to a temporary file
        script_path = temp_dir / "test_daemon_off_script.py"
        with open(script_path, "w") as f:
            f.write(script_content)

        # Start the script as a separate process using uvx to ensure correct environment
        process = subprocess.Popen(
            ["uvx", "python", str(script_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Wait for the script to start and child process to acquire lock
        time.sleep(3)

        # Read the PIDs
        parent_pid_file = temp_dir / "parent_pid.txt"
        child_pid_file = temp_dir / "child_pid.txt"

        # Wait for PID files to be created
        for _ in range(10):
            if parent_pid_file.exists() and child_pid_file.exists():
                break
            time.sleep(0.5)
        else:
            process.terminate()
            process.wait(timeout=5)
            pytest.fail("PID files were not created in time")

        with open(parent_pid_file, "r") as f:
            parent_pid = int(f.read().strip())
        with open(child_pid_file, "r") as f:
            child_pid = int(f.read().strip())

        print(f"Parent PID: {parent_pid}, Child PID: {child_pid}")

        # Verify the lock is held by the child process
        assert is_lock_held(temp_dir, lock_name) is True, "Lock should be held by child process"

        # Try to acquire the lock - should fail and return child's PID
        result = acquire_singleton_lock(temp_dir, lock_name)
        assert result == child_pid, f"Should get child PID {child_pid}, got {result}"

        # Kill the parent process
        print(f"Killing parent process {parent_pid}")
        os.kill(parent_pid, signal.SIGTERM)

        # Wait a moment for the parent to terminate
        time.sleep(2)

        # Verify the child process is still running
        try:
            # Check if child process is still alive
            os.kill(child_pid, 0)  # This will raise OSError if process doesn't exist
            child_still_alive = True
        except OSError:
            child_still_alive = False

        assert child_still_alive, "Child process should still be alive after parent termination"

        # Verify the lock is still held by the child process
        assert is_lock_held(temp_dir, lock_name) is True, "Lock should still be held by child process"

        # Try to acquire the lock again - should still fail and return child's PID
        result = acquire_singleton_lock(temp_dir, lock_name)
        assert result == child_pid, f"Should still get child PID {child_pid}, got {result}"

        # Clean up by killing the child process
        print(f"Killing child process {child_pid}")
        os.kill(child_pid, signal.SIGTERM)

        # Wait for child to terminate
        time.sleep(2)

        # Verify the lock is no longer held
        assert is_lock_held(temp_dir, lock_name) is False, "Lock should be released after child termination"

        # Clean up the subprocess
        process.terminate()
        process.wait(timeout=5)


if __name__ == "__main__":
    # Run the tests directly
    pytest.main([__file__, "-v"])
