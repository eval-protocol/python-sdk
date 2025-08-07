#!/usr/bin/env python3
"""
Evaluation Watcher Process

This process monitors all evaluation rows and updates any evaluations that are still
"running" but whose associated process has terminated.

Usage:
    python -m eval_protocol.pytest.eval_watcher [--check-interval <seconds>]
"""

import argparse
import fcntl
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add freeze_support for multiprocessing compatibility
if __name__ == "__main__":
    multiprocessing.freeze_support()

from eval_protocol.dataset_logger import default_logger
from eval_protocol.dataset_logger.directory_utils import find_eval_protocol_dir
from eval_protocol.models import EvaluationRow


def get_lock_file_paths() -> tuple[Path, Path]:
    """Get the lock file paths using the same directory discovery logic."""
    eval_protocol_dir = Path(find_eval_protocol_dir())
    lock_file_path = eval_protocol_dir / "watcher.lock"
    pid_file_path = eval_protocol_dir / "watcher.pid"
    return lock_file_path, pid_file_path


def acquire_singleton_lock() -> Optional[int]:
    """
    Try to acquire the singleton lock. Returns the PID of the current holder if failed.

    Returns:
        None if lock acquired successfully, otherwise the PID of the current holder
    """
    lock_file_path, pid_file_path = get_lock_file_paths()

    try:
        # Try to acquire an exclusive lock on the lock file
        with open(lock_file_path, "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            # Write our PID to the PID file
            with open(pid_file_path, "w") as pid_file:
                pid_file.write(str(os.getpid()))

            return None  # Successfully acquired lock

    except (IOError, OSError):
        # Lock is held by another process
        try:
            if pid_file_path.exists():
                with open(pid_file_path, "r") as pid_file:
                    content = pid_file.read().strip()
                    if content.isdigit():
                        return int(content)
        except (IOError, OSError):
            pass
        return None


def release_singleton_lock():
    """Release the singleton lock."""
    lock_file_path, pid_file_path = get_lock_file_paths()
    try:
        if pid_file_path.exists():
            pid_file_path.unlink()
        if lock_file_path.exists():
            lock_file_path.unlink()
    except (IOError, OSError):
        pass


def is_process_running(pid: int) -> bool:
    """Check if a process is still running."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def find_running_evaluations() -> List[EvaluationRow]:
    """Find all evaluations currently in 'running' status ."""
    running_evaluations = []
    all_rows = default_logger.read()

    for row in all_rows:
        if row.eval_metadata and row.eval_metadata.status == "running":
            running_evaluations.append(row)

    return running_evaluations


def update_evaluation_to_stopped(row: EvaluationRow, reason: str) -> None:
    """Update an evaluation row to 'stopped' status."""
    try:
        if row.eval_metadata:
            row.eval_metadata.status = "stopped"
            row.eval_metadata.passed = False

        if row.evaluation_result is not None:
            row.evaluation_result.error = reason
        else:
            from eval_protocol.models import EvaluateResult

            row.evaluation_result = EvaluateResult(
                score=0.0, is_score_valid=False, reason=f"Evaluation stopped: {reason}", error=reason
            )

        default_logger.log(row)
        print(
            f"  📝 Updated evaluation '{row.eval_metadata.name if row.eval_metadata else 'Unknown'}' (Row ID: {row.input_metadata.row_id}) (PID: {row.pid}) to stopped status"
        )

    except Exception as e:
        print(f"  ⚠️  Error updating evaluation row: {e}")


def check_and_update_terminated_evaluations() -> int:
    """Check for evaluations with terminated processes and update them to stopped status."""
    running_evaluations = find_running_evaluations()

    if not running_evaluations:
        return 0

    print(f"🔍 Checking {len(running_evaluations)} running evaluations for terminated processes...")
    for row in running_evaluations:
        print(f" Row ID: {row.input_metadata.row_id}  PID: {row.pid}")

    terminated_count = 0
    for row in running_evaluations:
        if row.pid:
            if not is_process_running(row.pid):
                update_evaluation_to_stopped(row, f"Process {row.pid} terminated")
                terminated_count += 1
        else:
            update_evaluation_to_stopped(row, f"Process {row.pid} terminated")
            terminated_count += 1

    if terminated_count > 0:
        print(f"  ✅ Updated {terminated_count} evaluations to stopped status")

    return terminated_count


def run_watcher_loop(check_interval: float) -> None:
    """Main monitoring loop."""
    print(f"🔍 Starting evaluation watcher (PID: {os.getpid()})")
    print(f"  Check interval: {check_interval} seconds")
    print("  Monitoring all evaluation rows for terminated processes")

    consecutive_empty_checks = 0
    max_empty_checks = 3

    try:
        while True:
            running_evaluations = find_running_evaluations()

            if running_evaluations:
                consecutive_empty_checks = 0
                check_and_update_terminated_evaluations()
            else:
                consecutive_empty_checks += 1
                if consecutive_empty_checks >= max_empty_checks:
                    print(
                        f"🔍 No running evaluations found for {consecutive_empty_checks} consecutive checks. Exiting watcher."
                    )
                    break
                else:
                    print(
                        f"🔍 No running evaluations found ({consecutive_empty_checks}/{max_empty_checks} consecutive checks)"
                    )

            time.sleep(check_interval)

    except KeyboardInterrupt:
        print("\n🛑 Evaluation watcher interrupted by user")
    except Exception as e:
        print(f"\n❌ Evaluation watcher error: {e}")
    finally:
        print("🔍 Evaluation watcher stopped")


def _start_watcher_process(check_interval: float) -> multiprocessing.Process:
    """Start the watcher in a background process."""
    # Ensure we're not in a frozen state and multiprocessing is properly initialized
    if multiprocessing.current_process().name != "MainProcess":
        raise RuntimeError("Cannot start watcher process from within another process")

    process = multiprocessing.Process(target=_watcher_process_main, args=(check_interval,), name="eval-watcher")
    process.start()
    return process


def _watcher_process_main(check_interval: float) -> None:
    """Main entry point for the watcher process - acquires lock and runs the loop."""
    # Try to acquire the lock in this process
    current_holder_pid = acquire_singleton_lock()

    if current_holder_pid is not None:
        # Another process is already running
        print(f"🔍 Evaluation watcher already running in process {current_holder_pid}")
        return

    # We acquired the lock, run the watcher loop
    try:
        run_watcher_loop(check_interval)
    finally:
        # Always release the lock when we exit
        release_singleton_lock()


def ensure_singleton_watcher(check_interval: float = 5.0) -> bool:
    """
    Ensure the singleton EvaluationWatcher instance exists and is running.
    This function is OS-level global - only one watcher will run across all processes.

    Args:
        check_interval: How often to check for terminated processes (seconds)

    Returns:
        True if watcher was started successfully, False if another watcher is already running
    """
    # Check if we're already in a subprocess
    if multiprocessing.current_process().name != "MainProcess":
        return False

    # Start the watcher in a background process
    try:
        process = _start_watcher_process(check_interval)
        print(f"🔍 Started evaluation watcher in background process (PID: {process.pid})")
        return True
    except Exception as e:
        print(f"❌ Failed to start evaluation watcher: {e}")
        return False


def is_watcher_running() -> bool:
    """Check if the evaluation watcher is currently running."""
    current_holder_pid = acquire_singleton_lock()
    if current_holder_pid is None:
        # We acquired the lock, so no one else is running
        release_singleton_lock()
        return False

    # Check if the holder is still alive
    assert current_holder_pid is not None  # For type checker
    is_alive = is_process_running(current_holder_pid)
    if not is_alive:
        # Clean up stale lock
        release_singleton_lock()

    return is_alive


def get_watcher_pid() -> Optional[int]:
    """Get the PID of the currently running evaluation watcher."""
    _, pid_file_path = get_lock_file_paths()
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


def main():
    """Main entry point for the evaluation watcher."""
    parser = argparse.ArgumentParser(
        description="Monitor all evaluation rows and update those with terminated processes to stopped status"
    )
    parser.add_argument(
        "--check-interval",
        type=float,
        default=1.0,
        help="How often to check for terminated processes (seconds, default: 1.0)",
    )

    args = parser.parse_args()

    # Run the watcher directly (not as a background process)
    run_watcher_loop(args.check_interval)


if __name__ == "__main__":
    # Ensure multiprocessing is properly initialized
    multiprocessing.freeze_support()
    main()
