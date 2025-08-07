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
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, List, Optional

from eval_protocol.dataset_logger import default_logger
from eval_protocol.dataset_logger.directory_utils import find_eval_protocol_dir
from eval_protocol.logging_utils import get_logger
from eval_protocol.models import EvaluationRow
from eval_protocol.singleton_lock import (
    acquire_singleton_lock,
    get_lock_file_paths,
    get_lock_holder_pid,
    is_lock_held,
    is_process_running,
    release_singleton_lock,
)

# Initialize logger
logger = get_logger("eval_watcher")

# Lock configuration
LOCK_NAME = "eval_watcher"


# Signal handler to automatically reap zombie processes
def _reap_zombies(signum, frame):
    """Reap zombie child processes to prevent them from accumulating."""
    try:
        while True:
            # Wait for any child process, but don't block
            pid, status = os.waitpid(-1, os.WNOHANG)
            if pid == 0:  # No more children to reap
                break
    except OSError:
        # No child processes
        pass


# Set up signal handler for SIGCHLD if available
if hasattr(signal, "SIGCHLD"):
    signal.signal(signal.SIGCHLD, _reap_zombies)


def get_eval_protocol_dir() -> Path:
    """Get the evaluation protocol directory for lock files."""
    return Path(find_eval_protocol_dir())


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
        logger.info(
            f"  📝 Updated evaluation '{row.eval_metadata.name if row.eval_metadata else 'Unknown'}' (Row ID: {row.input_metadata.row_id}) (PID: {row.pid}) to stopped status"
        )

    except Exception as e:
        logger.error(f"  ⚠️  Error updating evaluation row: {e}")


def check_and_update_terminated_evaluations() -> int:
    """Check for evaluations with terminated processes and update them to stopped status."""
    running_evaluations = find_running_evaluations()

    if not running_evaluations:
        return 0

    logger.info(f"🔍 Checking {len(running_evaluations)} running evaluations for terminated processes...")
    for row in running_evaluations:
        logger.info(f" Row ID: {row.input_metadata.row_id}  PID: {row.pid}")

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
        logger.info(f"  ✅ Updated {terminated_count} evaluations to stopped status")

    return terminated_count


def signal_handler(signum, frame):
    """Handle termination signals gracefully."""
    signal_name = signal.Signals(signum).name
    logger.info(f"\n🛑 Evaluation watcher received signal {signum} (Signal: {signal_name})")
    if signum == signal.SIGTERM:
        logger.info("SIGTERM received: ignoring to avoid exit during VSCode pytest debugging.")
        return
    logger.info("Shutting down gracefully.")
    sys.exit(0)


def run_watcher_loop(check_interval: float) -> None:
    """Main monitoring loop."""
    logger.info(f"🔍 Starting evaluation watcher (PID: {os.getpid()})")
    logger.info(f"  Check interval: {check_interval} seconds")
    logger.info("  Monitoring all evaluation rows for terminated processes")

    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    consecutive_empty_checks = 0
    max_empty_checks = 30

    try:
        while True:
            running_evaluations = find_running_evaluations()

            if running_evaluations:
                consecutive_empty_checks = 0
                check_and_update_terminated_evaluations()
            else:
                consecutive_empty_checks += 1
                if consecutive_empty_checks >= max_empty_checks:
                    logger.info(
                        f"🔍 No running evaluations found for {consecutive_empty_checks} consecutive checks. Exiting watcher."
                    )
                    break
                else:
                    logger.info(
                        f"🔍 No running evaluations found ({consecutive_empty_checks}/{max_empty_checks} consecutive checks)"
                    )

            time.sleep(check_interval)

    except KeyboardInterrupt:
        logger.info("\n🛑 Evaluation watcher interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Evaluation watcher error: {e}")
    finally:
        logger.info("🔍 Evaluation watcher stopped")


def _start_watcher_process(check_interval: float) -> Optional[int]:
    """Start the watcher in a completely independent background process using subprocess.

    We use subprocess.Popen with start_new_session=True instead of multiprocessing.Process
    because VSCode's test debugger kill button sends SIGTERM/SIGKILL to the entire process
    tree, including child processes. By using subprocess with a new session, we create
    a truly independent process that survives when the parent pytest process is killed.
    """

    # Use subprocess to create a completely independent process
    # This ensures the process survives even if the parent pytest process is killed
    try:
        # Get the current script path
        current_script = __file__

        # Create the subprocess with complete independence
        process = subprocess.Popen(
            [sys.executable, current_script, "--daemon", "--check-interval", str(check_interval)],
            # These flags make the process completely independent
            start_new_session=True,  # Creates a new session, detaching from parent
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            close_fds=True,
        )

        return process.pid

    except Exception as e:
        logger.error(f"❌ Failed to start watcher process: {e}")
        return None


def ensure_singleton_watcher(check_interval: float = 2.0) -> Optional[int]:
    """
    Ensure the singleton EvaluationWatcher instance exists and is running.
    This function is OS-level global - only one watcher will run across all processes.
    The watcher runs as a completely independent process that survives if the main process dies.

    Args:
        check_interval: How often to check for terminated processes (seconds)

    Returns:
        PID of the watcher process if it was started successfully, None if it failed to start
    """
    # Check if a watcher is already running before attempting to start a new one
    if is_watcher_running():
        logger.info("🔍 Evaluation watcher is already running")
        return None

    # Start the watcher in a completely independent background process
    pid = _start_watcher_process(check_interval)
    logger.info(f"🔍 Started evaluation watcher in independent background process (PID: {pid})")
    return pid


def is_watcher_running() -> bool:
    """Check if the evaluation watcher is currently running."""
    return is_lock_held(get_eval_protocol_dir(), LOCK_NAME)


def get_watcher_pid(timeout: float = 10.0) -> Optional[int]:
    """Get the PID of the currently running evaluation watcher. Tries for 10 seconds."""
    interval = 0.1
    started = time.time()
    while time.time() - started < timeout:
        pid = get_lock_holder_pid(get_eval_protocol_dir(), LOCK_NAME)
        if pid is not None:
            return pid
        time.sleep(interval)
    return None


def stop_watcher() -> bool:
    """Stop the currently running evaluation watcher."""
    pid = get_watcher_pid()
    if pid is None:
        logger.info("🔍 No evaluation watcher is currently running")
        return False

    try:
        os.kill(pid, signal.SIGKILL)
        logger.info(f"🔍 Sent SIGTERM to evaluation watcher process {pid}")
        return True
    except OSError as e:
        logger.error(f"❌ Failed to stop evaluation watcher process {pid}: {e}")
        return False


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
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run in daemon mode (internal use only)",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop the currently running evaluation watcher",
    )

    args = parser.parse_args()

    # Handle stop command
    if args.stop:
        stop_watcher()
        return

    # If running in daemon mode, try to acquire the lock and run the watcher loop
    if args.daemon:
        logger.info(f"🔍 Daemon mode: attempting to acquire lock (PID: {os.getpid()})")
        # Try to acquire the lock in this process
        current_holder_pid = acquire_singleton_lock(get_eval_protocol_dir(), LOCK_NAME)

        if current_holder_pid is not None:
            # Another process is already running
            logger.info(f"🔍 Evaluation watcher already running in process {current_holder_pid}")
            return

        logger.info(f"🔍 Daemon mode: acquired lock successfully (PID: {os.getpid()})")
        # We acquired the lock, run the watcher loop
        try:
            run_watcher_loop(args.check_interval)
        except SystemExit:
            # Graceful shutdown
            pass
        finally:
            # Always release the lock when we exit
            logger.info(f"🔍 Daemon mode: releasing lock (PID: {os.getpid()})")
            release_singleton_lock(get_eval_protocol_dir(), LOCK_NAME)
    else:
        # Run the watcher directly (not as a background process)
        run_watcher_loop(args.check_interval)


if __name__ == "__main__":
    main()
