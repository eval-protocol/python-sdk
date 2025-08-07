#!/usr/bin/env python3
"""
Tests for the evaluation watcher functionality.

This module tests:
1. Singleton behavior - ensuring only one watcher can run at a time
2. Process termination detection - ensuring evaluations are updated to stopped when processes die
"""

import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest

from eval_protocol.dataset_logger import default_logger
from eval_protocol.logging_utils import get_logger
from eval_protocol.models import EvalMetadata, EvaluationRow, Message
from eval_protocol.pytest.eval_watcher import (
    ensure_singleton_watcher,
    get_watcher_pid,
    is_watcher_running,
    stop_watcher,
)
from eval_protocol.singleton_lock import (
    acquire_singleton_lock,
    is_process_running,
    release_singleton_lock,
)

# Initialize logger
logger = get_logger("test_eval_watcher")


class TestEvalWatcherSingleton:
    """Test that the evaluation watcher behaves as a singleton."""

    def test_singleton_behavior(self):
        """Test that only one watcher can run at a time."""
        # stop any existing watcher
        stop_watcher()

        # Check if watcher is already running (likely due to evaluation_test.py import)
        initial_running = is_watcher_running()

        assert not initial_running, "No watcher should be running"

        # Try to start a new watcher
        result = ensure_singleton_watcher(check_interval=1.0)

        assert isinstance(result, int), "Should start watcher when none is running"
        assert is_watcher_running(), "Watcher should be running"
        current_pid = get_watcher_pid()
        assert current_pid is not None, "Should get PID of running watcher"

    def test_singleton_lock_cleanup(self):
        """Test that singleton lock is properly cleaned up when watcher stops."""

        ensure_singleton_watcher(check_interval=1.0)

        assert is_watcher_running(), "Watcher should be running"

        # Get current PID
        original_pid = get_watcher_pid()
        assert original_pid is not None

        # Stop the watcher using SIGKILL (since SIGTERM is ignored)
        try:
            os.kill(original_pid, signal.SIGABRT)
            logger.info(f"🔍 Sent SIGKILL to evaluation watcher process {original_pid}")
        except OSError as e:
            logger.error(f"❌ Failed to stop evaluation watcher process {original_pid}: {e}")
            pytest.skip("Could not kill watcher process")

        # Wait longer for cleanup - the watcher process needs time to exit
        max_wait = 10.0
        wait_interval = 0.5
        waited = 0.0

        while waited < max_wait:
            if not is_watcher_running():
                break
            time.sleep(wait_interval)
            waited += wait_interval

        # Verify lock is released
        assert not is_watcher_running(), "Watcher should no longer be running"
        pid = get_watcher_pid()
        assert pid is None, "Should not have a PID after stopping"

    def test_multiple_start_stop_cycles(self):
        """Test multiple start/stop cycles work correctly."""
        for i in range(2):  # Reduced cycles to avoid interfering with other tests
            ensure_singleton_watcher(check_interval=1.0)
            # Get current PID
            current_pid = get_watcher_pid()
            assert current_pid is not None

            # Stop watcher using SIGKILL (since SIGTERM is ignored)
            try:
                os.kill(current_pid, signal.SIGKILL)
                logger.info(f"🔍 Sent SIGKILL to evaluation watcher process {current_pid}")
            except OSError as e:
                logger.error(f"❌ Failed to stop evaluation watcher process {current_pid}: {e}")
                pytest.skip("Could not kill watcher process")

            # Wait longer for cleanup - SIGKILL should be immediate but give some time
            max_wait = 15.0
            wait_interval = 0.5
            waited = 0.0

            while waited < max_wait:
                if not is_watcher_running():
                    break
                time.sleep(wait_interval)
                waited += wait_interval

            assert not is_watcher_running(), f"Watcher should not be running on cycle {i}"

    def test_watcher_pid_consistency(self):
        """Test that watcher PID is consistent and valid."""
        # Ensure watcher is running
        ensure_singleton_watcher(check_interval=1.0)

        # Get PID multiple times
        pid1 = get_watcher_pid()
        pid2 = get_watcher_pid()

        assert pid1 is not None
        assert pid2 is not None
        assert pid1 == pid2, "PID should be consistent"
        assert is_process_running(pid1), "PID should correspond to a running process"


class TestEvalWatcherProcessTermination:
    """Test that the evaluation watcher detects terminated processes and updates evaluations."""

    def setup_method(self):
        """Set up test environment."""
        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        self.original_datasets_dir = os.environ.get("EVAL_PROTOCOL_DATASETS_DIR")
        os.environ["EVAL_PROTOCOL_DATASETS_DIR"] = self.temp_dir

    def teardown_method(self):
        """Clean up after each test."""
        # Restore original environment
        if self.original_datasets_dir:
            os.environ["EVAL_PROTOCOL_DATASETS_DIR"] = self.original_datasets_dir
        else:
            os.environ.pop("EVAL_PROTOCOL_DATASETS_DIR", None)

        # Clean up temporary directory
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_running_evaluation_row(self, pid: int) -> EvaluationRow:
        """Create an evaluation row with 'running' status and specified PID."""
        from eval_protocol.models import InputMetadata

        row = EvaluationRow(
            messages=[Message(role="user", content="Test message")],
            input_metadata=InputMetadata(row_id=f"test_row_{pid}"),
            eval_metadata=EvalMetadata(
                name="test_evaluation", status="running", num_runs=1, aggregation_method="mean"
            ),
            pid=pid,
        )

        # Log the row
        default_logger.log(row)
        return row

    def test_detects_terminated_process(self):
        """Test that watcher detects when a process terminates and updates evaluation."""
        # Ensure watcher is running
        ensure_singleton_watcher(check_interval=0.5)

        # Give the watcher time to fully start
        time.sleep(1.0)

        # Create a short-lived process and get its PID
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(0.1)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        pid = process.pid

        # Create evaluation row with running status for this PID
        row = self.create_running_evaluation_row(pid)

        # Wait for process to terminate
        process.wait()

        # Wait for watcher to detect the terminated process
        max_wait = 15.0  # Increased wait time
        wait_interval = 0.5
        waited = 0.0

        while waited < max_wait:
            # Read the evaluation row
            rows = default_logger.read()
            test_row = None
            for r in rows:
                if r.input_metadata.row_id == row.input_metadata.row_id:
                    test_row = r
                    break

            if test_row and test_row.eval_metadata and test_row.eval_metadata.status == "stopped":
                break

            time.sleep(wait_interval)
            waited += wait_interval

        # Verify the evaluation was updated
        assert test_row is not None, "Should find the test row"
        assert test_row.eval_metadata is not None, "Should have eval metadata"
        assert test_row.eval_metadata.status == "stopped", "Status should be updated to stopped"
        assert test_row.eval_metadata.passed is False, "Should be marked as not passed"

        # Verify error information is set
        assert test_row.evaluation_result is not None, "Should have evaluation result"
        assert test_row.evaluation_result.error is not None, "Should have error message"
        assert "terminated" in test_row.evaluation_result.error.lower(), "Error should mention termination"

    def test_detects_multiple_terminated_processes(self):
        """Test that watcher detects multiple terminated processes."""
        # Ensure watcher is running
        ensure_singleton_watcher(check_interval=0.5)

        # Create multiple short-lived processes
        processes = []
        pids = []
        rows = []

        for i in range(3):
            process = subprocess.Popen(
                [sys.executable, "-c", f"import time; time.sleep({0.1 + i * 0.1})"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            pid = process.pid
            pids.append(pid)
            processes.append(process)

            # Create evaluation row for this PID
            row = self.create_running_evaluation_row(pid)
            rows.append(row)

        # Wait for all processes to terminate
        for process in processes:
            process.wait()

        # Wait for watcher to detect all terminated processes
        max_wait = 15.0  # 15 seconds max wait
        wait_interval = 0.5
        waited = 0.0

        while waited < max_wait:
            # Read all evaluation rows
            all_rows = default_logger.read()
            stopped_count = 0

            for row in rows:
                for r in all_rows:
                    if r.input_metadata.row_id == row.input_metadata.row_id:
                        if r.eval_metadata and r.eval_metadata.status == "stopped":
                            stopped_count += 1
                        break

            if stopped_count == len(rows):
                break

            time.sleep(wait_interval)
            waited += wait_interval

        # Verify all evaluations were updated
        assert stopped_count == len(rows), f"Expected {len(rows)} stopped evaluations, got {stopped_count}"

        # Verify each row was properly updated
        all_rows = default_logger.read()
        for original_row in rows:
            for r in all_rows:
                if r.input_metadata.row_id == original_row.input_metadata.row_id:
                    assert r.eval_metadata is not None
                    assert r.eval_metadata.status == "stopped"
                    assert r.eval_metadata.passed is False
                    assert r.evaluation_result is not None
                    assert r.evaluation_result.error is not None
                    break

    def test_ignores_running_processes(self):
        """Test that watcher doesn't update evaluations for running processes."""
        # Ensure watcher is running
        if not is_watcher_running():
            ensure_singleton_watcher(check_interval=0.5)

        # Create a long-running process
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        pid = process.pid

        # Create evaluation row with running status for this PID
        row = self.create_running_evaluation_row(pid)

        # Wait a bit for watcher to run
        time.sleep(2.0)

        # Verify the evaluation is still running
        rows = default_logger.read()
        test_row = None
        for r in rows:
            if r.input_metadata.row_id == row.input_metadata.row_id:
                test_row = r
                break

        assert test_row is not None, "Should find the test row"
        assert test_row.eval_metadata is not None, "Should have eval metadata"
        assert test_row.eval_metadata.status == "running", "Status should still be running"

        # Clean up
        process.terminate()
        process.wait()

    def test_handles_none_pid(self):
        """Test that watcher handles evaluation rows with None PID."""
        # Ensure watcher is running
        ensure_singleton_watcher(check_interval=0.5)

        # Create evaluation row with None PID
        row = self.create_running_evaluation_row(None)

        # Wait for watcher to process the row
        max_wait = 5.0
        wait_interval = 0.5
        waited = 0.0

        while waited < max_wait:
            rows = default_logger.read()
            test_row = None
            for r in rows:
                if r.input_metadata.row_id == row.input_metadata.row_id:
                    test_row = r
                    break

            if test_row and test_row.eval_metadata and test_row.eval_metadata.status == "stopped":
                break

            time.sleep(wait_interval)
            waited += wait_interval

        # Verify the evaluation was updated
        assert test_row is not None, "Should find the test row"
        assert test_row.eval_metadata is not None, "Should have eval metadata"
        assert test_row.eval_metadata.status == "stopped", "Status should be updated to stopped"


class TestEvalWatcherIntegration:
    """Integration tests for the evaluation watcher."""

    def setup_method(self):
        """Set up test environment."""
        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        self.original_datasets_dir = os.environ.get("EVAL_PROTOCOL_DATASETS_DIR")
        os.environ["EVAL_PROTOCOL_DATASETS_DIR"] = self.temp_dir

    def teardown_method(self):
        """Clean up after each test."""
        # Restore original environment
        if self.original_datasets_dir:
            os.environ["EVAL_PROTOCOL_DATASETS_DIR"] = self.original_datasets_dir
        else:
            os.environ.pop("EVAL_PROTOCOL_DATASETS_DIR", None)

        # Clean up temporary directory
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_watcher_survives_parent_process_termination(self):
        """Test that watcher survives when parent process is killed (simulated)."""
        # Ensure watcher is running
        ensure_singleton_watcher(check_interval=1.0)

        original_pid = get_watcher_pid()
        assert original_pid is not None

        # Simulate parent process termination by directly killing the watcher process
        # and then checking if it restarts (in a real scenario, the watcher would be
        # in a separate session and survive parent termination)
        os.kill(original_pid, signal.SIGKILL)

        # Wait for the process to be killed
        max_wait = 10.0
        wait_interval = 0.5
        waited = 0.0

        while waited < max_wait:
            if not is_watcher_running():
                break
            time.sleep(wait_interval)
            waited += wait_interval

        # The watcher should no longer be running
        assert not is_watcher_running()

        # We can start a new watcher
        result = ensure_singleton_watcher(check_interval=1.0)
        assert isinstance(result, int)
        new_pid = get_watcher_pid()
        assert new_pid is not None
        assert new_pid != original_pid

    def test_watcher_handles_signal_gracefully(self):
        """Test that watcher handles termination signals gracefully."""
        # Ensure watcher is running
        if not is_watcher_running():
            ensure_singleton_watcher(check_interval=1.0)

        pid = get_watcher_pid()
        assert pid is not None

        # Send SIGKILL to the watcher (SIGTERM is ignored)
        os.kill(pid, signal.SIGKILL)

        # Wait for the process to be killed
        max_wait = 10.0
        wait_interval = 0.5
        waited = 0.0

        while waited < max_wait:
            if not is_watcher_running():
                break
            time.sleep(wait_interval)
            waited += wait_interval

        # Verify watcher has stopped
        assert not is_watcher_running()
        assert get_watcher_pid() is None

    def test_concurrent_watcher_startup(self):
        """Test that concurrent attempts to start watchers are handled correctly."""
        import queue
        import threading

        # stop any existing watcher
        stop_watcher()

        results = queue.Queue()

        def start_watcher():
            try:
                result = ensure_singleton_watcher(check_interval=1.0)
                results.put(result)
            except Exception as e:
                results.put(e)

        # Start multiple threads trying to start watchers simultaneously
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=start_watcher)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check results
        success_count = 0
        while not results.empty():
            result = results.get()
            if is_process_running(result):
                success_count += 1

        # Only one should succeed (or none if already running)
        assert success_count == 1, f"Expected 1 successful start, got {success_count}"
        assert is_watcher_running(), "Watcher should be running"
