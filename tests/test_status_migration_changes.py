#!/usr/bin/env python3
"""
Tests for the migration changes in the existing codebase.

This test suite verifies that:
- All migrated code works correctly with the new Status model
- The field name remains as 'rollout_status'
- All helper methods work as expected
- AIP-193 compliance is maintained
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from eval_protocol.models import Status, EvaluationRow, Message
from eval_protocol.types import TerminationReason


class TestMCPExecutionManagerMigration:
    """Test the migration changes in MCP execution manager."""

    def test_trajectory_terminated_status_creation(self):
        """Test that terminated trajectory creates correct status."""
        # Mock trajectory with termination
        trajectory = Mock()
        trajectory.terminated = True
        trajectory.termination_reason = TerminationReason.CONTROL_PLANE_SIGNAL
        trajectory.control_plane_summary = {"error_message": "No errors"}

        # Create evaluation row
        row = EvaluationRow(messages=[])

        # Simulate the status assignment from MCP execution manager
        extra_info = {"error_message": trajectory.control_plane_summary.get("error_message")}

        row.rollout_status = Status(
            code=Status.Code.FINISHED,
            message="Rollout finished",
            details=[
                {
                    "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                    "reason": "TERMINATION_REASON",
                    "domain": "evalprotocol.io",
                    "metadata": {"termination_reason": trajectory.termination_reason},
                }
            ]
            + (
                [
                    {
                        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                        "reason": "EXTRA_INFO",
                        "domain": "evalprotocol.io",
                        "metadata": extra_info,
                    }
                ]
                if extra_info
                else []
            ),
        )

        # Verify the status
        assert row.rollout_status.code == Status.Code.FINISHED
        assert row.rollout_status.message == "Rollout finished"
        assert row.rollout_status.is_finished()

        # Verify termination reason
        assert row.rollout_status.get_termination_reason() == TerminationReason.CONTROL_PLANE_SIGNAL

        # Verify extra info
        assert row.rollout_status.get_extra_info() == {"error_message": "No errors"}

        # Verify details structure
        assert len(row.rollout_status.details) == 2
        assert row.rollout_status.details[0]["reason"] == "TERMINATION_REASON"
        assert row.rollout_status.details[1]["reason"] == "EXTRA_INFO"

    def test_trajectory_running_status_creation(self):
        """Test that running trajectory creates correct status."""
        # Mock trajectory that's still running
        trajectory = Mock()
        trajectory.terminated = False

        # Create evaluation row
        row = EvaluationRow(messages=[])

        # Simulate the status assignment from MCP execution manager
        row.rollout_status = Status(code=Status.Code.RUNNING, message="Rollout is running", details=[])

        # Verify the status
        assert row.rollout_status.code == Status.Code.RUNNING
        assert row.rollout_status.message == "Rollout is running"
        assert row.rollout_status.is_running()
        assert not row.rollout_status.is_finished()
        assert not row.rollout_status.is_error()
        assert not row.rollout_status.is_stopped()

    def test_trajectory_terminated_without_error_message(self):
        """Test terminated trajectory without error message."""
        # Mock trajectory with termination but no error
        trajectory = Mock()
        trajectory.terminated = True
        trajectory.termination_reason = TerminationReason.USER_STOP
        trajectory.control_plane_summary = {}

        # Create evaluation row
        row = EvaluationRow(messages=[])

        # Simulate the status assignment
        extra_info = None
        if trajectory.control_plane_summary.get("error_message"):
            extra_info = {"error_message": trajectory.control_plane_summary.get("error_message")}

        row.rollout_status = Status(
            code=Status.Code.FINISHED,
            message="Rollout finished",
            details=[
                {
                    "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                    "reason": "TERMINATION_REASON",
                    "domain": "evalprotocol.io",
                    "metadata": {"termination_reason": trajectory.termination_reason},
                }
            ]
            + (
                [
                    {
                        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                        "reason": "EXTRA_INFO",
                        "domain": "evalprotocol.io",
                        "metadata": extra_info,
                    }
                ]
                if extra_info
                else []
            ),
        )

        # Verify the status
        assert row.rollout_status.code == Status.Code.FINISHED
        assert row.rollout_status.is_finished()
        assert row.rollout_status.get_termination_reason() == TerminationReason.USER_STOP

        # Should not have extra info since there was no error message
        assert row.rollout_status.get_extra_info() is None

        # Should only have termination reason detail
        assert len(row.rollout_status.details) == 1
        assert row.rollout_status.details[0]["reason"] == "TERMINATION_REASON"


class TestPytestUtilsMigration:
    """Test the migration changes in pytest utils."""

    def test_retry_success_status_update(self):
        """Test that retry success updates status correctly."""
        row = EvaluationRow(messages=[])

        # Simulate the status update from pytest utils
        row.rollout_status = Status(code=Status.Code.FINISHED, message="Rollout finished successfully", details=[])

        # Verify the status
        assert row.rollout_status.code == Status.Code.FINISHED
        assert row.rollout_status.message == "Rollout finished successfully"
        assert row.rollout_status.is_finished()
        assert not row.rollout_status.is_running()

    def test_retry_failure_status_update(self):
        """Test that retry failure updates status correctly."""
        row = EvaluationRow(messages=[])

        # Simulate the status update from pytest utils
        row.rollout_status = Status(
            code=Status.Code.INTERNAL,
            message="Test error message",
            details=[
                {
                    "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                    "reason": "ROLLOUT_ERROR",
                    "domain": "evalprotocol.io",
                    "metadata": {},
                }
            ],
        )

        # Verify the status
        assert row.rollout_status.code == Status.Code.INTERNAL
        assert row.rollout_status.message == "Test error message"
        assert row.rollout_status.is_error()
        assert not row.rollout_status.is_finished()

    def test_initial_processor_success_status_update(self):
        """Test that initial processor success updates status correctly."""
        row = EvaluationRow(messages=[])

        # Simulate the status update from pytest utils
        row.rollout_status = Status(code=Status.Code.FINISHED, message="Rollout finished successfully", details=[])

        # Verify the status
        assert row.rollout_status.code == Status.Code.FINISHED
        assert row.rollout_status.message == "Rollout finished successfully"
        assert row.rollout_status.is_finished()

    def test_initial_processor_failure_status_update(self):
        """Test that initial processor failure updates status correctly."""
        row = EvaluationRow(messages=[])

        # Simulate the status update from pytest utils
        row.rollout_status = Status(
            code=Status.Code.INTERNAL,
            message="Runtime error occurred",
            details=[
                {
                    "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                    "reason": "ROLLOUT_ERROR",
                    "domain": "evalprotocol.io",
                    "metadata": {},
                }
            ],
        )

        # Verify the status
        assert row.rollout_status.code == Status.Code.INTERNAL
        assert row.rollout_status.message == "Runtime error occurred"
        assert row.rollout_status.is_error()

    def test_error_status_checking(self):
        """Test that error status checking works correctly."""
        row = EvaluationRow(messages=[])

        # Set error status
        row.rollout_status = Status.rollout_error("Test error")

        # Should be detected as error
        assert row.rollout_status.is_error()

        # Should be able to get termination reason (None for error status)
        assert row.rollout_status.get_termination_reason() is None


class TestTestRetryMechanismMigration:
    """Test the migration changes in test_retry_mechanism.py."""

    def test_success_failure_detection(self):
        """Test that success/failure detection works correctly."""
        row = EvaluationRow(messages=[])

        # Test success case
        row.rollout_status = Status.rollout_finished()
        success = row.rollout_status.is_finished()
        assert success is True

        # Test failure case
        row.rollout_status = Status.rollout_error("Test error")
        success = row.rollout_status.is_finished()
        assert success is False

    def test_score_assignment_based_on_status(self):
        """Test that score assignment works based on status."""
        row = EvaluationRow(messages=[])

        # Test success score
        row.rollout_status = Status.rollout_finished()
        score = 1.0 if row.rollout_status.is_finished() else 0.0
        assert score == 1.0

        # Test failure score
        row.rollout_status = Status.rollout_error("Test error")
        score = 1.0 if row.rollout_status.is_finished() else 0.0
        assert score == 0.0


class TestStatusModelIntegration:
    """Test integration of Status model with existing functionality."""

    def test_status_creation_methods_integration(self):
        """Test that all status creation methods work together."""
        row = EvaluationRow(messages=[])

        # Test running status
        row.rollout_status = Status.rollout_running()
        assert row.rollout_status.is_running()
        assert row.rollout_status.code == Status.Code.RUNNING

        # Test finished status
        row.rollout_status = Status.rollout_finished()
        assert row.rollout_status.is_finished()
        assert row.rollout_status.code == Status.Code.FINISHED

        # Test error status
        row.rollout_status = Status.rollout_error("Test error")
        assert row.rollout_status.is_error()
        assert row.rollout_status.code == Status.Code.INTERNAL

    def test_termination_reason_integration(self):
        """Test integration of termination reason with status."""
        row = EvaluationRow(messages=[])

        # Test with termination reason
        termination_status = Status.rollout_finished(TerminationReason.CONTROL_PLANE_SIGNAL)
        row.rollout_status = termination_status

        assert row.rollout_status.is_finished()
        assert row.rollout_status.get_termination_reason() == TerminationReason.CONTROL_PLANE_SIGNAL

        # Test with termination reason and extra info
        extra_info = {"steps": 10, "reward": 0.8}
        termination_status_with_info = Status.rollout_finished(TerminationReason.USER_STOP, extra_info)
        row.rollout_status = termination_status_with_info

        assert row.rollout_status.is_finished()
        assert row.rollout_status.get_termination_reason() == TerminationReason.USER_STOP
        assert row.rollout_status.get_extra_info() == extra_info

    def test_error_handling_integration(self):
        """Test integration of error handling with status."""
        row = EvaluationRow(messages=[])

        # Test error with metadata
        error_info = {"error_code": "E001", "line": 42}
        error_status = Status.rollout_error("Runtime error", error_info)
        row.rollout_status = error_status

        assert row.rollout_status.is_error()
        assert row.rollout_status.get_extra_info() == error_info
        assert row.rollout_status.get_termination_reason() is None

        # Test error without metadata
        simple_error_status = Status.rollout_error("Simple error")
        row.rollout_status = simple_error_status

        assert row.rollout_status.is_error()
        assert row.rollout_status.get_extra_info() is None

    def test_status_transitions_integration(self):
        """Test that status transitions work correctly in integration."""
        row = EvaluationRow(messages=[])

        # Start with running
        row.rollout_status = Status.rollout_running()
        assert row.rollout_status.is_running()

        # Transition to finished
        row.rollout_status = Status.rollout_finished()
        assert row.rollout_status.is_finished()
        assert not row.rollout_status.is_running()

        # Transition to error
        row.rollout_status = Status.rollout_error("Something went wrong")
        assert row.rollout_status.is_error()
        assert not row.rollout_status.is_finished()

        # Transition back to finished
        row.rollout_status = Status.rollout_finished()
        assert row.rollout_status.is_finished()
        assert not row.rollout_status.is_error()


class TestAIP193Compliance:
    """Test AIP-193 compliance in the migrated code."""

    def test_error_info_structure_compliance(self):
        """Test that ErrorInfo structure follows AIP-193."""
        row = EvaluationRow(messages=[])

        # Create error status with metadata
        error_info = {"error_code": "E001", "timestamp": "2024-01-01"}
        error_status = Status.rollout_error("Test error", error_info)
        row.rollout_status = error_status

        # Check AIP-193 ErrorInfo structure
        assert len(row.rollout_status.details) == 1
        detail = row.rollout_status.details[0]

        # Required fields according to AIP-193
        assert detail["@type"] == "type.googleapis.com/google.rpc.ErrorInfo"
        assert "reason" in detail
        assert "domain" in detail
        assert "metadata" in detail

        # Domain should be service-specific
        assert detail["domain"] == "evalprotocol.io"

        # Metadata should contain the error info
        assert detail["metadata"] == error_info

    def test_termination_reason_structure_compliance(self):
        """Test that termination reason structure follows AIP-193."""
        row = EvaluationRow(messages=[])

        # Create status with termination reason
        termination_status = Status.rollout_finished(TerminationReason.CONTROL_PLANE_SIGNAL)
        row.rollout_status = termination_status

        # Check AIP-193 structure
        assert len(row.rollout_status.details) == 1
        detail = row.rollout_status.details[0]

        assert detail["@type"] == "type.googleapis.com/google.rpc.ErrorInfo"
        assert detail["reason"] == "TERMINATION_REASON"
        assert detail["domain"] == "evalprotocol.io"
        assert "metadata" in detail
        assert "termination_reason" in detail["metadata"]

    def test_multiple_details_compliance(self):
        """Test that multiple details follow AIP-193 structure."""
        row = EvaluationRow(messages=[])

        # Create status with both termination reason and extra info
        extra_info = {"steps": 15, "reward": 0.9}
        status = Status.rollout_finished(TerminationReason.CONTROL_PLANE_SIGNAL, extra_info)
        row.rollout_status = status

        # Should have two details
        assert len(row.rollout_status.details) == 2

        # Both should follow ErrorInfo structure
        for detail in row.rollout_status.details:
            assert detail["@type"] == "type.googleapis.com/google.rpc.ErrorInfo"
            assert "reason" in detail
            assert "domain" in detail
            assert "metadata" in detail
            assert detail["domain"] == "evalprotocol.io"

    def test_status_code_compliance(self):
        """Test that status codes follow gRPC standard."""
        row = EvaluationRow(messages=[])

        # Test standard gRPC codes
        statuses = [
            (Status.rollout_running(), Status.Code.RUNNING),
            (Status.rollout_finished(), Status.Code.FINISHED),  # Custom code
            (Status.rollout_error("Test"), Status.Code.INTERNAL),
        ]

        for status, expected_code in statuses:
            row.rollout_status = status
            assert row.rollout_status.code == expected_code


if __name__ == "__main__":
    pytest.main([__file__])
