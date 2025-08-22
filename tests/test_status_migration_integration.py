#!/usr/bin/env python3
"""
Integration tests for the Status model migration.

This test suite verifies that:
- All migrated code works correctly with the new Status model
- The field name remains as 'rollout_status'
- All helper methods work as expected
- AIP-193 compliance is maintained
"""

import pytest
from unittest.mock import Mock, patch
from eval_protocol.models import Status, EvaluationRow, Message
from eval_protocol.types import TerminationReason


class TestStatusFieldNamePreservation:
    """Test that the field name remains as 'rollout_status'."""

    def test_evaluation_row_has_rollout_status_field(self):
        """Test that EvaluationRow still has rollout_status field."""
        row = EvaluationRow(messages=[])

        # Should have rollout_status field
        assert hasattr(row, "rollout_status")
        assert not hasattr(row, "status")

        # Field should be of type Status
        assert isinstance(row.rollout_status, Status)

    def test_rollout_status_field_access(self):
        """Test direct access to rollout_status field."""
        row = EvaluationRow(messages=[])

        # Should be able to access directly
        assert row.rollout_status.code == Status.Code.RUNNING
        assert row.rollout_status.message == "Rollout is running"

        # Should be able to set directly
        row.rollout_status = Status.rollout_finished()
        assert row.rollout_status.code == Status.Code.FINISHED


class TestStatusTransitions:
    """Test transitioning between different status states."""

    def test_running_to_finished_transition(self):
        """Test transition from running to finished."""
        row = EvaluationRow(messages=[])

        # Start with running
        assert row.rollout_status.is_running()
        assert not row.rollout_status.is_finished()

        # Transition to finished
        row.rollout_status = Status.rollout_finished()
        assert not row.rollout_status.is_running()
        assert row.rollout_status.is_finished()

    def test_running_to_error_transition(self):
        """Test transition from running to error."""
        row = EvaluationRow(messages=[])

        # Start with running
        assert row.rollout_status.is_running()
        assert not row.rollout_status.is_error()

        # Transition to error
        row.rollout_status = Status.rollout_error("Something went wrong")
        assert not row.rollout_status.is_running()
        assert row.rollout_status.is_error()

    def test_error_to_finished_transition(self):
        """Test transition from error to finished."""
        row = EvaluationRow(messages=[])

        # Start with error
        row.rollout_status = Status.rollout_error("Initial error")
        assert row.rollout_status.is_error()

        # Transition to finished
        row.rollout_status = Status.rollout_finished()
        assert not row.rollout_status.is_error()
        assert row.rollout_status.is_finished()


class TestTerminationReasonIntegration:
    """Test integration of termination reason with the new Status model."""

    def test_termination_reason_in_status_details(self):
        """Test that termination reason is properly stored in status details."""
        row = EvaluationRow(messages=[])

        # Set status with termination reason
        termination_status = Status.rollout_finished(TerminationReason.CONTROL_PLANE_SIGNAL)
        row.rollout_status = termination_status

        # Should be finished
        assert row.rollout_status.is_finished()

        # Should have termination reason in details
        assert row.rollout_status.get_termination_reason() == TerminationReason.CONTROL_PLANE_SIGNAL

        # Check details structure
        assert len(row.rollout_status.details) == 1
        detail = row.rollout_status.details[0]
        assert detail["@type"] == "type.googleapis.com/google.rpc.ErrorInfo"
        assert detail["reason"] == "TERMINATION_REASON"
        assert detail["domain"] == "evalprotocol.io"
        assert detail["metadata"]["termination_reason"] == TerminationReason.CONTROL_PLANE_SIGNAL

    def test_termination_reason_with_extra_info(self):
        """Test termination reason with additional extra info."""
        row = EvaluationRow(messages=[])

        extra_info = {"steps": 10, "reward": 0.8}
        termination_status = Status.rollout_finished(TerminationReason.USER_STOP, extra_info)
        row.rollout_status = termination_status

        # Should have both termination reason and extra info
        assert row.rollout_status.get_termination_reason() == TerminationReason.USER_STOP
        assert row.rollout_status.get_extra_info() == extra_info

        # Check details structure
        assert len(row.rollout_status.details) == 2

        # First detail should be termination reason
        term_detail = row.rollout_status.details[0]
        assert term_detail["reason"] == "TERMINATION_REASON"

        # Second detail should be extra info
        extra_detail = row.rollout_status.details[1]
        assert extra_detail["reason"] == "EXTRA_INFO"
        assert extra_detail["metadata"] == extra_info

    def test_multiple_termination_reasons(self):
        """Test handling of multiple termination reasons (edge case)."""
        row = EvaluationRow(messages=[])

        # Create status with duplicate termination reason details
        details = [
            {
                "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                "reason": "TERMINATION_REASON",
                "domain": "evalprotocol.io",
                "metadata": {"termination_reason": TerminationReason.USER_STOP},
            },
            {
                "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                "reason": "TERMINATION_REASON",
                "domain": "evalprotocol.io",
                "metadata": {"termination_reason": TerminationReason.SKIPPABLE_ERROR},
            },
        ]

        status = Status(code=Status.Code.FINISHED, message="Test", details=details)
        row.rollout_status = status

        # Should return the first termination reason found
        assert row.rollout_status.get_termination_reason() == TerminationReason.USER_STOP


class TestErrorHandlingIntegration:
    """Test error handling integration with the new Status model."""

    def test_error_status_with_metadata(self):
        """Test error status with structured metadata."""
        row = EvaluationRow(messages=[])

        error_info = {
            "error_code": "E001",
            "line": 42,
            "function": "test_function",
            "timestamp": "2024-01-01T12:00:00Z",
        }

        error_status = Status.rollout_error("Runtime error occurred", error_info)
        row.rollout_status = error_status

        # Should be error
        assert row.rollout_status.is_error()

        # Should have error details
        assert row.rollout_status.get_extra_info() == error_info

        # Should not have termination reason
        assert row.rollout_status.get_termination_reason() is None

        # Check details structure
        assert len(row.rollout_status.details) == 1
        detail = row.rollout_status.details[0]
        assert detail["@type"] == "type.googleapis.com/google.rpc.ErrorInfo"
        assert detail["reason"] == "EXTRA_INFO"
        assert detail["domain"] == "evalprotocol.io"
        assert detail["metadata"] == error_info

    def test_error_status_without_metadata(self):
        """Test error status without additional metadata."""
        row = EvaluationRow(messages=[])

        error_status = Status.rollout_error("Simple error message")
        row.rollout_status = error_status

        # Should be error
        assert row.rollout_status.is_error()

        # Should not have extra info
        assert row.rollout_status.get_extra_info() is None

        # Should not have termination reason
        assert row.rollout_status.get_termination_reason() is None

        # Should have empty details
        assert row.rollout_status.details == []


class TestAIP193Compliance:
    """Test AIP-193 compliance features."""

    def test_error_info_structure(self):
        """Test that ErrorInfo follows AIP-193 structure."""
        row = EvaluationRow(messages=[])

        # Create status with error info
        error_info = {"error_code": "E001"}
        error_status = Status.rollout_error("Test error", error_info)
        row.rollout_status = error_status

        # Check AIP-193 ErrorInfo structure
        assert len(row.rollout_status.details) == 1
        detail = row.rollout_status.details[0]

        # Required fields
        assert detail["@type"] == "type.googleapis.com/google.rpc.ErrorInfo"
        assert "reason" in detail
        assert "domain" in detail
        assert "metadata" in detail

        # Domain should be service-specific
        assert detail["domain"] == "evalprotocol.io"

        # Metadata should contain the error info
        assert detail["metadata"] == error_info

    def test_multiple_detail_types(self):
        """Test that multiple detail types can coexist."""
        row = EvaluationRow(messages=[])

        # Create status with both termination reason and extra info
        extra_info = {"steps": 15, "reward": 0.9}
        status = Status.rollout_finished(TerminationReason.CONTROL_PLANE_SIGNAL, extra_info)
        row.rollout_status = status

        # Should have two details
        assert len(row.rollout_status.details) == 2

        # First detail should be termination reason
        term_detail = row.rollout_status.details[0]
        assert term_detail["reason"] == "TERMINATION_REASON"

        # Second detail should be extra info
        extra_detail = row.rollout_status.details[1]
        assert extra_detail["reason"] == "EXTRA_INFO"

        # Both should follow ErrorInfo structure
        for detail in row.rollout_status.details:
            assert detail["@type"] == "type.googleapis.com/google.rpc.ErrorInfo"
            assert "reason" in detail
            assert "domain" in detail
            assert "metadata" in detail

    def test_status_code_mapping(self):
        """Test that status codes map correctly to gRPC codes."""
        row = EvaluationRow(messages=[])

        # Test different status types and their codes
        statuses = [
            (Status.rollout_running(), Status.Code.RUNNING),
            (Status.rollout_finished(), Status.Code.FINISHED),
            (Status.rollout_error("Test"), Status.Code.INTERNAL),
        ]

        for status, expected_code in statuses:
            row.rollout_status = status
            assert row.rollout_status.code == expected_code


class TestSerializationAndDeserialization:
    """Test that Status can be properly serialized and deserialized."""

    def test_status_model_dump(self):
        """Test that Status can be dumped to dict."""
        row = EvaluationRow(messages=[])

        # Set a complex status
        extra_info = {"steps": 10, "reward": 0.8}
        termination_status = Status.rollout_finished(TerminationReason.CONTROL_PLANE_SIGNAL, extra_info)
        row.rollout_status = termination_status

        # Dump to dict
        status_dict = row.rollout_status.model_dump()

        # Check structure
        assert "code" in status_dict
        assert "message" in status_dict
        assert "details" in status_dict

        # Check values
        assert status_dict["code"] == Status.Code.FINISHED
        assert status_dict["message"] == "Rollout finished"
        assert len(status_dict["details"]) == 2

    def test_status_model_validate(self):
        """Test that Status can be reconstructed from dict."""
        row = EvaluationRow(messages=[])

        # Set a complex status
        extra_info = {"steps": 10, "reward": 0.8}
        original_status = Status.rollout_finished(TerminationReason.CONTROL_PLANE_SIGNAL, extra_info)
        row.rollout_status = original_status

        # Dump and reconstruct
        status_dict = row.rollout_status.model_dump()
        reconstructed_status = Status.model_validate(status_dict)

        # Should be equivalent
        assert reconstructed_status.code == original_status.code
        assert reconstructed_status.message == original_status.message
        assert len(reconstructed_status.details) == len(original_status.details)

        # Should preserve functionality
        assert reconstructed_status.get_termination_reason() == TerminationReason.CONTROL_PLANE_SIGNAL
        assert reconstructed_status.get_extra_info() == extra_info


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_status_details(self):
        """Test Status with empty details."""
        row = EvaluationRow(messages=[])

        # Create status with empty details
        empty_status = Status(code=Status.Code.OK, message="Test", details=[])
        row.rollout_status = empty_status

        # Should handle gracefully
        assert row.rollout_status.get_termination_reason() is None
        assert row.rollout_status.get_extra_info() is None

    def test_malformed_status_details(self):
        """Test Status with malformed details."""
        row = EvaluationRow(messages=[])

        # Create status with malformed details
        malformed_details = [
            {"not_type": "invalid", "reason": "TEST"},
            {"@type": "type.googleapis.com/google.rpc.ErrorInfo", "metadata": {"termination_reason": "test"}},
        ]

        malformed_status = Status(code=Status.Code.OK, message="Test", details=malformed_details)
        row.rollout_status = malformed_status

        # Should handle gracefully
        assert row.rollout_status.get_termination_reason() is None
        assert row.rollout_status.get_extra_info() is None

    def test_large_metadata_handling(self):
        """Test Status with large metadata."""
        row = EvaluationRow(messages=[])

        # Create large metadata
        large_metadata = {f"key_{i}": f"value_{i}" for i in range(100)}

        # Should handle large metadata
        large_status = Status.rollout_error("Test error", large_metadata)
        row.rollout_status = large_status

        # Should preserve all metadata
        extra_info = row.rollout_status.get_extra_info()
        assert extra_info == large_metadata
        assert extra_info is not None
        assert len(extra_info) == 100


if __name__ == "__main__":
    pytest.main([__file__])
