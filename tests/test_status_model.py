#!/usr/bin/env python3
"""
Tests for the AIP-193 compatible Status model.

This test suite covers:
- Status code enum values
- Status creation methods
- Helper methods for checking status types
- AIP-193 compliance features
- Migration from RolloutStatus
"""

import pytest
from eval_protocol.models import Status, EvaluationRow, Message, ErrorInfo


class TestErrorInfoModel:
    """Test the ErrorInfo model."""

    def test_error_info_creation(self):
        """Test creating ErrorInfo instances."""
        error_info = ErrorInfo(
            reason="TEST_ERROR", domain="evalprotocol.io", metadata={"error_code": "E001", "line": 42}
        )

        assert error_info.reason == "TEST_ERROR"
        assert error_info.domain == "evalprotocol.io"
        assert error_info.metadata == {"error_code": "E001", "line": 42}

    def test_error_info_to_aip193_format(self):
        """Test conversion to AIP-193 format."""
        error_info = ErrorInfo(reason="TEST_ERROR", domain="evalprotocol.io", metadata={"error_code": "E001"})

        aip193_format = error_info.to_aip193_format()

        assert aip193_format["@type"] == "type.googleapis.com/google.rpc.ErrorInfo"
        assert aip193_format["reason"] == "TEST_ERROR"
        assert aip193_format["domain"] == "evalprotocol.io"
        assert aip193_format["metadata"] == {"error_code": "E001"}

    def test_error_info_factory_methods(self):
        """Test the factory methods for common error types."""
        # Test termination_reason
        term_error = ErrorInfo.termination_reason("goal_reached")
        assert term_error.reason == "TERMINATION_REASON"
        assert term_error.domain == "evalprotocol.io"
        assert term_error.metadata["termination_reason"] == "goal_reached"

        # Test extra_info
        extra_error = ErrorInfo.extra_info({"steps": 10, "reward": 0.8})
        assert extra_error.reason == "EXTRA_INFO"
        assert extra_error.domain == "evalprotocol.io"
        assert extra_error.metadata == {"steps": 10, "reward": 0.8}

        # Test rollout_error
        rollout_error = ErrorInfo.rollout_error({"error_code": "E001"})
        assert rollout_error.reason == "ROLLOUT_ERROR"
        assert rollout_error.domain == "evalprotocol.io"
        assert rollout_error.metadata == {"error_code": "E001"}

        # Test stopped_reason
        stopped_error = ErrorInfo.stopped_reason("user_request")
        assert stopped_error.reason == "STOPPED"
        assert stopped_error.domain == "evalprotocol.io"
        assert stopped_error.metadata["reason"] == "user_request"


class TestStatusModel:
    """Test the AIP-193 compatible Status model."""

    def test_status_code_enum_values(self):
        """Test that Status.Code enum has the correct values."""
        assert Status.Code.OK == 0
        assert Status.Code.CANCELLED == 1
        assert Status.Code.UNKNOWN == 2
        assert Status.Code.INVALID_ARGUMENT == 3
        assert Status.Code.DEADLINE_EXCEEDED == 4
        assert Status.Code.NOT_FOUND == 5
        assert Status.Code.ALREADY_EXISTS == 6
        assert Status.Code.PERMISSION_DENIED == 7
        assert Status.Code.RESOURCE_EXHAUSTED == 8
        assert Status.Code.FAILED_PRECONDITION == 9
        assert Status.Code.ABORTED == 10
        assert Status.Code.OUT_OF_RANGE == 11
        assert Status.Code.UNIMPLEMENTED == 12
        assert Status.Code.INTERNAL == 13
        assert Status.Code.UNAVAILABLE == 14
        assert Status.Code.DATA_LOSS == 15
        assert Status.Code.UNAUTHENTICATED == 16
        assert Status.Code.FINISHED == 100  # Custom code

    def test_status_creation_methods(self):
        """Test the convenience methods for creating Status instances."""
        # Test running status
        running_status = Status.rollout_running()
        assert running_status.code == Status.Code.OK
        assert running_status.message == "Rollout is running"
        assert running_status.details == []

        # Test finished status
        finished_status = Status.rollout_finished()
        assert finished_status.code == Status.Code.FINISHED
        assert finished_status.message == "Rollout finished successfully"
        assert finished_status.details == []

        # Test error status
        error_status = Status.rollout_error("Something went wrong")
        assert error_status.code == Status.Code.INTERNAL
        assert error_status.message == "Something went wrong"
        assert error_status.details == []

        # Test error status with extra info
        extra_info = {"error_code": "E001", "timestamp": "2024-01-01"}
        error_status_with_info = Status.rollout_error("Something went wrong", extra_info)
        assert error_status_with_info.code == Status.Code.INTERNAL
        assert error_status_with_info.message == "Something went wrong"
        assert len(error_status_with_info.details) == 1
        assert error_status_with_info.details[0]["@type"] == "type.googleapis.com/google.rpc.ErrorInfo"
        assert error_status_with_info.details[0]["reason"] == "ROLLOUT_ERROR"
        assert error_status_with_info.details[0]["domain"] == "evalprotocol.io"
        assert error_status_with_info.details[0]["metadata"] == extra_info

        # Test stopped status
        stopped_status = Status.rollout_stopped("User requested stop")
        assert stopped_status.code == Status.Code.CANCELLED
        assert stopped_status.message == "User requested stop"
        assert stopped_status.details == []

        # Test with termination reason
        termination_status = Status.with_termination_reason("goal_reached")
        assert termination_status.code == Status.Code.FINISHED
        assert termination_status.message == "Rollout finished"
        assert len(termination_status.details) == 1
        assert termination_status.details[0]["@type"] == "type.googleapis.com/google.rpc.ErrorInfo"
        assert termination_status.details[0]["reason"] == "TERMINATION_REASON"
        assert termination_status.details[0]["domain"] == "evalprotocol.io"
        assert termination_status.details[0]["metadata"]["termination_reason"] == "goal_reached"

        # Test with termination reason and extra info
        extra_info = {"steps": 10, "reward": 0.8}
        termination_status_with_info = Status.with_termination_reason("goal_reached", extra_info)
        assert termination_status_with_info.code == Status.Code.FINISHED
        assert len(termination_status_with_info.details) == 2
        # First detail should be termination reason
        assert termination_status_with_info.details[0]["reason"] == "TERMINATION_REASON"
        # Second detail should be extra info
        assert termination_status_with_info.details[1]["reason"] == "EXTRA_INFO"
        assert termination_status_with_info.details[1]["metadata"] == extra_info

    def test_status_helper_methods(self):
        """Test the helper methods for checking status types."""
        # Test is_running
        running_status = Status.rollout_running()
        assert running_status.is_running() is True
        assert running_status.is_finished() is False
        assert running_status.is_error() is False
        assert running_status.is_stopped() is False

        # Test is_finished
        finished_status = Status.rollout_finished()
        assert finished_status.is_running() is False
        assert finished_status.is_finished() is True
        assert finished_status.is_error() is False
        assert finished_status.is_stopped() is False

        # Test is_error
        error_status = Status.rollout_error("Test error")
        assert error_status.is_running() is False
        assert error_status.is_finished() is False
        assert error_status.is_error() is True
        assert error_status.is_stopped() is False

        # Test is_stopped
        stopped_status = Status.rollout_stopped("Test stop")
        assert stopped_status.is_running() is False
        assert stopped_status.is_finished() is False
        assert stopped_status.is_error() is False
        assert stopped_status.is_stopped() is True

    def test_get_termination_reason(self):
        """Test extracting termination reason from status details."""
        # Status without termination reason
        running_status = Status.rollout_running()
        assert running_status.get_termination_reason() is None

        # Status with termination reason
        termination_status = Status.with_termination_reason("goal_reached")
        assert termination_status.get_termination_reason() == "goal_reached"

        # Status with termination reason and extra info
        extra_info = {"steps": 10}
        termination_status_with_info = Status.with_termination_reason("timeout", extra_info)
        assert termination_status_with_info.get_termination_reason() == "timeout"

    def test_get_extra_info(self):
        """Test extracting extra info from status details."""
        # Status without extra info
        running_status = Status.rollout_running()
        assert running_status.get_extra_info() is None

        # Status with only termination reason (no extra info)
        termination_status = Status.with_termination_reason("goal_reached")
        assert termination_status.get_extra_info() is None

        # Status with extra info
        extra_info = {"steps": 10, "reward": 0.8}
        error_status = Status.rollout_error("Test error", extra_info)
        assert error_status.get_extra_info() == extra_info

        # Status with both termination reason and extra info
        termination_status_with_info = Status.with_termination_reason("goal_reached", extra_info)
        assert termination_status_with_info.get_extra_info() == extra_info

    def test_aip_193_compliance(self):
        """Test that Status model follows AIP-193 standards."""
        # Test ErrorInfo structure
        extra_info = {"error_code": "E001"}
        error_status = Status.rollout_error("Test error", extra_info)

        assert len(error_status.details) == 1
        detail = error_status.details[0]

        # Check AIP-193 ErrorInfo structure
        assert detail["@type"] == "type.googleapis.com/google.rpc.ErrorInfo"
        assert detail["reason"] == "ROLLOUT_ERROR"
        assert detail["domain"] == "evalprotocol.io"
        assert detail["metadata"] == extra_info

        # Test multiple details
        termination_status = Status.with_termination_reason("goal_reached", extra_info)
        assert len(termination_status.details) == 2

        # First detail should be termination reason
        term_detail = termination_status.details[0]
        assert term_detail["@type"] == "type.googleapis.com/google.rpc.ErrorInfo"
        assert term_detail["reason"] == "TERMINATION_REASON"

        # Second detail should be extra info
        extra_detail = termination_status.details[1]
        assert extra_detail["@type"] == "type.googleapis.com/google.rpc.ErrorInfo"
        assert extra_detail["reason"] == "EXTRA_INFO"

    def test_status_serialization(self):
        """Test that Status can be serialized and deserialized."""
        original_status = Status.with_termination_reason("goal_reached", {"steps": 10})

        # Test model_dump
        status_dict = original_status.model_dump()
        assert status_dict["code"] == Status.Code.FINISHED
        assert status_dict["message"] == "Rollout finished"
        assert len(status_dict["details"]) == 2

        # Test model_validate
        reconstructed_status = Status.model_validate(status_dict)
        assert reconstructed_status.code == original_status.code
        assert reconstructed_status.message == original_status.message
        assert len(reconstructed_status.details) == len(original_status.details)
        assert reconstructed_status.get_termination_reason() == "goal_reached"
        assert reconstructed_status.get_extra_info() == {"steps": 10}

    def test_status_equality(self):
        """Test Status equality and comparison."""
        status1 = Status.rollout_running()
        status2 = Status.rollout_running()
        status3 = Status.rollout_finished()

        # Same values should be equal
        assert status1 == status2

        # Different values should not be equal
        assert status1 != status3

        # Test hash
        assert hash(status1) == hash(status2)
        assert hash(status1) != hash(status3)


class TestStatusMigration:
    """Test the migration from RolloutStatus to Status."""

    def test_evaluation_row_default_status(self):
        """Test that EvaluationRow has the correct default status."""
        row = EvaluationRow(messages=[])

        # Should have rollout_status field (not status)
        assert hasattr(row, "rollout_status")
        assert not hasattr(row, "status")

        # Default status should be running
        assert row.rollout_status.code == Status.Code.OK
        assert row.rollout_status.message == "Rollout is running"
        assert row.rollout_status.details == []

    def test_backwards_compatibility_methods(self):
        """Test the backwards compatibility methods."""
        row = EvaluationRow(messages=[])

        # Test get_rollout_status
        status = row.get_rollout_status()
        assert status.code == Status.Code.OK
        assert status.message == "Rollout is running"

        # Test set_rollout_status
        new_status = Status.rollout_finished()
        row.set_rollout_status(new_status)
        assert row.rollout_status.code == Status.Code.FINISHED
        assert row.rollout_status.message == "Rollout finished successfully"

    def test_status_transitions(self):
        """Test transitioning between different status states."""
        row = EvaluationRow(messages=[])

        # Start with running
        assert row.rollout_status.is_running()

        # Transition to finished
        row.rollout_status = Status.rollout_finished()
        assert row.rollout_status.is_finished()
        assert not row.rollout_status.is_running()

        # Transition to error
        row.rollout_status = Status.rollout_error("Something went wrong")
        assert row.rollout_status.is_error()
        assert not row.rollout_status.is_finished()

        # Transition to stopped
        row.rollout_status = Status.rollout_stopped("User requested stop")
        assert row.rollout_status.is_stopped()
        assert not row.rollout_status.is_error()

    def test_termination_reason_integration(self):
        """Test integration of termination reason with status."""
        row = EvaluationRow(messages=[])

        # Set status with termination reason
        termination_status = Status.with_termination_reason("goal_reached", {"steps": 15})
        row.rollout_status = termination_status

        # Should be finished
        assert row.rollout_status.is_finished()

        # Should have termination reason
        assert row.rollout_status.get_termination_reason() == "goal_reached"

        # Should have extra info
        extra_info = row.rollout_status.get_extra_info()
        assert extra_info == {"steps": 15}

    def test_error_handling_integration(self):
        """Test error handling integration with status."""
        row = EvaluationRow(messages=[])

        # Set error status
        error_info = {"error_code": "E001", "line": 42}
        error_status = Status.rollout_error("Runtime error occurred", error_info)
        row.rollout_status = error_status

        # Should be error
        assert row.rollout_status.is_error()

        # Should have error details
        assert row.rollout_status.get_extra_info() == error_info

        # Should not have termination reason
        assert row.rollout_status.get_termination_reason() is None


class TestStatusEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_details(self):
        """Test Status with empty details."""
        status = Status(code=Status.Code.OK, message="Test", details=[])
        assert status.details == []
        assert status.get_termination_reason() is None
        assert status.get_extra_info() is None

    def test_malformed_details(self):
        """Test Status with malformed details."""
        malformed_details = [
            {"not_type": "invalid", "reason": "TEST"},
            {"@type": "type.googleapis.com/google.rpc.ErrorInfo", "metadata": {"termination_reason": "test"}},
        ]
        status = Status(code=Status.Code.OK, message="Test", details=malformed_details)

        # Should handle malformed details gracefully
        assert status.get_termination_reason() == "test"
        assert status.get_extra_info() is None

    def test_duplicate_detail_types(self):
        """Test Status with duplicate detail types."""
        details = [
            {
                "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                "reason": "TERMINATION_REASON",
                "domain": "evalprotocol.io",
                "metadata": {"termination_reason": "first"},
            },
            {
                "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                "reason": "TERMINATION_REASON",
                "domain": "evalprotocol.io",
                "metadata": {"termination_reason": "second"},
            },
        ]
        status = Status(code=Status.Code.OK, message="Test", details=details)

        # Should return the first termination reason found
        assert status.get_termination_reason() == "first"

    def test_large_metadata(self):
        """Test Status with large metadata."""
        large_metadata = {f"key_{i}": f"value_{i}" for i in range(100)}
        status = Status.rollout_error("Test error", large_metadata)

        # Should handle large metadata
        assert status.get_extra_info() == large_metadata
        assert len(status.details) == 1
        assert status.details[0]["metadata"] == large_metadata


if __name__ == "__main__":
    pytest.main([__file__])
