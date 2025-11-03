"""
Tests for the eval_protocol exception handling system.

Tests the status code to exception mapping functionality:
1. STATUS_CODE_TO_EXCEPTION mapping correctness
2. exception_for_status_code() function behavior
3. Success states don't raise exceptions (0, 100, 101, 102)
4. Error states raise appropriate exceptions (1-16)
5. Exception class inheritance and attributes
6. Integration with existing retry logic
"""

import pytest
from eval_protocol.models import Status
from eval_protocol.exceptions import (
    exception_for_status_code,
    STATUS_CODE_TO_EXCEPTION,
    EvalProtocolError,
    CancelledError,
    UnknownError,
    InvalidArgumentError,
    DeadlineExceededError,
    NotFoundError,
    AlreadyExistsError,
    PermissionDeniedError,
    ResourceExhaustedError,
    FailedPreconditionError,
    AbortedError,
    OutOfRangeError,
    UnimplementedError,
    InternalError,
    UnavailableError,
    DataLossError,
    UnauthenticatedError,
    RolloutFinishedError,
    RolloutRunningError,
    ScoreInvalidError,
)


def test_success_status_codes_no_exception():
    """Test that success/progress status codes don't raise exceptions."""
    success_codes = [
        (0, "OK"),
        (100, "FINISHED"),
        (101, "RUNNING"),
        (102, "SCORE_INVALID"),  # Changed to success state
    ]

    for code, name in success_codes:
        exception = exception_for_status_code(code)
        assert exception is None, f"Status code {code} ({name}) should not raise exception"


def test_error_status_codes_raise_exceptions():
    """Test that error status codes raise appropriate exceptions."""
    error_test_cases = [
        (1, CancelledError, "CANCELLED"),
        (2, UnknownError, "UNKNOWN"),
        (3, InvalidArgumentError, "INVALID_ARGUMENT"),
        (4, DeadlineExceededError, "DEADLINE_EXCEEDED"),
        (5, NotFoundError, "NOT_FOUND"),
        (6, AlreadyExistsError, "ALREADY_EXISTS"),
        (7, PermissionDeniedError, "PERMISSION_DENIED"),
        (8, ResourceExhaustedError, "RESOURCE_EXHAUSTED"),
        (9, FailedPreconditionError, "FAILED_PRECONDITION"),
        (10, AbortedError, "ABORTED"),
        (11, OutOfRangeError, "OUT_OF_RANGE"),
        (12, UnimplementedError, "UNIMPLEMENTED"),
        (13, InternalError, "INTERNAL"),
        (14, UnavailableError, "UNAVAILABLE"),
        (15, DataLossError, "DATA_LOSS"),
        (16, UnauthenticatedError, "UNAUTHENTICATED"),
    ]

    for code, expected_exception_class, name in error_test_cases:
        exception = exception_for_status_code(code)
        assert exception is not None, f"Status code {code} ({name}) should raise exception"
        assert isinstance(exception, expected_exception_class), (
            f"Status code {code} should raise {expected_exception_class.__name__}"
        )
        assert isinstance(exception, EvalProtocolError), "All exceptions should inherit from EvalProtocolError"


def test_status_code_mapping_completeness():
    """Test that STATUS_CODE_TO_EXCEPTION mapping covers all expected codes."""
    expected_codes = [
        0,  # OK
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,  # Standard gRPC codes
        100,
        101,
        102,  # Custom EP codes
    ]

    for code in expected_codes:
        assert code in STATUS_CODE_TO_EXCEPTION, f"Status code {code} missing from mapping"


def test_invalid_status_codes():
    """Test behavior with invalid/unknown status codes."""
    invalid_codes = [-1, 17, 99, 103, 999]

    for code in invalid_codes:
        exception = exception_for_status_code(code)
        assert exception is None, f"Invalid status code {code} should return None"


def test_exception_attributes():
    """Test that exceptions have the expected attributes."""
    # Test a few exception types
    test_cases = [
        (1, CancelledError, "CANCELLED"),
        (5, NotFoundError, "NOT_FOUND"),
        (13, InternalError, "INTERNAL"),
    ]

    for code, expected_class, name in test_cases:
        exception = exception_for_status_code(code)
        assert hasattr(expected_class, "status_code"), f"{expected_class.__name__} should have status_code attribute"
        assert expected_class.status_code == code, f"{expected_class.__name__}.status_code should be {code}"


def test_exception_raising_integration():
    """Test the pattern used in RemoteRolloutProcessor."""
    # Simulate the pattern used in remote_rollout_processor.py
    status_codes_to_test = [
        (0, False),  # OK - should not raise
        (5, True),  # NOT_FOUND - should raise NotFoundError
        (13, True),  # INTERNAL - should raise InternalError
        (100, False),  # FINISHED - should not raise
    ]

    for status_code, should_raise in status_codes_to_test:
        exception = exception_for_status_code(status_code)

        if should_raise:
            assert exception is not None, f"Status code {status_code} should create exception"
            # Test that we can raise it
            with pytest.raises(EvalProtocolError):
                raise exception
        else:
            assert exception is None, f"Status code {status_code} should not create exception"


def test_status_code_enum_consistency():
    """Test that our mapping is consistent with Status.Code enum."""
    # Test that our exception mapping aligns with Status.Code enum
    status_code_mapping = {
        Status.Code.OK: None,
        Status.Code.CANCELLED: CancelledError,
        Status.Code.UNKNOWN: UnknownError,
        Status.Code.INVALID_ARGUMENT: InvalidArgumentError,
        Status.Code.DEADLINE_EXCEEDED: DeadlineExceededError,
        Status.Code.NOT_FOUND: NotFoundError,
        Status.Code.ALREADY_EXISTS: AlreadyExistsError,
        Status.Code.PERMISSION_DENIED: PermissionDeniedError,
        Status.Code.RESOURCE_EXHAUSTED: ResourceExhaustedError,
        Status.Code.FAILED_PRECONDITION: FailedPreconditionError,
        Status.Code.ABORTED: AbortedError,
        Status.Code.OUT_OF_RANGE: OutOfRangeError,
        Status.Code.UNIMPLEMENTED: UnimplementedError,
        Status.Code.INTERNAL: InternalError,
        Status.Code.UNAVAILABLE: UnavailableError,
        Status.Code.DATA_LOSS: DataLossError,
        Status.Code.UNAUTHENTICATED: UnauthenticatedError,
        Status.Code.FINISHED: None,
        Status.Code.RUNNING: None,
        Status.Code.SCORE_INVALID: None,
    }

    for status_code_enum, expected_exception_class in status_code_mapping.items():
        code_value = int(status_code_enum)
        actual_exception_class = STATUS_CODE_TO_EXCEPTION.get(code_value)

        if expected_exception_class is None:
            assert actual_exception_class is None, (
                f"Status.Code.{status_code_enum.name} ({code_value}) should map to None"
            )
        else:
            assert actual_exception_class == expected_exception_class, (
                f"Status.Code.{status_code_enum.name} ({code_value}) should map to {expected_exception_class.__name__}"
            )


def test_exception_inheritance():
    """Test that all exception classes properly inherit from EvalProtocolError."""
    exception_classes = [
        CancelledError,
        UnknownError,
        InvalidArgumentError,
        DeadlineExceededError,
        NotFoundError,
        AlreadyExistsError,
        PermissionDeniedError,
        ResourceExhaustedError,
        FailedPreconditionError,
        AbortedError,
        OutOfRangeError,
        UnimplementedError,
        InternalError,
        UnavailableError,
        DataLossError,
        UnauthenticatedError,
        RolloutFinishedError,
        RolloutRunningError,
        ScoreInvalidError,
    ]

    for exception_class in exception_classes:
        assert issubclass(exception_class, EvalProtocolError), (
            f"{exception_class.__name__} should inherit from EvalProtocolError"
        )
        assert issubclass(exception_class, Exception), f"{exception_class.__name__} should inherit from Exception"


def test_real_world_usage_scenarios():
    """Test realistic usage patterns from RemoteRolloutProcessor."""
    # Test scenarios that might occur in practice
    scenarios = [
        # Success scenarios
        {"status_code": 0, "description": "Successful API call", "should_raise": False},
        {"status_code": 100, "description": "Rollout completed successfully", "should_raise": False},
        {"status_code": 101, "description": "Rollout still in progress", "should_raise": False},
        # Error scenarios that should trigger retry logic
        {
            "status_code": 4,
            "description": "Request timeout",
            "should_raise": True,
            "expected_exception": DeadlineExceededError,
        },
        {
            "status_code": 5,
            "description": "Model not found",
            "should_raise": True,
            "expected_exception": NotFoundError,
        },
        {
            "status_code": 7,
            "description": "API key invalid",
            "should_raise": True,
            "expected_exception": PermissionDeniedError,
        },
        {
            "status_code": 8,
            "description": "Rate limit exceeded",
            "should_raise": True,
            "expected_exception": ResourceExhaustedError,
        },
        {
            "status_code": 13,
            "description": "Internal server error",
            "should_raise": True,
            "expected_exception": InternalError,
        },
        {
            "status_code": 14,
            "description": "Service temporarily unavailable",
            "should_raise": True,
            "expected_exception": UnavailableError,
        },
    ]

    for scenario in scenarios:
        status_code = scenario["status_code"]
        description = scenario["description"]
        should_raise = scenario["should_raise"]

        # This is the pattern used in RemoteRolloutProcessor
        exception = exception_for_status_code(status_code)

        if should_raise:
            expected_exception = scenario["expected_exception"]
            assert exception is not None, f"Scenario '{description}' should create exception"
            assert isinstance(exception, expected_exception), (
                f"Scenario '{description}' should create {expected_exception.__name__}"
            )

            # Test that the exception can be raised and caught for retry logic
            with pytest.raises(expected_exception):
                raise exception

        else:
            assert exception is None, f"Scenario '{description}' should not create exception"


def test_exception_status_code_attributes():
    """Test that all exceptions have correct status_code attributes."""
    expected_mappings = [
        (CancelledError, 1),
        (UnknownError, 2),
        (InvalidArgumentError, 3),
        (DeadlineExceededError, 4),
        (NotFoundError, 5),
        (AlreadyExistsError, 6),
        (PermissionDeniedError, 7),
        (ResourceExhaustedError, 8),
        (FailedPreconditionError, 9),
        (AbortedError, 10),
        (OutOfRangeError, 11),
        (UnimplementedError, 12),
        (InternalError, 13),
        (UnavailableError, 14),
        (DataLossError, 15),
        (UnauthenticatedError, 16),
        (RolloutFinishedError, 100),
        (RolloutRunningError, 101),
        (ScoreInvalidError, 102),
    ]

    for exception_class, expected_code in expected_mappings:
        assert hasattr(exception_class, "status_code"), f"{exception_class.__name__} should have status_code attribute"
        assert exception_class.status_code == expected_code, (
            f"{exception_class.__name__}.status_code should be {expected_code}"
        )


def test_integration_with_retry_logic():
    """Test that our exceptions integrate properly with existing retry logic."""
    from eval_protocol.pytest.exception_config import DEFAULT_RETRYABLE_EXCEPTIONS

    # Test that our error exceptions are covered by retry logic
    our_error_exceptions = [
        UnknownError,
        DeadlineExceededError,
        NotFoundError,
        PermissionDeniedError,
        UnavailableError,
        UnauthenticatedError,
        ResourceExhaustedError,
    ]

    for exception_class in our_error_exceptions:
        assert exception_class in DEFAULT_RETRYABLE_EXCEPTIONS, (
            f"{exception_class.__name__} should be in DEFAULT_RETRYABLE_EXCEPTIONS for retry support"
        )


def test_exception_message_preservation():
    """Test that error messages are properly preserved in exceptions."""
    test_cases = [
        (13, "test error", InternalError),
        (5, "Model xyz not found", NotFoundError),
        (7, "Invalid API key", PermissionDeniedError),
    ]

    for status_code, message, expected_exception_class in test_cases:
        # Test with message
        exception = exception_for_status_code(status_code, message)
        assert exception is not None
        assert isinstance(exception, expected_exception_class)
        assert str(exception) == message, f"Exception should preserve message '{message}'"

        # Test without message (should still work)
        exception_no_msg = exception_for_status_code(status_code)
        assert exception_no_msg is not None
        assert isinstance(exception_no_msg, expected_exception_class)
