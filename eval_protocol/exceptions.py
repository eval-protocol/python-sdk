"""
Custom exceptions for Eval Protocol that map to gRPC Status codes.

These exceptions provide a clean way to handle errors and map them to appropriate
Status objects following the AIP-193 standard.
"""

from typing import Optional


class EvalProtocolError(Exception):
    """
    Base exception for all Eval Protocol specific errors.

    Maps to Status.Code and can be converted to Status objects for structured logging.
    """

    pass


# Standard gRPC status code exceptions
class CancelledError(EvalProtocolError):
    """Operation was cancelled (Status.Code.CANCELLED = 1)"""

    status_code = 1


class UnknownError(EvalProtocolError):
    """Unknown error occurred (Status.Code.UNKNOWN = 2)"""

    status_code = 2


class InvalidArgumentError(EvalProtocolError):
    """Invalid argument provided (Status.Code.INVALID_ARGUMENT = 3)"""

    status_code = 3


class DeadlineExceededError(EvalProtocolError):
    """Deadline exceeded (Status.Code.DEADLINE_EXCEEDED = 4)"""

    status_code = 4


class NotFoundError(EvalProtocolError):
    """Resource not found (Status.Code.NOT_FOUND = 5)"""

    status_code = 5


class AlreadyExistsError(EvalProtocolError):
    """Resource already exists (Status.Code.ALREADY_EXISTS = 6)"""

    status_code = 6


class PermissionDeniedError(EvalProtocolError):
    """Permission denied (Status.Code.PERMISSION_DENIED = 7)"""

    status_code = 7


class ResourceExhaustedError(EvalProtocolError):
    """Resource exhausted (Status.Code.RESOURCE_EXHAUSTED = 8)"""

    status_code = 8


class FailedPreconditionError(EvalProtocolError):
    """Failed precondition (Status.Code.FAILED_PRECONDITION = 9)"""

    status_code = 9


class AbortedError(EvalProtocolError):
    """Operation was aborted (Status.Code.ABORTED = 10)"""

    status_code = 10


class OutOfRangeError(EvalProtocolError):
    """Value out of range (Status.Code.OUT_OF_RANGE = 11)"""

    status_code = 11


class UnimplementedError(EvalProtocolError):
    """Operation is not implemented (Status.Code.UNIMPLEMENTED = 12)"""

    status_code = 12


class InternalError(EvalProtocolError):
    """Internal server error (Status.Code.INTERNAL = 13)"""

    status_code = 13


class UnavailableError(EvalProtocolError):
    """Service unavailable (Status.Code.UNAVAILABLE = 14)"""

    status_code = 14


class DataLossError(EvalProtocolError):
    """Unrecoverable data loss (Status.Code.DATA_LOSS = 15)"""

    status_code = 15


class UnauthenticatedError(EvalProtocolError):
    """Request lacks valid authentication (Status.Code.UNAUTHENTICATED = 16)"""

    status_code = 16


# Custom EP exceptions
class RolloutFinishedError(EvalProtocolError):
    """Rollout completed successfully (Status.Code.FINISHED = 100)"""

    status_code = 100


class RolloutRunningError(EvalProtocolError):
    """Rollout is still running (Status.Code.RUNNING = 101)"""

    status_code = 101


class ScoreInvalidError(EvalProtocolError):
    """Score is invalid (Status.Code.SCORE_INVALID = 102)"""

    status_code = 102


# Convenience mapping from status codes to exception classes
# Only actual error conditions should raise exceptions
STATUS_CODE_TO_EXCEPTION = {
    0: None,  # OK - success, no exception
    1: CancelledError,
    2: UnknownError,
    3: InvalidArgumentError,
    4: DeadlineExceededError,
    5: NotFoundError,
    6: AlreadyExistsError,
    7: PermissionDeniedError,
    8: ResourceExhaustedError,
    9: FailedPreconditionError,
    10: AbortedError,
    11: OutOfRangeError,
    12: UnimplementedError,
    13: InternalError,
    14: UnavailableError,
    15: DataLossError,
    16: UnauthenticatedError,
    100: None,  # FINISHED - success, no exception
    101: None,  # RUNNING - in progress, no exception
    102: None,  # SCORE_INVALID - success, no exception
}


def exception_for_status_code(code: int, message: str = "") -> Optional[EvalProtocolError]:
    """
    Create an exception instance for a given status code.

    Args:
        code: Status code from Status.Code enum
        message: Optional error message to include in the exception

    Returns:
        Exception instance or None if code is OK (0)
    """
    exception_class = STATUS_CODE_TO_EXCEPTION.get(code)
    if exception_class is None:
        return None
    return exception_class(message) if message else exception_class()
