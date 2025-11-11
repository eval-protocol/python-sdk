import os
import logging
import importlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar, Dict, List, Literal, Optional, TypedDict, Union

JSONType = Union[Dict[str, Any], List[Any], str, int, float, bool, None]

from openai.types import CompletionUsage
from openai.types.chat.chat_completion_message import (
    FunctionCall,
)
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from pydantic import BaseModel, ConfigDict, Field

from eval_protocol.get_pep440_version import get_pep440_version
from eval_protocol.human_id import generate_id
from eval_protocol.types import TerminationReason


class ErrorInfo(BaseModel):
    """
    AIP-193 ErrorInfo model for structured error details.

    This model follows Google's AIP-193 standard for ErrorInfo:
    https://google.aip.dev/193#errorinfo

    Attributes:
        reason (str): A short snake_case description of the cause of the error.
        domain (str): The logical grouping to which the reason belongs.
        metadata (Dict[str, Any]): Additional dynamic information as context.
    """

    # Constants for reason values
    REASON_TERMINATION_REASON: ClassVar[str] = "TERMINATION_REASON"
    REASON_EXTRA_INFO: ClassVar[str] = "EXTRA_INFO"

    # Domain constant
    DOMAIN: ClassVar[str] = "evalprotocol.io"

    reason: str = Field(..., description="Short snake_case description of the error cause")
    domain: str = Field(..., description="Logical grouping for the error reason")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional dynamic information as context")

    def to_aip193_format(self) -> Dict[str, Any]:
        """Convert to AIP-193 format with @type field."""
        return {
            "@type": "type.googleapis.com/google.rpc.ErrorInfo",
            "reason": self.reason,
            "domain": self.domain,
            "metadata": self.metadata,
        }

    @classmethod
    def termination_reason(cls, reason: TerminationReason) -> "ErrorInfo":
        """Create an ErrorInfo for termination reason."""
        # Convert TerminationReason enum to string if needed
        reason_str = reason.value if isinstance(reason, TerminationReason) else reason
        return cls(
            reason=cls.REASON_TERMINATION_REASON, domain=cls.DOMAIN, metadata={"termination_reason": reason_str}
        )

    @classmethod
    def extra_info(cls, metadata: Dict[str, Any]) -> "ErrorInfo":
        """Create an ErrorInfo for extra information."""
        return cls(reason=cls.REASON_EXTRA_INFO, domain=cls.DOMAIN, metadata=metadata)


class Status(BaseModel):
    """
    AIP-193 compatible Status model for standardized error responses.

    This model follows Google's AIP-193 standard for error handling:
    https://google.aip.dev/193

    Attributes:
        code (int): The status code, must be the numeric value of one of the elements
                   of google.rpc.Code enum (e.g., 5 for NOT_FOUND).
        message (str): Developer-facing, human-readable debug message in English.
        details (List[Dict[str, Any]]): Additional error information, each packed in
                                       a google.protobuf.Any message format.
    """

    code: "Status.Code" = Field(..., description="The status code from google.rpc.Code enum")
    message: str = Field(..., description="Developer-facing, human-readable debug message in English")
    details: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Additional error information, each packed in a google.protobuf.Any message format",
    )

    # Convenience constants for common status codes
    class Code(int, Enum):
        """Common gRPC status codes as defined in google.rpc.Code"""

        OK = 0
        CANCELLED = 1
        UNKNOWN = 2
        INVALID_ARGUMENT = 3
        DEADLINE_EXCEEDED = 4
        NOT_FOUND = 5
        ALREADY_EXISTS = 6
        PERMISSION_DENIED = 7
        RESOURCE_EXHAUSTED = 8
        FAILED_PRECONDITION = 9
        ABORTED = 10
        OUT_OF_RANGE = 11
        UNIMPLEMENTED = 12
        INTERNAL = 13
        UNAVAILABLE = 14
        DATA_LOSS = 15
        UNAUTHENTICATED = 16

        # Custom codes for EP (using higher numbers to avoid conflicts)
        FINISHED = 100
        RUNNING = 101
        SCORE_INVALID = 102

    @classmethod
    def rollout_running(cls) -> "Status":
        """Create a status indicating the rollout is running."""
        return cls(code=cls.Code.RUNNING, message="Rollout is running", details=[])

    @classmethod
    def eval_running(cls) -> "Status":
        """Create a status indicating the evaluation is running."""
        return cls(code=cls.Code.RUNNING, message="Evaluation is running", details=[])

    @classmethod
    def eval_finished(cls) -> "Status":
        """Create a status indicating the evaluation finished."""
        return cls(code=cls.Code.FINISHED, message="Evaluation finished", details=[])

    @staticmethod
    def _build_details_with_extra_info(extra_info: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Helper to build details list from extra_info."""
        if extra_info:
            return [ErrorInfo.extra_info(extra_info).to_aip193_format()]
        return []

    @classmethod
    def aborted(cls, message: str, details: Optional[List[Dict[str, Any]]] = None) -> "Status":
        """Create a status indicating the evaluation was aborted."""
        return cls(code=cls.Code.ABORTED, message=message, details=details or [])

    @classmethod
    def rollout_finished(
        cls,
        termination_reason: Optional[TerminationReason] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> "Status":
        """Create a status indicating the rollout finished."""
        details = []
        if termination_reason:
            details.append(ErrorInfo.termination_reason(termination_reason).to_aip193_format())
        if extra_info:
            details.append(ErrorInfo.extra_info(extra_info).to_aip193_format())
        return cls.finished("Rollout finished", details)

    @classmethod
    def finished(cls, message: str, details: Optional[List[Dict[str, Any]]] = None) -> "Status":
        """Create a status indicating the rollout finished."""
        return cls(code=cls.Code.FINISHED, message=message, details=details or [])

    # Error methods organized by Status.Code enum values (1-16)

    # CANCELLED = 1
    @classmethod
    def rollout_cancelled_error(cls, error_message: str, extra_info: Optional[Dict[str, Any]] = None) -> "Status":
        """Create a status indicating the rollout was cancelled."""
        return cls.cancelled_error(error_message, cls._build_details_with_extra_info(extra_info))

    @classmethod
    def cancelled_error(cls, error_message: str, details: Optional[List[Dict[str, Any]]] = None) -> "Status":
        """Create a status indicating the operation was cancelled."""
        return cls(code=cls.Code.CANCELLED, message=error_message, details=details or [])

    # UNKNOWN = 2
    @classmethod
    def rollout_unknown_error(cls, error_message: str, extra_info: Optional[Dict[str, Any]] = None) -> "Status":
        """Create a status indicating the rollout failed with an unknown error."""
        return cls.unknown_error(error_message, cls._build_details_with_extra_info(extra_info))

    @classmethod
    def unknown_error(cls, error_message: str, details: Optional[List[Dict[str, Any]]] = None) -> "Status":
        """Create a status indicating an unknown error occurred."""
        return cls(code=cls.Code.UNKNOWN, message=error_message, details=details or [])

    # INVALID_ARGUMENT = 3
    @classmethod
    def rollout_invalid_argument_error(
        cls, error_message: str, extra_info: Optional[Dict[str, Any]] = None
    ) -> "Status":
        """Create a status indicating the rollout failed with an invalid argument error."""
        return cls.invalid_argument_error(error_message, cls._build_details_with_extra_info(extra_info))

    @classmethod
    def invalid_argument_error(cls, error_message: str, details: Optional[List[Dict[str, Any]]] = None) -> "Status":
        """Create a status indicating an invalid argument error occurred."""
        return cls(code=cls.Code.INVALID_ARGUMENT, message=error_message, details=details or [])

    # DEADLINE_EXCEEDED = 4
    @classmethod
    def rollout_deadline_exceeded_error(
        cls, error_message: str, extra_info: Optional[Dict[str, Any]] = None
    ) -> "Status":
        """Create a status indicating the rollout failed with a deadline exceeded error."""
        return cls.deadline_exceeded_error(error_message, cls._build_details_with_extra_info(extra_info))

    @classmethod
    def deadline_exceeded_error(cls, error_message: str, details: Optional[List[Dict[str, Any]]] = None) -> "Status":
        """Create a status indicating a deadline exceeded error occurred."""
        return cls(code=cls.Code.DEADLINE_EXCEEDED, message=error_message, details=details or [])

    # NOT_FOUND = 5
    @classmethod
    def rollout_not_found_error(cls, error_message: str, extra_info: Optional[Dict[str, Any]] = None) -> "Status":
        """Create a status indicating the rollout failed with a not found error."""
        return cls.not_found_error(error_message, cls._build_details_with_extra_info(extra_info))

    @classmethod
    def not_found_error(cls, error_message: str, details: Optional[List[Dict[str, Any]]] = None) -> "Status":
        """Create a status indicating a not found error occurred."""
        return cls(code=cls.Code.NOT_FOUND, message=error_message, details=details or [])

    # ALREADY_EXISTS = 6
    @classmethod
    def rollout_already_exists_error(cls, error_message: str, extra_info: Optional[Dict[str, Any]] = None) -> "Status":
        """Create a status indicating the rollout failed with an already exists error."""
        return cls.already_exists_error(error_message, cls._build_details_with_extra_info(extra_info))

    @classmethod
    def already_exists_error(cls, error_message: str, details: Optional[List[Dict[str, Any]]] = None) -> "Status":
        """Create a status indicating an already exists error occurred."""
        return cls(code=cls.Code.ALREADY_EXISTS, message=error_message, details=details or [])

    # PERMISSION_DENIED = 7
    @classmethod
    def rollout_permission_denied_error(
        cls, error_message: str, extra_info: Optional[Dict[str, Any]] = None
    ) -> "Status":
        """Create a status indicating the rollout failed with a permission denied error."""
        return cls.permission_denied_error(error_message, cls._build_details_with_extra_info(extra_info))

    @classmethod
    def permission_denied_error(cls, error_message: str, details: Optional[List[Dict[str, Any]]] = None) -> "Status":
        """Create a status indicating a permission denied error occurred."""
        return cls(code=cls.Code.PERMISSION_DENIED, message=error_message, details=details or [])

    # RESOURCE_EXHAUSTED = 8
    @classmethod
    def rollout_resource_exhausted_error(
        cls, error_message: str, extra_info: Optional[Dict[str, Any]] = None
    ) -> "Status":
        """Create a status indicating the rollout failed with a resource exhausted error."""
        return cls.resource_exhausted_error(error_message, cls._build_details_with_extra_info(extra_info))

    @classmethod
    def resource_exhausted_error(cls, error_message: str, details: Optional[List[Dict[str, Any]]] = None) -> "Status":
        """Create a status indicating a resource exhausted error occurred."""
        return cls(code=cls.Code.RESOURCE_EXHAUSTED, message=error_message, details=details or [])

    # FAILED_PRECONDITION = 9
    @classmethod
    def rollout_failed_precondition_error(
        cls, error_message: str, extra_info: Optional[Dict[str, Any]] = None
    ) -> "Status":
        """Create a status indicating the rollout failed with a failed precondition error."""
        return cls.failed_precondition_error(error_message, cls._build_details_with_extra_info(extra_info))

    @classmethod
    def failed_precondition_error(cls, error_message: str, details: Optional[List[Dict[str, Any]]] = None) -> "Status":
        """Create a status indicating a failed precondition error occurred."""
        return cls(code=cls.Code.FAILED_PRECONDITION, message=error_message, details=details or [])

    # ABORTED = 10
    @classmethod
    def rollout_aborted_error(cls, error_message: str, extra_info: Optional[Dict[str, Any]] = None) -> "Status":
        """Create a status indicating the rollout was aborted."""
        return cls.aborted(error_message, cls._build_details_with_extra_info(extra_info))

    # OUT_OF_RANGE = 11
    @classmethod
    def rollout_out_of_range_error(cls, error_message: str, extra_info: Optional[Dict[str, Any]] = None) -> "Status":
        """Create a status indicating the rollout failed with an out of range error."""
        return cls.out_of_range_error(error_message, cls._build_details_with_extra_info(extra_info))

    @classmethod
    def out_of_range_error(cls, error_message: str, details: Optional[List[Dict[str, Any]]] = None) -> "Status":
        """Create a status indicating an out of range error occurred."""
        return cls(code=cls.Code.OUT_OF_RANGE, message=error_message, details=details or [])

    # UNIMPLEMENTED = 12
    @classmethod
    def rollout_unimplemented_error(cls, error_message: str, extra_info: Optional[Dict[str, Any]] = None) -> "Status":
        """Create a status indicating the rollout failed with an unimplemented error."""
        return cls.unimplemented_error(error_message, cls._build_details_with_extra_info(extra_info))

    @classmethod
    def unimplemented_error(cls, error_message: str, details: Optional[List[Dict[str, Any]]] = None) -> "Status":
        """Create a status indicating an unimplemented error occurred."""
        return cls(code=cls.Code.UNIMPLEMENTED, message=error_message, details=details or [])

    # INTERNAL = 13
    @classmethod
    def rollout_internal_error(cls, error_message: str, extra_info: Optional[Dict[str, Any]] = None) -> "Status":
        """Create a status indicating the rollout failed with an internal error."""
        return cls.internal_error(error_message, cls._build_details_with_extra_info(extra_info))

    @classmethod
    def internal_error(cls, error_message: str, details: Optional[List[Dict[str, Any]]] = None) -> "Status":
        """Create a status indicating an internal error occurred."""
        return cls(code=cls.Code.INTERNAL, message=error_message, details=details or [])

    # For backwards compatibility
    @classmethod
    def rollout_error(cls, error_message: str, extra_info: Optional[Dict[str, Any]] = None) -> "Status":
        """Create a status indicating the rollout failed with an error."""
        return cls.internal_error(error_message, cls._build_details_with_extra_info(extra_info))

    @classmethod
    def error(cls, error_message: str, details: Optional[List[Dict[str, Any]]] = None) -> "Status":
        """Create a status indicating an error occurred."""
        return cls(code=cls.Code.INTERNAL, message=error_message, details=details or [])

    # UNAVAILABLE = 14
    @classmethod
    def rollout_unavailable_error(cls, error_message: str, extra_info: Optional[Dict[str, Any]] = None) -> "Status":
        """Create a status indicating the rollout failed with an unavailable error."""
        return cls.unavailable_error(error_message, cls._build_details_with_extra_info(extra_info))

    @classmethod
    def unavailable_error(cls, error_message: str, details: Optional[List[Dict[str, Any]]] = None) -> "Status":
        """Create a status indicating an unavailable error occurred."""
        return cls(code=cls.Code.UNAVAILABLE, message=error_message, details=details or [])

    # DATA_LOSS = 15
    @classmethod
    def rollout_data_loss_error(cls, error_message: str, extra_info: Optional[Dict[str, Any]] = None) -> "Status":
        """Create a status indicating the rollout failed with a data loss error."""
        return cls.data_loss_error(error_message, cls._build_details_with_extra_info(extra_info))

    @classmethod
    def data_loss_error(cls, error_message: str, details: Optional[List[Dict[str, Any]]] = None) -> "Status":
        """Create a status indicating a data loss error occurred."""
        return cls(code=cls.Code.DATA_LOSS, message=error_message, details=details or [])

    # UNAUTHENTICATED = 16
    @classmethod
    def rollout_unauthenticated_error(
        cls, error_message: str, extra_info: Optional[Dict[str, Any]] = None
    ) -> "Status":
        """Create a status indicating the rollout failed with an unauthenticated error."""
        return cls.unauthenticated_error(error_message, cls._build_details_with_extra_info(extra_info))

    @classmethod
    def unauthenticated_error(cls, error_message: str, details: Optional[List[Dict[str, Any]]] = None) -> "Status":
        """Create a status indicating an unauthenticated error occurred."""
        return cls(code=cls.Code.UNAUTHENTICATED, message=error_message, details=details or [])

    @classmethod
    def score_invalid(
        cls, message: str = "Score is invalid", details: Optional[List[Dict[str, Any]]] = None
    ) -> "Status":
        """Create a status indicating the score is invalid."""
        return cls(code=cls.Code.SCORE_INVALID, message=message, details=details or [])

    def is_running(self) -> bool:
        """Check if the status indicates the rollout is running."""
        return self.code == self.Code.RUNNING

    def is_finished(self) -> bool:
        """Check if the status indicates the rollout finished successfully."""
        return self.code == self.Code.FINISHED

    def is_error(self) -> bool:
        """Check if the status indicates the rollout failed with an error."""
        return self.code == self.Code.INTERNAL

    def is_stopped(self) -> bool:
        """Check if the status indicates the rollout was stopped."""
        return self.code == self.Code.CANCELLED

    def is_score_invalid(self) -> bool:
        """Check if the status indicates the score is invalid."""
        return self.code == self.Code.SCORE_INVALID

    def get_termination_reason(self) -> Optional[TerminationReason]:
        """Extract termination reason from details if present."""
        for detail in self.details:
            metadata = detail.get("metadata", {})
            if detail.get("reason") == ErrorInfo.REASON_TERMINATION_REASON and "termination_reason" in metadata:
                try:
                    return TerminationReason.from_str(metadata["termination_reason"])
                except ValueError:
                    # If the reason is not a valid enum value, return None
                    return None
        return None

    def get_extra_info(self) -> Optional[Dict[str, Any]]:
        """Extract extra info from details if present."""
        for detail in self.details:
            metadata = detail.get("metadata", {})
            reason = detail.get("reason")
            # Skip termination_reason and stopped details, return other error info
            if reason in [ErrorInfo.REASON_EXTRA_INFO]:
                return metadata
        return None

    def __hash__(self) -> int:
        """Generate a hash for the Status object."""
        # Use a stable hash based on code, message, and details
        import hashlib

        # Create a stable string representation
        hash_data = f"{self.code}:{self.message}:{len(self.details)}"

        # Add details content for more uniqueness
        for detail in sorted(self.details, key=lambda x: str(x)):
            hash_data += f":{str(detail)}"

        # Generate hash
        hash_obj = hashlib.sha256(hash_data.encode("utf-8"))
        return int.from_bytes(hash_obj.digest()[:8], byteorder="big")


class ChatCompletionContentPartTextParam(BaseModel):
    text: str = Field(..., description="The text content.")
    type: Literal["text"] = Field("text", description="The type of the content part.")

    # Provide dict-like access for tests and ergonomic usage
    def __getitem__(self, key: str) -> Any:
        if key == "text":
            return self.text
        if key == "type":
            return self.type
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        return (k for k in ("text", "type"))

    def values(self):
        return (self.text, self.type)

    def items(self):
        return [("text", self.text), ("type", self.type)]

    def __iter__(self):
        # Iterate over keys only
        return iter(["text", "type"])


class Message(BaseModel):
    """Chat message model with trajectory evaluation support."""

    role: str  # assistant, user, system, tool
    content: Optional[Union[str, List[ChatCompletionContentPartTextParam]]] = Field(
        default="", description="The content of the message."
    )
    reasoning_content: Optional[str] = Field(
        default=None, description="Optional hidden chain-of-thought or reasoning content."
    )
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None
    function_call: Optional[FunctionCall] = None
    control_plane_step: Optional[Dict[str, Any]] = None
    weight: Optional[int] = None

    def dump_mdoel_for_chat_completion_request(self):
        """Only keep chat completion accepted fields"""
        return self.model_dump(exclude_none=True, exclude={"control_plane_step", "reasoning_content", "weight"})

    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        if isinstance(obj, dict):
            if "role" not in obj:
                raise ValueError("Role is required")
            # Be lenient: if tool_calls entries are missing required 'id', synthesize one
            tool_calls_obj = obj.get("tool_calls")
            if isinstance(tool_calls_obj, list):
                fixed_tool_calls = []
                for tc in tool_calls_obj:
                    if isinstance(tc, dict):
                        if not tc.get("id"):
                            tc = {**tc, "id": generate_id()}
                    fixed_tool_calls.append(tc)
                obj = {**obj, "tool_calls": fixed_tool_calls}
        return super().model_validate(obj, *args, **kwargs)


class MetricResult(BaseModel):
    """Result of a single metric evaluation.

    Attributes:
        is_score_valid (bool): Whether the score is valid for this metric (required).
        score (float): The score for this metric.
        reason (str): Explanation for the score.
    """

    is_score_valid: bool = True
    score: float = Field(..., ge=0.0, le=1.0)
    reason: str
    data: Dict[str, Any] = Field(default_factory=dict, description="Optional extra metric data for debugging.")

    def __getitem__(self, key: str) -> Any:
        if key in self.__fields__:  # Changed to __fields__ for Pydantic v1 compatibility
            value = getattr(self, key)
            return value
        raise KeyError(f"'{key}'")

    def __contains__(self, key: str) -> bool:
        return key in self.__fields__  # Changed to __fields__

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self):
        return self.__fields__.keys()  # Changed to __fields__

    def values(self):
        # For consistency with __getitem__ returning raw attribute values (including nested models)
        return [getattr(self, key) for key in self.__fields__.keys()]  # Changed to __fields__

    def items(self):
        # Exclude 'data' from items to keep items hashable and match tests
        return [(key, getattr(self, key)) for key in self.__fields__.keys() if key != "data"]  # Changed to __fields__

    def __iter__(self):
        # Exclude 'data' to match expectations in tests
        return iter([k for k in self.__fields__.keys() if k != "data"])  # Changed to __fields__


class StepOutput(BaseModel):
    """Defines the base reward and other metrics for a single conceptual step within a rollout,
    as determined by the user's reward function.
    """

    step_index: Union[int, str] = Field(
        description="User-defined index for the step (e.g., assistant message index, turn number). This is used by the system to map this output to the internal StepData."
    )
    base_reward: float = Field(description="Base reward calculated by the user's reward function for this step.")
    terminated: bool = Field(default=False, description="Whether the environment signaled termination at this step.")
    control_plane_info: Optional[Dict[str, Any]] = Field(
        default=None, description="Structured info from the environment's control plane."
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional dictionary of custom metrics for this step.",
    )
    reason: Optional[str] = Field(
        default=None,
        description="Optional explanation for the step's base reward or metrics.",
    )


class EvaluateResult(BaseModel):
    """The complete result of an evaluator.
    For standard evaluation, it provides an overall score and component metrics.
    For Reinforcement Learning, it can also provide per-step base rewards via 'step_outputs'.

    This unified model serves both per-turn and per-trajectory evaluation scenarios.

    Attributes:
        score (float): The overall evaluation score.
        is_score_valid (bool): Whether the overall score is valid. Defaults to True.
        reason (Optional[str]): Optional explanation for the overall score.
        metrics (Dict[str, MetricResult]): Dictionary of component metrics for detailed evaluation.
        step_outputs (Optional[List[StepOutput]]): For RL, a list of outputs for each conceptual step,
                                                  providing base rewards.
        error (Optional[str]): Optional error message if evaluation failed.
        trajectory_info (Optional[Dict[str, Any]]): Additional trajectory-level information.
        final_control_plane_info (Optional[Dict[str, Any]]): The final control plane state that led to termination.
        agg_score (Optional[float]): The aggregated score of the evaluation across all runs.
        standard_error (Optional[float]): The standard error of the evaluation across all runs.
    """

    score: float = Field(..., description="The overall evaluation score, typically between 0.0 and 1.0.")
    is_score_valid: bool = Field(default=True, description="Whether the overall score is valid.")
    reason: Optional[str] = Field(default=None, description="Optional explanation for the overall score.")
    metrics: Dict[str, MetricResult] = Field(
        default_factory=dict,
        description="Dictionary of component metrics for detailed breakdown.",
    )

    # New field for RL per-step base rewards
    step_outputs: Optional[List[StepOutput]] = Field(
        default=None,
        description="For RL, a list of outputs for each conceptual step, providing base rewards.",
    )

    error: Optional[str] = Field(
        default=None,
        description="Optional error message if the evaluation itself encountered an issue.",
    )

    # New fields for unified trajectory and row-wise results
    trajectory_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional trajectory-level information (duration, steps, termination_reason, etc.).",
    )

    final_control_plane_info: Optional[Dict[str, Any]] = Field(
        default=None, description="The final control plane state that led to termination."
    )

    agg_score: Optional[float] = Field(
        default=None,
        description="The aggregated score of the evaluation across all runs.",
    )

    standard_error: Optional[float] = Field(
        default=None,
        description="The standard error of the evaluation across all runs.",
    )

    def __getitem__(self, key: str) -> Any:
        if key in self.__fields__:  # Changed to __fields__
            value = getattr(self, key)
            # If the value is a dict of MetricResult, and we want __getitem__ on metrics
            # to return a dict of dicts (rather than dict of MetricResult objects),
            # we'd need special handling here.
            # For now, return the raw attribute value, consistent with MetricResult.__getitem__
            return value
        raise KeyError(f"'{key}'")

    def __contains__(self, key: str) -> bool:
        return key in self.__fields__  # Changed to __fields__

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self):
        return self.__fields__.keys()  # Changed to __fields__

    def values(self):
        # For consistency with __getitem__ returning raw attribute values
        return [getattr(self, key) for key in self.__fields__.keys()]  # Changed to __fields__

    def items(self):
        return [(key, getattr(self, key)) for key in self.__fields__.keys()]  # Changed to __fields__

    def __iter__(self):
        return iter(self.__fields__.keys())  # Changed to __fields__


CompletionParams = Dict[str, Any]
"""
The completion parameters for the respective LLM SDK or agent framework.
Depending on the rollout processor, this might be the parameters passed to
LiteLLM completion call or parameters for the "run" method of the "Agent" class
in Pydantic AI.  You can also customize this dictionary to whatever you need if
you implement your own custom rollout processor.
"""


class InputMetadata(BaseModel):
    """Comprehensive metadata for input to evaluation and logging systems."""

    model_config = ConfigDict(extra="allow")

    row_id: Optional[str] = Field(
        default=None,
        description=(
            "Unique string to ID the row. If not provided, a stable hash will be generated "
            "based on the row's content. The hash removes fields that are not typically stable "
            "across processes such as created_at, execution_metadata, and pid."
        ),
    )
    completion_params: CompletionParams = Field(
        default_factory=dict, description="Completion endpoint parameters used"
    )
    dataset_info: Optional[Dict[str, Any]] = Field(
        default=None, description="Dataset row details: seed, system_prompt, environment_context, etc"
    )
    session_data: Optional[Dict[str, Any]] = Field(
        default=None, description="Session metadata like timestamp (input only, no duration/usage)"
    )


class EvaluationThreshold(BaseModel):
    """Threshold configuration for evaluation tests.

    The success field is required - tests must specify a minimum success rate.
    The standard_error field is optional - if provided, tests must also meet the maximum standard error requirement.
    """

    success: float = Field(
        ..., description="Minimum success rate threshold (fraction of total score, 0.0 to 1.0)", ge=0.0, le=1.0
    )
    standard_error: float | None = Field(
        default=None,
        description="Maximum standard error threshold (fraction of total score, 0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )


class EvaluationThresholdDict(TypedDict):
    success: float
    standard_error: float | None


class EvalMetadata(BaseModel):
    """Metadata about the evaluation that was run."""

    name: str = Field(..., description="Name of the evaluation")
    description: Optional[str] = Field(None, description="Description of the evaluation")
    version: str = Field(
        default_factory=get_pep440_version,
        description="Version of the evaluation. Should be populated with a PEP 440 version string.",
    )
    status: Optional[Status] = Field(None, description="Status of the evaluation")
    num_runs: int = Field(..., description="Number of times the evaluation was repeated")
    aggregation_method: str = Field(..., description="Method used to aggregate scores across runs")
    passed_threshold: Optional[EvaluationThreshold] = Field(
        None, description="Threshold configuration for test success"
    )
    passed: Optional[bool] = Field(None, description="Whether the evaluation passed based on the threshold")


class CostMetrics(BaseModel):
    """Cost metrics for LLM API calls."""

    input_cost: Optional[float] = Field(None, description="Cost in USD for input tokens.")

    output_cost: Optional[float] = Field(None, description="Cost in USD for output tokens.")

    total_cost_dollar: Optional[float] = Field(None, description="Total cost in USD for the API call.")


class ExecutionMetadata(BaseModel):
    """Metadata about the execution of the evaluation."""

    invocation_id: Optional[str] = Field(
        default_factory=generate_id,
        description="The ID of the invocation that this row belongs to.",
    )

    experiment_id: Optional[str] = Field(
        default_factory=generate_id,
        description="The ID of the experiment that this row belongs to.",
    )

    rollout_id: Optional[str] = Field(
        default_factory=generate_id,
        description="The ID of the rollout that this row belongs to.",
    )

    run_id: Optional[str] = Field(
        default=None,
        description=("The ID of the run that this row belongs to."),
    )

    usage: Optional[CompletionUsage] = Field(
        default=None, description="Token usage statistics from LLM calls during execution."
    )

    cost_metrics: Optional[CostMetrics] = Field(default=None, description="Cost breakdown for LLM API calls.")

    duration_seconds: Optional[float] = Field(
        default=None,
        description="Processing duration in seconds for this evaluation row. Note that if it gets retried, this will be the duration of the last attempt.",
    )

    experiment_duration_seconds: Optional[float] = Field(
        default=None,
        description="Processing duration in seconds for an entire experiment. Note that includes time it took for retries.",
    )


class EvaluationRow(BaseModel):
    """
    Unified data structure for a single evaluation unit that contains messages,
    tools, and evaluation results. This can represent either a single turn evaluation
    or a complete trajectory evaluation.

    This model serves as the canonical format for evaluation data across the system,
    supporting both row-wise batch evaluation and trajectory-based RL evaluation.
    """

    model_config = ConfigDict(extra="allow")

    # Core OpenAI ChatCompletion compatible conversation data
    messages: List[Message] = Field(
        default_factory=list, description="List of messages in the conversation. Also known as a trajectory."
    )

    # Tool and function call information
    tools: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Available tools/functions that were provided to the agent."
    )

    # Input-related metadata (grouped together for cleaner organization)
    input_metadata: InputMetadata = Field(
        default_factory=lambda: InputMetadata(),
        description="Metadata related to the input (dataset info, model config, session data, etc.).",
    )

    rollout_status: Status = Field(
        default_factory=Status.rollout_running,
        description="The status of the rollout following AIP-193 standards.",
    )

    # Ground truth reference (moved from EvaluateResult to top level)
    ground_truth: Optional[JSONType] = Field(
        default=None, description="JSON-serializable ground truth reference for this evaluation."
    )

    # Unified evaluation result
    evaluation_result: Optional[EvaluateResult] = Field(
        default=None, description="The evaluation result for this row/trajectory."
    )

    execution_metadata: ExecutionMetadata = Field(
        default_factory=lambda: ExecutionMetadata(run_id=None),
        description="Metadata about the execution of the evaluation.",
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="The timestamp when the row was created (UTC).",
    )

    eval_metadata: Optional[EvalMetadata] = Field(
        default=None, description="Metadata about the evaluation that was run."
    )

    pid: Optional[int] = Field(
        default=None,
        description="The PID of the process that created the row. This is used by the evaluation watcher to detect stopped evaluations.",
    )

    def is_trajectory_evaluation(self) -> bool:
        """
        Returns True if this represents a trajectory evaluation (has step_outputs),
        False if it represents a single turn evaluation.
        """
        return (
            self.evaluation_result is not None
            and self.evaluation_result.step_outputs is not None
            and len(self.evaluation_result.step_outputs) > 0
        )

    def get_conversation_length(self) -> int:
        """Returns the number of messages in the conversation."""
        return len(self.messages)

    def get_system_message(self) -> Message:
        """Returns the system message from the conversation. Returns empty Message if none found."""
        system_messages = [msg for msg in self.messages if msg.role == "system"]
        if not system_messages:
            return Message(role="system", content="")
        return system_messages[0]

    def get_assistant_messages(self) -> List[Message]:
        """Returns only the assistant messages from the conversation."""
        return [msg for msg in self.messages if msg.role == "assistant"]

    def last_assistant_message(self) -> Optional[Message]:
        """Returns the last assistant message from the conversation. Returns None if none found."""
        assistant_messages = self.get_assistant_messages()
        if not assistant_messages:
            return None
        return assistant_messages[-1]

    def get_first_user_message(self) -> Optional[Message]:
        """Returns the first user message from the conversation. Returns None if none found."""
        user_messages = self.get_user_messages()
        if not user_messages:
            return None
        return user_messages[0]

    def get_user_messages(self) -> List[Message]:
        """Returns only the user messages from the conversation."""
        return [msg for msg in self.messages if msg.role == "user"]

    def get_input_metadata(self, key: str, default: Any = None) -> Any:
        """Helper method to get a specific value from input_metadata (InputMetadata fields)."""
        if self.input_metadata is None:
            return default
        return getattr(self.input_metadata, key, default)

    def get_steps(self) -> int:
        """Get number of steps from control_plane_step data."""
        return len([msg for msg in self.messages if msg.control_plane_step])

    def get_total_reward(self) -> float:
        """Get total reward from control_plane_step data."""
        messages_with_control_plane = [msg for msg in self.messages if msg.control_plane_step]
        if not messages_with_control_plane:
            return 0.0
        total = 0.0
        for msg in messages_with_control_plane:
            step = msg.control_plane_step or {}
            try:
                total += float(step.get("reward", 0.0))
            except (TypeError, ValueError):
                continue
        return total

    def get_terminated(self) -> bool:
        """Get termination status from control_plane_step data."""
        messages_with_control_plane = [msg for msg in self.messages if msg.control_plane_step]
        if not messages_with_control_plane:
            return False
        for msg in messages_with_control_plane:
            step = msg.control_plane_step or {}
            if bool(step.get("terminated", False)):
                return True
        return False

    def get_termination_reason(self) -> str:
        """Get termination reason from the final control_plane_step data."""
        # Find the last message with control_plane_step that has termination_reason
        for msg in reversed(self.messages):
            if msg.control_plane_step and msg.control_plane_step.get("termination_reason"):
                reason = msg.control_plane_step.get("termination_reason")
                return str(reason)
        return "unknown"

    def __hash__(self) -> int:
        # Use a stable hash that works across Python processes
        return self._stable_hash()

    def _stable_hash(self) -> int:
        """Generate a stable hash that works across Python processes."""
        import hashlib

        # Get the stable JSON representation
        json_str = self._stable_json()

        # Use SHA-256 for deterministic hashing across processes
        hash_obj = hashlib.sha256(json_str.encode("utf-8"))

        # Convert to a positive integer (first 8 bytes)
        hash_bytes = hash_obj.digest()[:8]
        return int.from_bytes(hash_bytes, byteorder="big")

    def _stable_json(self) -> str:
        """Generate a stable JSON string representation for hashing."""
        # Produce a canonical, key-sorted JSON across nested structures and
        # exclude volatile fields that can differ across processes
        import json
        from enum import Enum

        def canonicalize(value):
            # Recursively convert to a structure with deterministic key ordering
            if isinstance(value, dict):
                return {k: canonicalize(value[k]) for k in sorted(value.keys())}
            if isinstance(value, list):
                return [canonicalize(v) for v in value]
            if isinstance(value, Enum):
                return value.value
            return value

        # Dump to a plain Python structure first
        data = self.model_dump(
            exclude_none=True,
            exclude_defaults=True,
            by_alias=True,
            exclude={"created_at", "execution_metadata", "pid"},
        )

        # Ensure deterministic ordering for all nested dicts
        canonical_data = canonicalize(data)

        # Compact, sorted JSON string
        return json.dumps(canonical_data, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


# Original dataclass-based models for backwards compatibility
# These are deprecated and will be removed in a future version
# Use EvaluateResult and MetricResult instead
# MetricRewardOutput and RewardOutput are fully removed.


# --- Models for New Agent Evaluation Framework (V2) ---


class ResourceServerConfig(BaseModel):
    """
    Configuration for a resource server required by a task.
    """

    start_command: str = Field(
        description="The command to start the server. The string '{port}' will be replaced with a dynamically allocated free port."
    )
    health_check_url: str = Field(
        description="The URL to poll to check if the server is ready. The string '{port}' will be replaced with the allocated port."
    )


class EvaluationCriteriaModel(BaseModel):
    """
    Defines criteria for evaluating task success, often by querying the final state of a resource.
    """

    final_state_query: Optional[str] = Field(
        default=None,
        description="A query (e.g., SQL) to run on the final state of the resource.",
    )
    expected_query_result_transform: Optional[str] = Field(
        default=None,
        description="A Python lambda string (e.g., 'lambda x: x > 0') to transform and evaluate the query result to a boolean.",
    )

    # Explicit fields for ground truth data for BFCL evaluation
    ground_truth_function_calls: Optional[List[List[str]]] = Field(
        default=None, description="Ground truth function calls for BFCL evaluation."
    )
    ground_truth_comparable_state: Optional[Dict[str, Any]] = Field(
        default=None, description="Ground truth comparable state for BFCL evaluation."
    )

    # Future: Could include other complex evaluation logic or references


class TaskDefinitionModel(BaseModel):
    """
    Pydantic model for validating the structure of a V2 agent evaluation task definition file (YAML/JSON).
    """

    name: str = Field(description="Unique name for the task.")
    description: Optional[str] = Field(default=None, description="A brief description of the task.")

    resource_type: str = Field(
        description="The type of ForkableResource to use (e.g., 'SQLResource', 'PythonStateResource', 'FileSystemResource', 'DockerResource')."
    )
    base_resource_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration dictionary passed to the base resource's setup() method.",
    )

    tools_module_path: Optional[str] = Field(
        default=None,
        description="Optional Python import path to a module containing custom tool functions for this task.",
    )
    reward_function_path: str = Field(
        description="Python import path to the reward function (e.g., 'my_module.my_reward_func')."
    )

    goal_description: Optional[str] = Field(
        default=None,
        description="A human-readable description of the agent's goal for this task.",
    )
    evaluation_criteria: Optional[EvaluationCriteriaModel] = Field(
        default=None,
        description="Criteria used by the Orchestrator to determine if the primary goal was achieved.",
    )

    initial_user_prompt: Optional[str] = Field(
        default=None,
        description="The initial prompt or message to start the agent interaction. Deprecated if 'messages' field is used for multi-turn.",
    )
    messages: Optional[List[Dict[str, Any]]] = Field(  # Explicit field for initial/multi-turn messages
        default=None,
        description="A list of messages to start the conversation, can represent multiple user turns for sequential processing.",
    )

    # PoC / Task specific parameters
    poc_max_turns: int = Field(
        default=3,
        ge=1,
        description="For PoC Orchestrator, the maximum number of interaction turns.",
    )

    # Allow other custom fields to be captured if needed by specific tasks or resources
    # These will be accessible via `model_extra` if `model_config` has `extra = 'allow'`
    # Or define a specific field:
    # custom_task_params: Dict[str, Any] = Field(default_factory=dict)
    resource_server: Optional[ResourceServerConfig] = Field(
        default=None,
        description="Configuration for a background server required for the task.",
    )

    num_rollouts: int = Field(
        default=1,
        ge=1,
        description="Number of parallel rollouts to execute for this task definition.",
    )

    # Data-driven evaluation fields
    dataset_path: Optional[str] = Field(
        default=None,
        description="Path to dataset file (JSONL) containing experimental conditions for data-driven evaluation.",
    )
    num_rollouts_per_sample: int = Field(
        default=1,
        ge=1,
        description="Number of rollouts to execute per sample from the dataset.",
    )

    class Config:
        extra = "allow"  # Allow and capture extra fields not explicitly defined
        # For Pydantic v2, it's model_config = {"extra": "allow"}
        # Assuming Pydantic v1 style for now based on existing file, can update if needed.
        # If using Pydantic v1, `Config.extra = "allow"` is correct.
        # For Pydantic v2, this should be:
        # from pydantic import ConfigDict
        # model_config = ConfigDict(extra='allow')
        # For Pydantic v1, `Config.extra = "allow"` is correct.


class MCPConfigurationServerStdio(BaseModel):
    """Represents a MCP configuration server."""

    command: str  # command to run the MCP server
    args: List[str] = Field(default_factory=list)  # to pass to the command
    env: List[str] = Field(default_factory=list)  # List of environment variables to verify exist in the environment


class MCPConfigurationServerUrl(BaseModel):
    """Represents a Remote MCP configuration server."""

    url: str  # url to the MCP server
    authorization: Optional[str] = None


class MCPMultiClientConfiguration(BaseModel):
    """Represents a MCP configuration."""

    mcpServers: Dict[str, Union[MCPConfigurationServerStdio, MCPConfigurationServerUrl]]
