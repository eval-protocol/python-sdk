from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from mcp.client.session import ClientSession


class TerminationReason(str, Enum):
    """Enum representing different reasons why a trajectory terminated.

    MAX_STEPS: Trajectory ends because we hit the step limit
    CONTROL_PLANE_SIGNAL: Trajectory ends because the control plane signals termination (e.g. env goal reached or failure condition)
    USER_STOP: Trajectory ends because the simulated user signals to stop
    SKIPPABLE_ERROR: Trajectory ends because of an error, this trajectory can be discarded/skipped during postprocessing/evaluation.
    NON_SKIPPABLE_ERROR: Trajectory is interrupted due to some non-skippable error (e.g. policy returns unexpected response and we need to terminate the rollout).
    STOP: Trajectory ends by the policy (mapped to llm response stop reason "stop")
    LENGTH: Trajectory ends by the policy (mapped to llm response stop reason "length")
    TOOL_CALLS: Trajectory ends by the policy with a hanging tool call response (mapped to llm response stop reason "tool_calls")
    """

    MAX_STEPS = "max_steps"
    CONTROL_PLANE_SIGNAL = "control_plane_signal"
    USER_STOP = "user_stop"
    SKIPPABLE_ERROR = "skippable_error"
    NON_SKIPPABLE_ERROR = "non_skippable_error"
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"

    @classmethod
    def from_str(cls, value: str) -> "TerminationReason":
        if value == "stop":
            return cls.STOP
        elif value == "length":
            return cls.LENGTH
        elif value == "max_steps":
            return cls.MAX_STEPS
        elif value == "control_plane_signal":
            return cls.CONTROL_PLANE_SIGNAL
        elif value == "user_stop":
            return cls.USER_STOP
        elif value == "skippable_error":
            return cls.SKIPPABLE_ERROR
        elif value == "non_skippable_error":
            return cls.NON_SKIPPABLE_ERROR
        elif value == "tool_calls":
            return cls.TOOL_CALLS
        else:
            raise ValueError(f"Invalid termination reason: {value}")


@dataclass
class MCPToolCall:
    """Represents a tool call to be executed via MCP."""

    tool_name: str
    arguments: Dict[str, Any]
    tool_call_id: Optional[str] = None


@dataclass
class DatasetRow:
    """Represents a row from the dataset JSONL."""

    id: str
    seed: int
    system_prompt: str
    user_prompt_template: str
    environment_context: Dict[str, Any]
    user_simulation: Optional[Dict[str, Any]] = None


@dataclass
class MCPSession:
    """Represents a single MCP session with an environment."""

    session_id: str
    base_url: str
    seed: Optional[int]
    model_id: str
    dataset_row: Optional[DatasetRow] = None
    terminated: bool = False
    last_observation: Any = None

    # Persistent MCP connection components
    _exit_stack: Optional[AsyncExitStack] = None
    _mcp_session: Optional[ClientSession] = None


@dataclass
class Trajectory:
    """Represents a complete rollout trajectory."""

    session: MCPSession
    observations: List[Any]
    actions: List[str]
    rewards: List[float]
    terminated: bool
    total_reward: float
    steps: int
    duration: float
    control_plane_steps: List[Dict[str, Any]]
    control_plane_summary: Dict[str, Any]
    termination_reason: str
    conversation_history: List[Dict[str, Any]]
    usage: Dict[str, int] = field(default_factory=dict)
