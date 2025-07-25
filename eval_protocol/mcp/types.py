from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TerminationReason(str, Enum):
    """Enum representing different reasons why a trajectory terminated.

    MAX_STEPS: Trajectory ends because we hit the step limit
    CONTROL_PLANE_SIGNAL: Trajectory ends because the control plane signals termination (e.g. env goal reached or failure condition)
    USER_STOP: Trajectory ends because the simulated user signals to stop
    """

    MAX_STEPS = "max_steps"
    CONTROL_PLANE_SIGNAL = "control_plane_signal"
    USER_STOP = "user_stop"


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
    _exit_stack: Optional[Any] = None
    _mcp_session: Optional[Any] = None


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
    llm_usage_summary: Dict[str, int] = field(default_factory=dict)


@dataclass
class LLMUsageStats:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
