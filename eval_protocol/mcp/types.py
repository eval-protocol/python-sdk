from dataclasses import dataclass
from typing import Any, Dict, List, Optional


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
