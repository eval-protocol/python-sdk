"""
Parameter types
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from eval_protocol.dataset_logger import default_logger
from eval_protocol.dataset_logger.dataset_logger import DatasetLogger

from ..models import CompletionParams, EvaluationRow, Message
from .exception_config import ExceptionHandlerConfig

ModelParam = str  # gpt-4o, gpt-4o-mini, accounts/fireworks/models/llama-3.1-8b-instruct
DatasetPathParam = str
InputMessagesParam = list[Message]
EvaluationInputParam = dict[str, Any]  # pyright: ignore[reportExplicitAny]
RolloutProcessorInputParam = dict[str, Any]  # pyright: ignore[reportExplicitAny]

Dataset = list[EvaluationRow]

EvaluationTestMode = Literal["pointwise", "groupwise", "all"]
"""
"pointwise": (default) applies test function to each row (rollout result).
"groupwise": applies test function to a group of rollout results from the same original row (for use cases such as dpo/grpo).
"all": applies test function to the whole dataset.
"""

ServerMode = Literal["per_run", "shared"]
"""
"per_run": start a new MCP server for each eval run / training step, only reuse the same server only for retries within that run.
"shared": start a single MCP server the first time it's needed, then reuse that same server across multiple eval runs / training steps.
"""

"""
Test function types
"""
# Type variable for the decorated function
from collections.abc import Awaitable

# TestFunction can be either:
# 1. an async/sync function that accepts EvaluationRow and returns EvaluationRow
# 2. an async/sync function that accepts list[EvaluationRow] and returns list[EvaluationRow]
TestFunction = (
    Callable[[], EvaluationRow]
    | Callable[[], Awaitable[EvaluationRow]]
    | Callable[[], Dataset]
    | Callable[[], Awaitable[Dataset]]
    | Callable[[EvaluationRow], EvaluationRow]
    | Callable[[EvaluationRow], Awaitable[EvaluationRow]]
    | Callable[[list[EvaluationRow]], list[EvaluationRow]]
    | Callable[[list[EvaluationRow]], Awaitable[list[EvaluationRow]]]
    | Callable[[Dataset], Dataset]
    | Callable[[Dataset], Awaitable[Dataset]]
)


"""
Rollout processor types
"""


@dataclass
class RolloutProcessorConfig:
    completion_params: CompletionParams  # input parameters for inference
    mcp_config_path: str
    semaphore: asyncio.Semaphore  # shared semaphore for unified concurrency control
    server_script_path: str | None = (
        None  # TODO: change from server_script_path to mcp_config_path for agent rollout processor
    )
    steps: int = 30  # max number of rollout steps
    logger: DatasetLogger = default_logger  # logger to use during rollout for mid-rollout logs
    kwargs: dict[str, Any] = field(  # pyright: ignore[reportExplicitAny]
        default_factory=dict
    )  # any additional kwargs to pass to the rollout processor
    exception_handler_config: ExceptionHandlerConfig | None = None  # configuration for exception handling with backoff
