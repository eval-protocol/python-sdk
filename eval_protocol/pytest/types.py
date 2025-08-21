"""
Parameter types
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional

from eval_protocol.dataset_logger import default_logger
from eval_protocol.dataset_logger.dataset_logger import DatasetLogger

from ..models import CompletionParams, EvaluationRow, Message
from .exception_config import ExceptionHandlerConfig

ModelParam = str  # gpt-4o, gpt-4o-mini, accounts/fireworks/models/llama-3.1-8b-instruct
DatasetPathParam = str
InputMessagesParam = List[Message]
InputRowsParam = List[EvaluationRow]
EvaluationInputParam = Dict[str, Any]
RolloutProcessorInputParam = Dict[str, Any]

Dataset = List[EvaluationRow]

EvaluationTestMode = Literal["pointwise", "groupwise", "all"]
"""
"pointwise": (default) applies test function to each row (rollout result).
"groupwise": applies test function to a group of rollout results from the same original row (for use cases such as dpo/grpo).
"all": applies test function to the whole dataset.
"""

"""
Test function types
"""
TestFunction = Callable[..., Dataset]

"""
Rollout processor types
"""


@dataclass
class RolloutProcessorConfig:
    completion_params: CompletionParams  # input parameters for inference
    mcp_config_path: str
    server_script_path: Optional[str] = (
        None  # TODO: change from server_script_path to mcp_config_path for agent rollout processor
    )
    max_concurrent_rollouts: int = 8  # maximum number of concurrent rollouts
    steps: int = 30  # max number of rollout steps
    logger: DatasetLogger = default_logger  # logger to use during rollout for mid-rollout logs
    kwargs: Dict[str, Any] = field(default_factory=dict)  # any additional kwargs to pass to the rollout processor
    exception_handler_config: Optional[ExceptionHandlerConfig] = (
        None  # configuration for exception handling with backoff
    )
