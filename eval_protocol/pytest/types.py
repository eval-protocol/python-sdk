"""
Parameter types
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional

from eval_protocol.dataset_logger import default_logger
from eval_protocol.dataset_logger.dataset_logger import DatasetLogger

from ..models import EvaluationRow, Message

ModelParam = str  # gpt-4o, gpt-4o-mini, accounts/fireworks/models/llama-3.1-8b-instruct
DatasetPathParam = str
RolloutInputParam = Dict[str, Any]
InputMessagesParam = List[Message]
EvaluationInputParam = Dict[str, Any]

Dataset = List[EvaluationRow]

EvaluationTestMode = Literal["batch", "pointwise"]
"""
"batch": (default) expects test function to handle full dataset.
"pointwise": applies test function to each row.

How to choose between "batch" and "pointwise":
If your evaluation requires the rollout of all rows to be passed into your eval compute the score, use "batch".
If your evaluation can be computed pointwise, use "pointwise" as EP can pipeline the rollouts and evals to be faster.
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
    model: ModelParam
    input_params: RolloutInputParam  # optional input parameters for inference
    mcp_config_path: str
    server_script_path: Optional[str] = (
        None  # TODO: change from server_script_path to mcp_config_path for agent rollout processor
    )
    max_concurrent_rollouts: int = 8  # maximum number of concurrent rollouts
    steps: int = 30  # max number of rollout steps
    logger: DatasetLogger = default_logger  # logger to use during rollout for mid-rollout logs


RolloutProcessor = Callable[[List[EvaluationRow], RolloutProcessorConfig], List[EvaluationRow]]
