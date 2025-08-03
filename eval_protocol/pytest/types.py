"""
Parameter types
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal

from ..models import EvaluationRow, Message

ModelParam = str  # gpt-4o, gpt-4o-mini, accounts/fireworks/models/llama-3.1-8b-instruct
DatasetPathParam = str
InputParam = Dict[str, Any]
InputMessagesParam = List[Message]

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
    input_params: InputParam  # optional input parameters for inference
    mcp_config_path: str  # for agent rollout processor
    initial_messages: list[Message]  # for agent rollout processor


RolloutProcessor = Callable[[EvaluationRow, RolloutProcessorConfig], List[EvaluationRow]]
