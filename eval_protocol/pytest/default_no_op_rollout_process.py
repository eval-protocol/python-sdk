from typing import List

from eval_protocol.models import EvaluationRow
from eval_protocol.pytest.types import RolloutProcessorConfig


def default_no_op_rollout_processor(rows: List[EvaluationRow], config: RolloutProcessorConfig) -> List[EvaluationRow]:
    """
    Simply passes input dataset through to the test function. This can be useful
    if you want to run the rollout yourself.
    """
    return rows
