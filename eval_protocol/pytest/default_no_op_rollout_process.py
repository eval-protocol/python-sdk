from typing import AsyncIterator, List

from eval_protocol.models import EvaluationRow
from eval_protocol.pytest.types import RolloutProcessorConfig


async def default_no_op_rollout_processor(
    rows: List[EvaluationRow], config: RolloutProcessorConfig
) -> AsyncIterator[EvaluationRow]:
    """
    Simply passes input dataset through to the test function. This can be useful
    if you want to run the rollout yourself.
    """
    for row in rows:
        yield row
