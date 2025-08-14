import asyncio
from typing import List

from eval_protocol.models import EvaluationRow
from eval_protocol.pytest.types import RolloutProcessorConfig


def default_no_op_rollout_processor(
    rows: List[EvaluationRow], config: RolloutProcessorConfig
) -> List[asyncio.Task[EvaluationRow]]:
    """
    Simply passes input dataset through to the test function. This can be useful
    if you want to run the rollout yourself.
    """

    async def return_row(row: EvaluationRow) -> EvaluationRow:
        return row

    # Create tasks that immediately return the rows (no-op)
    tasks = [asyncio.create_task(return_row(row)) for row in rows]
    return tasks
