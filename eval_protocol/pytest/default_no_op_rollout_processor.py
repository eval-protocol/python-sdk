import asyncio
from typing import List

from eval_protocol.models import EvaluationRow
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig


class NoOpRolloutProcessor(RolloutProcessor):
    """
    No-op rollout processor that passes input dataset through unchanged.

    Simply returns the input rows as completed tasks. This is useful for testing
    or when you want to handle rollout processing manually.
    """

    def __call__(self, rows: List[EvaluationRow], config: RolloutProcessorConfig) -> List[asyncio.Task[EvaluationRow]]:
        """Process rows by returning them unchanged (no-op implementation)."""

        async def return_row(row: EvaluationRow) -> EvaluationRow:
            return row

        # Create tasks that immediately return the rows (no-op)
        tasks = [asyncio.create_task(return_row(row)) for row in rows]
        return tasks

    # Inherits cleanup() from RolloutProcessor - no override needed
