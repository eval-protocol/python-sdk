import asyncio
from typing import List

from eval_protocol.models import EvaluationRow
from eval_protocol.pytest.types import RolloutProcessorConfig


class BaseRolloutProcessor:
    """
    Base rollout processor - minimal implementation that all others inherit from.

    This is the Strategy pattern base class. It provides:
    1. __call__(rows, config) -> tasks (the main interface)
    2. cleanup() -> None (resource cleanup)

    All other processors inherit from this and override as needed.
    """

    def __call__(self, rows: List[EvaluationRow], config: RolloutProcessorConfig) -> List[asyncio.Task[EvaluationRow]]:
        """Process rows by returning them unchanged (no-op implementation)."""

        async def return_row(row: EvaluationRow) -> EvaluationRow:
            return row

        # Create tasks that immediately return the rows (no-op)
        tasks = [asyncio.create_task(return_row(row)) for row in rows]
        return tasks

    def cleanup(self) -> None:
        """No-op cleanup - override in subclasses if needed."""
        pass
