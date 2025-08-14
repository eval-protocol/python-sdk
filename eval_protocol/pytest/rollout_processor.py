import asyncio
from abc import ABC, abstractmethod
from typing import List

from eval_protocol.models import EvaluationRow
from eval_protocol.pytest.types import RolloutProcessorConfig


class RolloutProcessor(ABC):
    """
    Abstract base class for all rollout processor strategies.
    """

    @abstractmethod
    def __call__(self, rows: List[EvaluationRow], config: RolloutProcessorConfig) -> List[asyncio.Task[EvaluationRow]]:
        """Process evaluation rows and return async tasks. Must be implemented by subclasses."""
        pass

    def cleanup(self) -> None:
        """Cleanup resources. Override in subclasses if cleanup is needed."""
        pass
