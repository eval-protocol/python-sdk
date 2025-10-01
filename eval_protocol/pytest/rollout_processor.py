import asyncio
from abc import ABC, abstractmethod

from eval_protocol.models import EvaluationRow
from eval_protocol.pytest.types import RolloutProcessorConfig


class RolloutProcessor(ABC):
    """
    Abstract base class for all rollout processor strategies.
    """

    def setup(self) -> None:
        """Setup resources. Override in subclasses if setup is needed. Executed once per invocation."""
        pass

    @abstractmethod
    def __call__(self, rows: list[EvaluationRow], config: RolloutProcessorConfig) -> list[asyncio.Task[EvaluationRow]]:
        """Process evaluation rows and return async tasks. Must be implemented by subclasses."""
        pass

    def cleanup(self) -> None:
        """Cleanup resources. Override in subclasses if cleanup is needed."""
        pass
