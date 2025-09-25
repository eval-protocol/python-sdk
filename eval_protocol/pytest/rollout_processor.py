import asyncio
from abc import ABC, abstractmethod

from eval_protocol.models import EvaluationRow
from eval_protocol.pytest.types import RolloutProcessorConfig


class RolloutProcessor(ABC):
    """
    Abstract base class for all rollout processor strategies.
    """

    supports_pipelining: bool = (
        True  # Whether this processor supports pipelined evaluation (evaluate rows as rollouts complete)
    )

    @abstractmethod
    def __call__(self, rows: list[EvaluationRow], config: RolloutProcessorConfig) -> list[asyncio.Task[EvaluationRow]]:
        """Process evaluation rows and return async tasks. Must be implemented by subclasses."""
        pass

    def postprocess(self, finished_rollout_rows: list[EvaluationRow]) -> list[EvaluationRow]:
        """Post-process rollout results to produce evaluation inputs. Only available for processors that return False from supports_pipelining."""
        return finished_rollout_rows

    def cleanup(self) -> None:
        """Cleanup resources. Override in subclasses if cleanup is needed."""
        pass
