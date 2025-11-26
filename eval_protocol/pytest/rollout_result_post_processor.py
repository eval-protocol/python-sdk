"""
Rollout result post-processing plugin for quality checks.

This module provides an abstract base class for post-processing rollout results
to guard response quality. Post-processors can validate results and raise
ResponseQualityError if quality checks fail.
"""

from abc import ABC, abstractmethod

from eval_protocol.models import EvaluationRow


class RolloutResultPostProcessor(ABC):
    """
    Abstract base class for rollout result post-processing plugins.

    Post-processors validate rollout results and can raise ResponseQualityError
    if quality checks fail. This allows for customizable quality guards that
    can be overridden by users.
    """

    @abstractmethod
    def process(self, result: EvaluationRow) -> None:
        """
        Process and validate a rollout result.

        This method should perform quality checks on the result. If quality
        checks fail, it should raise ResponseQualityError with an appropriate
        message.

        Args:
            result: The EvaluationRow result from the rollout

        Raises:
            ResponseQualityError: If quality checks fail
        """
        pass


class NoOpRolloutResultPostProcessor(RolloutResultPostProcessor):
    """
    Default no-op implementation of RolloutResultPostProcessor.

    This implementation does not perform any quality checks and always passes.
    Use this as a default when no post-processing is needed.
    """

    def process(self, result: EvaluationRow) -> None:
        """
        No-op implementation that does not perform any quality checks.

        Args:
            result: The EvaluationRow result from the rollout
        """
        pass

