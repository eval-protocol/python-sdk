from abc import ABC, abstractmethod
from typing import Any

from eval_protocol.pytest.types import TestFunction


class Trainer(ABC):
    def __init__(self, test_fn: TestFunction):
        self.test_fn = test_fn

    @abstractmethod
    def train(self, *args: Any, **kwargs: Any) -> Any:
        """Run training and return the optimized model/program."""
        ...

    @abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        """Evaluate the optimized model/program."""
        ...
