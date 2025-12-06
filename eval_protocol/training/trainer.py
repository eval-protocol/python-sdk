from abc import ABC, abstractmethod

from eval_protocol.pytest.types import TestFunction


class Trainer(ABC):
    def __init__(self, test_fn: TestFunction):
        self.test_fn = test_fn

    @abstractmethod
    def train(self, *args, **kwargs): ...

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        # evaluation logic possibly can be shared since it's EP. TBD
        ...
