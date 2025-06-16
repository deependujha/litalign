from abc import ABC, abstractmethod
from typing import Any, List


class BaseTrainer(ABC):
    def __init__(self) -> None:
        """Base class for trainers in the LitAlign framework.

        This class defines the interface for all trainers, ensuring that
        any derived trainer implements the necessary methods for training,
        evaluation, and optimization.
        """

    @abstractmethod
    def rollout(self, batch: List[Any]) -> None:
        pass

    @abstractmethod
    def evaluate(self, batch: List[Any], responses: Any) -> None:
        pass

    @abstractmethod
    def optimize(self, batch: List[Any], responses: Any, rewards: Any) -> None:
        pass
