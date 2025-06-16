from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    def __init__(self, *args, **kwargs):
        """Base class for trainers in the LitAlign framework.

        This class defines the interface for all trainers, ensuring that
        any derived trainer implements the necessary methods for training,
        evaluation, and optimization.
        """

    @abstractmethod
    def rollout(self, batch):
        pass

    @abstractmethod
    def evaluate(self, batch, responses):
        pass

    @abstractmethod
    def optimize(self, batch, responses, rewards):
        pass
