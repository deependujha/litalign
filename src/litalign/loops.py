from typing import Any, Dict

from torch.utils.data import DataLoader

from litalign.trainer import BaseTrainer


class Loop:
    def __init__(self, trainer: BaseTrainer, dataloader: DataLoader):
        self.trainer = trainer
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        self.epoch = 0
        self.step_count = 0

    def step(self) -> Dict[str, Any]:
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.epoch += 1
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)

        # Step through RLHF phases
        responses = self.trainer.rollout(batch)
        rewards = self.trainer.evaluate(batch, responses)
        loss = self.trainer.optimize(batch, responses, rewards)

        self.step_count += 1
        return {
            "epoch": self.epoch,
            "step": self.step_count,
            "loss": loss,
        }
