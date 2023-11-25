from typing import Any
import lightning as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from astar_refactored import Astar
import heuristics
from neural_astar import NeuralAstar
import torch


class NeuralAstarTrainingModule(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.astar = NeuralAstar()
        self.requires_grad_ = True

    def forward(self, matrix_batch, start_batch, goal_batch) -> Any:
        return self.astar(matrix_batch, start_batch, goal_batch)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        torch.set_grad_enabled(True)
        print("CONFIGURING OPTIMIZERS")
        return torch.optim.RMSprop(self.astar.parameters(), 0.01)  # Learning rate

    def training_step(self, train_batch, batch_idx):
        torch.set_grad_enabled(True)
        matrix_batch, start_batch, goal_batch, trajectory_batch = train_batch
        _, visited_nodes_batch = self.forward(matrix_batch, start_batch, goal_batch)
        visited_nodes_batch = visited_nodes_batch.unsqueeze(0)
        print("3")
        loss = torch.nn.L1Loss()(visited_nodes_batch, trajectory_batch)
        print("4")
        self.log("metrics/train_loss", loss)
        loss.backward()
        print("5")

        return loss

    def validation_step(self, *args: Any, **kwargs: Any):
        torch.set_grad_enabled(True)

        print("VALIDATION")
        pass
