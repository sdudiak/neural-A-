#!/usr/bin/env python
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__)))

from typing import Any
import lightning as pl
import torch


class NeuralAstarTrainingModule(pl.LightningModule):
    def __init__(self, neural_astar, learning_rate=0.01) -> None:
        super().__init__()
        self.neural_astar_ = neural_astar
        self.learning_rate_ = learning_rate
        self

    def forward(self, matrix_batch, start_batch, goal_batch):
        matrix_batch = matrix_batch.unsqueeze(1).float()
        start_batch = start_batch.unsqueeze(1)
        goal_batch = goal_batch.unsqueeze(1)
        _, searched_nodes_batch = self.neural_astar_(
            matrix_batch, start_batch, goal_batch
        )
        return searched_nodes_batch

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.RMSprop(
            self.neural_astar_.parameters(), self.learning_rate_
        )  # Learning rate

    def training_step(self, train_batch, batch_idx):
        matrix_batch, start_batch, goal_batch, trajectory_batch = train_batch
        visited_nodes_batch = self(matrix_batch, start_batch, goal_batch)
        visited_nodes_batch = visited_nodes_batch.squeeze(1)
        loss = torch.nn.MSELoss()(trajectory_batch, visited_nodes_batch)
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss
