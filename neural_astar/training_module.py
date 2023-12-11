#!/usr/bin/env python
import sys
import os
import torch
import lightning as pl


class NeuralAstarTrainingModule(pl.LightningModule):
    """
    Training module for Neural A* model.

    Attributes:
        neural_astar_ (NeuralAstar): Neural A* model.
        learning_rate_ (float): Learning rate for optimization.
    """

    def __init__(self, neural_astar, learning_rate=0.01) -> None:
        """
        Initialize the NeuralAstarTrainingModule.

        Args:
            neural_astar (NeuralAstar): Neural A* model.
            learning_rate (float): Learning rate for optimization.
        """
        super().__init__()
        self.neural_astar_ = neural_astar
        self.learning_rate_ = learning_rate

    def forward(self, matrix_batch, start_batch, goal_batch):
        """
        Forward pass through the NeuralAstar model.

        Args:
            matrix_batch: Batch of matrices.
            start_batch: Batch of start points.
            goal_batch: Batch of goal points.

        Returns:
            torch.Tensor: Batch of searched nodes.
        """
        matrix_batch = matrix_batch.unsqueeze(1).float()
        start_batch = start_batch.unsqueeze(1)
        goal_batch = goal_batch.unsqueeze(1)
        _, searched_nodes_batch = self.neural_astar_(
            matrix_batch, start_batch, goal_batch
        )
        return searched_nodes_batch

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer for training.

        Returns:
            torch.optim.Optimizer: Optimizer for training.
        """
        return torch.optim.RMSprop(
            self.neural_astar_.parameters(), self.learning_rate_
        )  # Learning rate

    def training_step(self, train_batch, batch_idx):
        """
        Training step for the NeuralAstar model.

        Args:
            train_batch: Batch of training data.
            batch_idx: Index of the batch.

        Returns:
            torch.Tensor: Loss for the training step.
        """
        matrix_batch, start_batch, goal_batch, trajectory_batch = train_batch
        trajectory_batch = trajectory_batch.unsqueeze(1)
        visited_nodes_batch = self(matrix_batch, start_batch, goal_batch)
        loss = torch.nn.MSELoss()(trajectory_batch, visited_nodes_batch)
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss
