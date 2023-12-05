#!/usr/bin/env python
import sys
import os
import torch
import lightning as pl
from classic_astar import Astar
from differentiable_astar import DifferentiableAstar
from encoders import AstarEncoder


class NeuralAstar(pl.LightningModule):
    """
    Neural network model incorporating both classic A* and differentiable A*.

    Attributes:
        classic_astar_ (Astar): Classic A* algorithm.
        differentiable_astar_ (DifferentiableAstar): Differentiable A* algorithm.
        encoder_ (AstarEncoder): A* encoder.
        use_start_goal_data_ (bool): Flag indicating whether to use start and goal data.
    """

    def __init__(
        self,
        classic_astar: Astar,
        differentiable_astar: DifferentiableAstar,
        encoder: AstarEncoder,
        use_start_goal_data: bool,
    ) -> None:
        """
        Initialize the NeuralAstar model.

        Args:
            classic_astar (Astar): Classic A* algorithm.
            differentiable_astar (DifferentiableAstar): Differentiable A* algorithm.
            encoder (AstarEncoder): A* encoder.
            use_start_goal_data (bool): Flag indicating whether to use start and goal data.
        """
        super().__init__()
        self.classic_astar_ = classic_astar
        self.differentiable_astar_ = differentiable_astar
        self.encoder_ = encoder
        self.encoder_ = self.encoder_.eval()
        self.use_start_goal_data_ = use_start_goal_data

    def _get_cost_batch(self, matrix_batch, start_goal_batch=None):
        """
        Get the cost batch using the encoder.

        Args:
            matrix_batch: Batch of matrices.
            start_goal_batch: Batch of start and goal points (optional).

        Returns:
            torch.tensor: Cost batch.
        """
        if start_goal_batch is not None:
            encoder_input = torch.cat((matrix_batch, start_goal_batch), dim=1)
            return self.encoder_(encoder_input)
        else:
            return self.encoder_(matrix_batch)

    def forward(self, matrix_batch, start_batch, goal_batch, use_classic_astar=False):
        """
        Forward pass through the NeuralAstar model.

        Args:
            matrix_batch: Batch of matrices.
            start_batch: Batch of start points.
            goal_batch: Batch of goal points.
            use_classic_astar (bool): Flag indicating whether to use classic A*.

        Returns:
            tuple: Tuple containing paths_batch and visited_nodes_batch.
        """
        if self.use_start_goal_data_:
            costmap_batch = self._get_cost_batch(matrix_batch, start_batch + goal_batch)
        else:
            costmap_batch = self._get_cost_batch(matrix_batch)

        if use_classic_astar:
            return self.classic_astar_.forward(
                matrix_batch, start_batch, goal_batch, costmap_batch
            )
        else:
            return self.differentiable_astar_.forward(
                1 - matrix_batch, start_batch, goal_batch, costmap_batch
            )  # Invert matrix_batch
