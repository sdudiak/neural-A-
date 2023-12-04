#!/usr/bin/env python
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__)))

import torch
import lightning as pl

from classic_astar import Astar
from differentiable_astar import DifferentiableAstar
from encoders_refactored import AstarEncoder


class NeuralAstar(pl.LightningModule):
    def __init__(
        self,
        classic_astar: Astar,
        differentiable_astar: DifferentiableAstar,
        encoder: AstarEncoder,
        use_start_goal_data: bool,
    ) -> None:
        super().__init__()
        self.classic_astar_ = classic_astar
        self.differentiable_astar_ = differentiable_astar
        self.encoder_ = encoder
        self.encoder_ = self.encoder_.eval()
        self.use_start_goal_data_ = use_start_goal_data

    def _get_cost_batch(self, matrix_batch, start_goal_batch=None):
        if start_goal_batch is not None:
            encoder_input = torch.cat((matrix_batch, start_goal_batch), dim=1)
            return self.encoder_(encoder_input)
        else:
            return self.encoder_(matrix_batch)

    def forward(self, matrix_batch, start_batch, goal_batch, use_classic_astar=False):
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
