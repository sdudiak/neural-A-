import math
import torch
import torch.nn as nn
from heuristics import differentiable_euclidian

import lightning as pl


class DifferentiableAstar(pl.LightningModule):
    def __init__(self, max_iterations=50000, costmap_weight = 1):
        super().__init__()

        neighbour_filter = torch.ones(1, 1, 3, 3)
        neighbour_filter[0, 0, 1, 1] = 0

        self.neighbour_filter = nn.Parameter(neighbour_filter, requires_grad=False)
        self.heuristic = differentiable_euclidian
        self.max_iterations_ = max_iterations
        self.costmap_weight_ = costmap_weight

    def _select_neighbours(
        self, node: torch.tensor, neighbour_filter: torch.tensor
    ) -> torch.tensor:
        
        node = node.unsqueeze(0)
        batch_size = node.shape[1]
        neighbours = nn.functional.conv2d(
            node, neighbour_filter, padding=1, groups=batch_size
        ).squeeze()
        neighbours = neighbours.squeeze(0)
        return neighbours

    def _backtrack_path(
        self,
        start_batch: torch.tensor,
        goal_batch: torch.tensor,
        parents_batch: torch.tensor,
        t: int,
    ) -> torch.tensor:
        parents_batch = parents_batch.long()
        goal_batch = goal_batch.long()
        start_batch = start_batch.long()
        paths_batch = goal_batch.long()
        batch_size = len(parents_batch)
        current_node_idx = (parents_batch * goal_batch.view(batch_size, -1)).sum(-1)
        for _ in range(t):
            paths_batch.view(batch_size, -1)[range(batch_size), current_node_idx] = 1
            current_node_idx = parents_batch[range(batch_size), current_node_idx]
        return paths_batch

    def _straight_through_softmax_(self, val: torch.tensor) -> torch.tensor:
        val_ = val.reshape(val.shape[0], -1)
        y = val_ / (val_.sum(dim=-1, keepdim=True))
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y)
        y_hard[range(len(y_hard)), ind] = 1
        y_hard = y_hard.reshape_as(val)
        y = y.reshape_as(val)
        return (y_hard - y).detach() + y

    def forward(
        self,
        matrix_batch: torch.tensor,
        start_batch: torch.tensor,
        goal_batch: torch.tensor,
        costmap_batch: torch.tensor,

    ):
        costmap_batch = costmap_batch.squeeze(1)
        start_batch = start_batch.squeeze(1)
        goal_batch = goal_batch.squeeze(1)
        matrix_batch = matrix_batch.squeeze(1)

        costmap_batch = costmap_batch * self.costmap_weight_

        batch_size = start_batch.shape[0]
        neighbour_filter = torch.repeat_interleave(self.neighbour_filter, batch_size, 0)

        open_list = start_batch

        closed_list = torch.zeros_like(start_batch)
        h = self.heuristic(goal_batch) + costmap_batch
        g = torch.zeros_like(start_batch)

        parents_batch = (
            torch.ones_like(start_batch).reshape(batch_size, -1)
            * goal_batch.reshape(batch_size, -1).max(-1, keepdim=True)[-1]
        )

        for t in range(self.max_iterations_):
            # Get total cost matrix
            f = g + h
            # Tau constant is selected as a square root of the matrix width
            f_exp = torch.exp(-1 * f / math.sqrt(costmap_batch.shape[-1]))

            f_exp = f_exp * open_list

            current_node = self._straight_through_softmax_(f_exp)

            # break if arriving at the goal
            dist_to_goal = (current_node * goal_batch).sum((1, 2), keepdim=True)
            is_unsolved = (dist_to_goal == 0).float()

            closed_list = closed_list + current_node
            closed_list = torch.clamp(closed_list, 0, 1)  # Normalize data to <0,1> range
            open_list = (
                open_list - is_unsolved * current_node
            )  # remove selected node from open maps
            open_list = torch.clamp(open_list, 0, 1)

            # open neighboring nodes, add them to the openlist if they satisfy certain requirements
            neighbor_nodes = self._select_neighbours(
                current_node, neighbour_filter
            )
            neighbor_nodes = neighbor_nodes * matrix_batch

            g2 = self._select_neighbours(
                (g + costmap_batch) * current_node, neighbour_filter
            )

            idx = (1 - open_list) * (1 - closed_list)
            idx = idx * neighbor_nodes
            idx = idx.detach()
            g = g2 * idx + g * (1 - idx)
            g = g.detach()

            # update open maps
            open_list = torch.clamp(open_list + idx, 0, 1)
            open_list = open_list.detach()

            # for backtracking

            idx = idx.reshape(batch_size, -1)

            snm = current_node.reshape(batch_size, -1)
            new_parents_batch = snm.max(-1, keepdim=True)[1]
            parents_batch = new_parents_batch * idx + parents_batch * (1 - idx)

            if torch.all(is_unsolved.flatten() == 0):
                break

        # backtracking
        paths_batch = self._backtrack_path(start_batch, goal_batch, parents_batch, t)

        return paths_batch.unsqueeze(1), closed_list.unsqueeze(1), 
