#!/usr/bin/env python
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__)))

import numpy as np
import lightning as pl
import torch

from collections import namedtuple

from custom_types import onehottensor2node2d, nodelist2otensor
from custom_exceptions import PathNotFoundException

Node2d = namedtuple("Node2d", ["x", "y"])

NodeAstar = namedtuple("NodeAstar", ["node", "parent", "h", "g", "cost"])


class Astar(pl.LightningDataModule):
    def __init__(
        self,
        heuristic: callable,
        costmap_weight: int = 1,
        max_iterations: int = 50000000,
        heuristic_weight: int = 5,
    ) -> None:
        super().__init__()
        self.costmap_weight_ = costmap_weight
        self.heuristic_ = heuristic
        self.max_iterations_ = max_iterations
        self.heuristic_weight = heuristic_weight

    def _select_neighbours(self, node: Node2d, max_x, max_y):
        neighbour_candidates = [
            Node2d(node.x + 1, node.y + 1),
            Node2d(node.x + 1, node.y),
            Node2d(node.x + 1, node.y - 1),
            Node2d(node.x, node.y - 1),
            Node2d(node.x - 1, node.y - 1),
            Node2d(node.x - 1, node.y),
            Node2d(node.x - 1, node.y + 1),
            Node2d(node.x, node.y + 1),
        ]
        neighbour_candidates = [
            n for n in neighbour_candidates if 0 <= n.x < max_x and 0 <= n.y < max_y
        ]
        return neighbour_candidates

    def _backtrack_path(self, closed_list):
        optimal_path = list()
        current_node = closed_list[-1]  # Start from the goal node
        while current_node.parent is not None:
            optimal_path.insert(0, current_node.node)
            current_node = [
                node for node in closed_list if node.node == current_node.parent
            ][0]
        return optimal_path

    def _get_searched_nodes(self, closed_list):
        search_history = list()
        for astar_node in closed_list:
            search_history.append(astar_node.node)
        return search_history

    def forward(self, matrix_batch, start_batch, goal_batch, costmap_batch=None):
        if costmap_batch is None:
            costmap_batch = torch.zeros_like(matrix_batch)
        matrix_batch = matrix_batch.float()
        paths_batch = torch.empty_like(matrix_batch)
        visited_nodes_batch = torch.empty_like(matrix_batch)
        for batch_idx, (matrix, start, goal, costmap) in enumerate(
            zip(matrix_batch, start_batch, goal_batch, costmap_batch)
        ):
            np_costmap = costmap[0].detach().numpy()
            np_matrix = matrix[0].detach().numpy()
            start_node = onehottensor2node2d(start)
            goal_node = onehottensor2node2d(goal)
            path, searched_nodes = self._run_astar(
                np_matrix, start_node, goal_node, np_costmap
            )
            path_tensor = nodelist2otensor(np_matrix.shape[0], path)
            searched_nodes_tensor = nodelist2otensor(np_matrix.shape[0], searched_nodes)
            paths_batch[batch_idx, 0] = path_tensor
            visited_nodes_batch[batch_idx, 0] = searched_nodes_tensor

        return paths_batch, visited_nodes_batch

    def _run_astar(self, matrix, start, goal, costmap=None) -> None:
        if costmap is None:
            costmap = np.zeros_like(matrix)
        current_node = None
        open_list = list()
        closed_list = list()
        open_list.append(NodeAstar(start, None, 0, 0, 0))
        iterations = 0

        while open_list:
            if iterations > self.max_iterations_:
                print("Max iterations achieved")
                break
            current_node = min(open_list, key=lambda node: node.cost)
            open_list.remove(current_node)
            closed_list.append(current_node)
            if current_node.node == goal:
                path = self._backtrack_path(closed_list)
                searched_nodes = self._get_searched_nodes(closed_list)
                return path, searched_nodes

            children = self._select_neighbours(
                current_node.node, matrix.shape[0], matrix.shape[1]
            )
            for child in children:
                if (
                    child in [closed_node.node for closed_node in closed_list]
                    or matrix[child.x, child.y] == 1
                ):
                    continue
                g = (
                    current_node.g
                    + 1
                    + costmap[current_node.node.x, current_node.node.y]
                    * self.costmap_weight_
                )  # 1 is default traversal cost
                h = self.heuristic_(child, goal) * self.heuristic_weight
                child_node = NodeAstar(child, current_node.node, h, g, h + g)

                for open_node in open_list:
                    if open_node.node == child_node.node:
                        if open_node.g < child_node.g:
                            continue
                if child_node.node not in [open_node.node for open_node in open_list]:
                    open_list.append(child_node)
            iterations += 1

        raise (PathNotFoundException("Path not found"))
