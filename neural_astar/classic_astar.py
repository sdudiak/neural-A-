#!/usr/bin/env python
import sys
import os
from typing import Optional, Tuple, List, Union, Any, Callable

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
        heuristic: Callable[[Node2d, Node2d], float],
        costmap_weight: int = 1,
        max_iterations: int = 50000000,
        heuristic_weight: int = 5,
    ) -> None:
        """
        Initialize the Astar LightningDataModule.

        :param heuristic: The heuristic function to be used.
        :type heuristic: Callable[[Node2d, Node2d], float]

        :param costmap_weight: Weight applied to the costmap during traversal (default is 1).
        :type costmap_weight: int, optional

        :param max_iterations: Maximum number of iterations for the A* algorithm (default is 50000000).
        :type max_iterations: int, optional

        :param heuristic_weight: Weight applied to the heuristic during traversal (default is 5).
        :type heuristic_weight: int, optional
        """
        super().__init__()
        self.costmap_weight_ = costmap_weight
        self.heuristic_ = heuristic
        self.max_iterations_ = max_iterations
        self.heuristic_weight = heuristic_weight

    def _select_neighbours(self, node: Node2d, max_x: int, max_y: int) -> List[Node2d]:
        """
        Select valid neighbors for a given node.

        :param node: The current node.
        :type node: Node2d

        :param max_x: Maximum x-coordinate limit.
        :type max_x: int

        :param max_y: Maximum y-coordinate limit.
        :type max_y: int

        :return: List of valid neighbor nodes.
        :rtype: List[Node2d]
        """
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

    def _backtrack_path(self, closed_list: List[NodeAstar]) -> List[Node2d]:
        """
        Backtrack the optimal path from the closed list.

        :param closed_list: List of closed nodes.
        :type closed_list: List[NodeAstar]

        :return: Optimal path.
        :rtype: List[Node2d]
        """
        optimal_path = list()
        current_node = closed_list[-1]  # Start from the goal node
        while current_node.parent is not None:
            optimal_path.insert(0, current_node.node)
            current_node = [
                node for node in closed_list if node.node == current_node.parent
            ][0]
        return optimal_path

    def _get_searched_nodes(self, closed_list: List[NodeAstar]) -> List[Node2d]:
        """
        Extract the list of searched nodes from the closed list.

        :param closed_list: List of closed nodes.
        :type closed_list: List[NodeAstar]

        :return: List of searched nodes.
        :rtype: List[Node2d]
        """
        search_history = [astar_node.node for astar_node in closed_list]
        return search_history

    def forward(
        self,
        matrix_batch: torch.Tensor,
        start_batch: torch.Tensor,
        goal_batch: torch.Tensor,
        costmap_batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for A* algorithm.

        :param matrix_batch: Batch of input matrices.
        :type matrix_batch: torch.Tensor

        :param start_batch: Batch of start nodes.
        :type start_batch: torch.Tensor

        :param goal_batch: Batch of goal nodes.
        :type goal_batch: torch.Tensor

        :param costmap_batch: Batch of costmap matrices (optional).
        :type costmap_batch: torch.Tensor, optional

        :return: Tuple of paths and visited nodes.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
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

    def _run_astar(
        self,
        matrix: np.ndarray,
        start: Node2d,
        goal: Node2d,
        costmap: np.ndarray = None,
    ) -> Tuple[List[Node2d], List[Node2d]]:
        """
        Run the A* algorithm.

        :param matrix: Input matrix.
        :type matrix: np.ndarray

        :param start: Start node.
        :type start: Node2d

        :param goal: Goal node.
        :type goal: Node2d

        :param costmap: Costmap matrix (optional).
        :type costmap: np.ndarray, optional

        :return: Tuple of optimal path and list of searched nodes.
        :rtype: Tuple[List[Node2d], List[Node2d]]
        """
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

        raise PathNotFoundException("Path not found")
