#!/usr/bin/env python
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__)))


from collections import namedtuple
from custom_exceptions import *
import numpy as np
import torch


Node2d = namedtuple("Node2d", ["x", "y"])
"""
A custom type for a graph node in a costmap representation
"""

ClassicAstarNode = namedtuple("ClassicAstarNode", ["node", "parent", "h", "g", "cost"])


def node2onehottensor(tensor_shape: int, node: Node2d) -> torch.tensor:
    onehot = torch.zeros((tensor_shape, tensor_shape))
    onehot[node.x, node.y] = 1
    return onehot


def nodelist2otensor(tensor_shape: int, node_list):
    tensor = torch.zeros((tensor_shape, tensor_shape))
    for node in node_list:
        tensor[node.x, node.y] = 1
    return tensor

def onehottensor2node2d(tensor):
    coords = torch.nonzero(tensor == 1)
    print(coords)
    return Node2d(coords[:,0],coords[:,1])


class ProblemInstance:
    """
    Defines the problem for the A* class to solve
    """

    def __init__(
        self, input_matrix: np.ndarray, start_node: Node2d, goal_node: Node2d
    ) -> None:
        self.matrix = input_matrix
        self.start = start_node
        self.goal = goal_node
        self.solution = None
        self.search_history = None

        self.run_all_checks()

    def run_all_checks(self) -> None:
        self.check_types()
        self.check_if_2d()
        self.check_if_binary()
        self.check_bounds()
        self.check_points_difference()

    def check_types(self) -> None:
        if not all(
            [
                isinstance(self.matrix, np.ndarray),
                isinstance(self.start, Node2d),
                isinstance(self.goal, Node2d),
            ]
        ):
            print(type(self.matrix))
            raise (InvalidProblemException("Problem attruibutes have incorrect types"))

    def check_if_2d(self) -> None:
        if self.matrix.ndim != 2:
            raise (
                InvalidProblemException(
                    "Given problem matrix does not have two dimensions"
                )
            )

    def check_bounds(self) -> None:
        matrix_shape = self.matrix.shape
        if not (
            0 <= self.start.x < matrix_shape[1]
            and 0 <= self.goal.x < matrix_shape[1]
            and 0 <= self.start.y < matrix_shape[0]
            and 0 <= self.goal.y < matrix_shape[1]
        ):
            raise (InvalidProblemException("Start or Goal node out of map bounds"))

    def check_if_binary(self) -> None:
        if not np.all(np.logical_or(self.matrix == 0, self.matrix == 1)):
            raise (InvalidProblemException("Given matrix is not binary"))

    def check_points_difference(self) -> None:
        if self.goal == self.start:
            raise (InvalidProblemException("Start and Goal are the same point"))

    def check_if_obstructed(self) -> None:
        if (
            self.matrix[self.goal.x, self.goal.y] != 0
            or self.matrix[self.start.x, self.start.y] != 0
        ):
            raise (InvalidProblemException("Start or Goal are obstructed"))

    def has_solution(self) -> bool:
        return self.solution is not None
