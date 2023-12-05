#!/usr/bin/env python
import sys
import os
from typing import Tuple, List

sys.path.append(os.path.join(os.path.dirname(__file__)))

from collections import namedtuple
import numpy as np
import torch

Node2d = namedtuple("Node2d", ["x", "y"])
"""
A 2D coordinate node represented by its x and y coordinates.
"""

ClassicAstarNode = namedtuple("ClassicAstarNode", ["node", "parent", "h", "g", "cost"])
"""
A node representation used in the Classic A* algorithm.

Attributes:
    node (Node2d): The current 2D node.
    parent (Node2d): The parent node in the path.
    h (float): The heuristic cost from the current node to the goal.
    g (float): The cost from the start node to the current node.
    cost (float): The total cost (g + h) for the current node.
"""


def node2onehottensor(tensor_shape: int, node: Node2d) -> torch.Tensor:
    """
    Convert a Node2d to a one-hot tensor.

    :param tensor_shape: The shape of the resulting one-hot tensor.
    :type tensor_shape: int

    :param node: The input 2D node.
    :type node: Node2d

    :return: The one-hot tensor.
    :rtype: torch.Tensor
    """
    onehot = torch.zeros((tensor_shape, tensor_shape))
    onehot[node.x, node.y] = 1
    return onehot


def nodelist2otensor(tensor_shape: int, node_list: List[Node2d]) -> torch.Tensor:
    """
    Convert a list of Node2d to a binary tensor.

    :param tensor_shape: The shape of the resulting binary tensor.
    :type tensor_shape: int

    :param node_list: The list of 2D nodes.
    :type node_list: List[Node2d]

    :return: The binary tensor.
    :rtype: torch.Tensor
    """
    tensor = torch.zeros((tensor_shape, tensor_shape))
    for node in node_list:
        tensor[node.x, node.y] = 1
    return tensor


def onehottensor2node2d(tensor: torch.Tensor) -> Node2d:
    """
    Convert a one-hot tensor to a Node2d.

    :param tensor: The input one-hot tensor.
    :type tensor: torch.Tensor

    :return: The corresponding 2D node.
    :rtype: Node2d
    """
    coords = torch.nonzero(tensor == 1)
    return Node2d(coords[:, -2].item(), coords[:, -1].item())
