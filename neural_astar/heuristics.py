#!/usr/bin/env python
import sys
import os
from typing import List, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__)))

import math
import torch
from custom_types import Node2d


def chebyshev(p1: Node2d, p2: Node2d) -> int:
    """
    Calculate the Chebyshev distance between two 2D nodes.

    :param p1: The first 2D node.
    :type p1: Node2d

    :param p2: The second 2D node.
    :type p2: Node2d

    :return: The Chebyshev distance between the two nodes.
    :rtype: int
    """
    return max(abs(p1.x - p2.x), abs(p1.y - p2.y))


def euclidean(p1: Node2d, p2: Node2d) -> float:
    """
    Calculate the Euclidean distance between two 2D nodes.

    :param p1: The first 2D node.
    :type p1: Node2d

    :param p2: The second 2D node.
    :type p2: Node2d

    :return: The Euclidean distance between the two nodes.
    :rtype: float
    """
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def differentiable_euclidean(goal_maps: torch.Tensor) -> torch.Tensor:
    """
    Calculate the differentiable Euclidean distance for a tensor of goal maps.

    :param goal_maps: The tensor of goal maps.
    :type goal_maps: torch.Tensor

    :return: The differentiable Euclidean tensor of distances from goals
    :rtype: torch.Tensor
    """
    num_samples, H, W = goal_maps.shape[0], goal_maps.shape[-2], goal_maps.shape[-1]
    grid = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    loc = torch.stack(grid, dim=0).type_as(goal_maps)
    loc_expand = loc.reshape(2, -1).unsqueeze(0).expand(num_samples, 2, -1)
    goal_loc = torch.einsum("kij, bij -> bk", loc, goal_maps)
    goal_loc_expand = goal_loc.unsqueeze(-1).expand(num_samples, 2, -1)
    euc = torch.sqrt(((loc_expand - goal_loc_expand) ** 2).sum(1))
    h = euc.reshape_as(goal_maps)
    return h
