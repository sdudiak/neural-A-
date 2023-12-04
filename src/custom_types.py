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
    return Node2d(coords[:, -2], coords[:, -1])
