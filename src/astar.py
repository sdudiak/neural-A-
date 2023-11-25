#!/usr/bin/env python

# Fix to include error TODO change it to something better
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__)))

from custom_exceptions import NotConfiguredException
from custom_types import ProblemInstance, Node2d, ClassicAstarNode
from display import Displayer
import heuristics
from typing import List
import torch.nn as nn
import numpy as np
import lightning as pl


class AstarBase():
    """
    A base class for implementing A* algorithm variants
    """

    def __init__(self) -> None:
        super().__init__()
        """
        Basic constructor for the AstarBase class
        """

        self._is_configured = False
        self._problem_instance = None
        self._search_history = None
        self._displayer = Displayer()
        self._costmap = None
        self._costmap_weight = 1

    # def define_problem(self, problem):
    def run(self) -> None:
        """
        Not implemented for the base class
        """
        raise NotImplementedError

    def configure_problem(self, problem_instance: ProblemInstance) -> None:
        self._problem_instance = problem_instance  # No need to check if the problem is valid, creating an invalid problem throws an error immediatelly
        self._costmap = np.zeros(self._problem_instance.matrix.shape)
        self._is_configured = True

    def add_costmap(self, costmap):
        if not self._is_configured:
            raise (NotConfiguredException("No problem instance"))
        self._costmap = costmap

    def display_problem(
        self,
        show_solution: bool = False,
        show_history: bool = False,
        show_costmap: bool = False,
    ) -> None:
        if not self._is_configured:
            raise (NotConfiguredException("No problem instance"))

        self._displayer.add_matrix(self._problem_instance.matrix)
        self._displayer.add_start(self._problem_instance.start)
        self._displayer.add_goal(self._problem_instance.goal)
        if show_solution:
            self._displayer.add_solution(self._problem_instance.solution)
        if show_history:
            self._displayer.add_searched_nodes(self._problem_instance.search_history)
        if show_costmap:
            self._displayer.add_cost_matrix(self._costmap)
        self._displayer.draw()


class ClassicAstar(AstarBase):
    def __init__(self, heuristic) -> None:
        self.heuristic = heuristic
        super().__init__()

    def select_neighbours_(self, node: Node2d):
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
            n
            for n in neighbour_candidates
            if 0 <= n.x < self._problem_instance.matrix.shape[0]
            and 0 <= n.y < self._problem_instance.matrix.shape[1]
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

    def _get_search_history(self, closed_list):
        search_history = list()
        for astar_node in closed_list:
            search_history.append(astar_node.node)
        return search_history

    # TODO add search history
    def run(self, save_searched_tiles: bool = False) -> None:
        # Copies for the sake of clarity
        start = self._problem_instance.start
        goal = self._problem_instance.goal
        matrix = self._problem_instance.matrix

        current_node = None
        open_list = list()
        closed_list = list()

        open_list.append(ClassicAstarNode(start, None, 0, 0, 0))
        while open_list:
            current_node = min(open_list, key=lambda node: node.cost)
            open_list.remove(current_node)
            closed_list.append(current_node)
            if current_node.node == goal:
                if save_searched_tiles:
                    self._problem_instance.search_history = self._get_search_history(
                        closed_list
                    )
                self._problem_instance.solution = self._backtrack_path(closed_list)
                return

            children = self.select_neighbours_(current_node.node)
            for child in children:
                if (
                    child in [closed_node.node for closed_node in closed_list]
                    or matrix[child.x, child.y] == 1
                ):
                    continue
                g = (
                    current_node.g
                    + 1
                    + self._costmap[current_node.node.x, current_node.node.y] * self._costmap_weight
                )  # 1 is default traversal cost
                h = self.heuristic(child, goal)
                child_node = ClassicAstarNode(child, current_node.node, h, g, h + g)

                for open_node in open_list:
                    if open_node.node == child_node.node:
                        if open_node.g < child_node.g:
                            continue
                if child_node.node not in [open_node.node for open_node in open_list]:
                    open_list.append(child_node)
        return None

