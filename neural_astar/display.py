import sys
import os
from typing import Optional, Tuple
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from custom_exceptions import NotConfiguredException

# Astar colormap macros
FREE = 0
OCCUPIED = 1
EXPLORED = 2
TRAVERSED = 3
START = 4
GOAL = 5

AstarColormapList = [0 for _ in range(6)]
AstarColormapList[FREE] = "white"
AstarColormapList[OCCUPIED] = "black"
AstarColormapList[EXPLORED] = "yellow"
AstarColormapList[TRAVERSED] = "red"
AstarColormapList[START] = "green"
AstarColormapList[GOAL] = "blue"

# Definition of colormap and norms
astar_cmap = mpl.colors.ListedColormap(AstarColormapList)
bounds = [FREE, OCCUPIED, EXPLORED, TRAVERSED, START, GOAL, 6]
astar_cmap_norm = mpl.colors.BoundaryNorm(bounds, astar_cmap.N)

green_colormap = mpl.colormaps["Greens"]
green_norm = mpl.colors.Normalize(vmin=0, vmax=1)


def matrix_check(func):
    """
    Decorator function to check if the Displayer has a matrix to show.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The decorated function.
    """

    def wrapper(self, *args, **kwargs):
        if self.matrix_ is None:
            raise NotConfiguredException("Displayer does not have a matrix to show")
        return func(self, *args, **kwargs)

    return wrapper


class Displayer:
    """
    Class for displaying matrices with various components highlighted.

    Attributes:
        matrix_ (Optional[np.ndarray]): The matrix to be displayed.
        cost_matrix_ (Optional[np.ndarray]): The cost matrix for additional information.
        has_cost_ (bool): Flag indicating if a cost matrix is available.
        plot_ready_ (bool): Flag indicating if the plot is ready for display.
        plot_ (Optional[Tuple[plt.Figure, plt.Axes]]): The plot figure and axes.
        plot_title_ (Optional[str]): The title for the plot.
    """

    def __init__(self) -> None:
        self.matrix_: Optional[np.ndarray] = None
        self.cost_matrix_: Optional[np.ndarray] = None
        self.has_cost_: bool = False
        self.plot_ready_: bool = False
        self.plot_: Optional[Tuple[plt.Figure, plt.Axes]] = None
        self.plot_title_: Optional[str] = None

    def add_matrix(self, matrix: np.ndarray) -> None:
        """
        Add the main matrix to the Displayer.

        Args:
            matrix (np.ndarray): The matrix to be displayed.
        """
        self.matrix_ = copy.deepcopy(matrix)

    def _color_main_matrix(self, input, color):
        """
        Color the main matrix based on specific elements.

        Args:
            input: The input to determine coloring.
            color: The color to apply.
        """
        coords = torch.nonzero(input == 1)
        for c in coords:
            if self.matrix_[c[0], c[1]] < color:
                self.matrix_[c[0], c[1]] = color

    @matrix_check
    def add_start(self, start):
        """
        Add the starting point to the main matrix.

        Args:
            start: The starting point.
        """
        self._color_main_matrix(start, START)

    @matrix_check
    def add_goal(self, goal):
        """
        Add the goal point to the main matrix.

        Args:
            goal: The goal point.
        """
        self._color_main_matrix(goal, GOAL)

    @matrix_check
    def add_solution(self, solution):
        """
        Add the solution path to the main matrix.

        Args:
            solution: The solution path.
        """
        self._color_main_matrix(solution, TRAVERSED)

    @matrix_check
    def add_searched_nodes(self, search_history):
        """
        Add the searched nodes to the main matrix.

        Args:
            search_history: The searched nodes.
        """
        self._color_main_matrix(search_history, EXPLORED)

    @matrix_check
    def add_cost_matrix(self, cost_matrix):
        """
        Add the cost matrix to the Displayer.

        Args:
            cost_matrix: The cost matrix.
        """
        self.cost_matrix_ = copy.deepcopy(cost_matrix)
        self.has_cost_ = True

    def add_title(self, title):
        """
        Add a title to the plot.

        Args:
            title: The title for the plot.
        """
        self.plot_title_ = title

    @matrix_check
    def prepare_plot(self):
        """
        Prepare the plot for display.
        """
        # Prevent modification of the original matrix
        fig, ax = plt.subplots()
        main_matrix = ax.imshow(self.matrix_, cmap=astar_cmap, norm=astar_cmap_norm)
        white_mask = self.matrix_ == 0.0  # Mask for empty tiles
        # Add another plot only to the white tiles
        if self.has_cost_:
            self.cost_matrix_[~white_mask] = np.nan  # Set non-white tiles to NaN
            cost_matrix = ax.imshow(
                self.cost_matrix_, cmap=green_colormap, norm=green_norm
            )
        if self.plot_title_ is not None:
            plt.title(self.plot_title_)
        self.plot_ready_ = True
        self.plot_ = (fig, ax)

    @matrix_check
    def draw(self):
        """
        Draw the prepared plot.
        """
        if self.plot_ready_:
            fig, ax = self.plot_
            plt.figure(fig.number)
            plt.show()

    def draw_full_astar_output(
        self, matrix, start, goal, search_history, path, costmap=None
    ):
        """
        Draw the full A* algorithm output.

        Args:
            matrix: The main matrix.
            start: The starting point.
            goal: The goal point.
            search_history: The searched nodes.
            path: The solution path.
            costmap: The cost map.
        """
        self.add_matrix(matrix)
        self.add_start(start)
        self.add_goal(goal)
        self.add_solution(path)
        self.add_searched_nodes(search_history)
        if costmap is not None:
            self.add_cost_matrix(costmap)
        self.prepare_plot()
        self.draw()
