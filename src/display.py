import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__)))

from custom_exceptions import NotConfiguredException
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from enum import Enum
from custom_types import Node2d, ProblemInstance
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
import torch


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


import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

green_colormap = mpl.colormaps["Greens"]
green_norm = mpl.colors.Normalize(vmin=0, vmax=1)


def matrix_check(func):
    def wrapper(self, *args, **kwargs):
        if self.matrix_ is None:
            raise NotConfiguredException("Displayer does not have a matrix to show")
        return func(self, *args, **kwargs)

    return wrapper


class Displayer:
    def __init__(self) -> None:
        self.matrix_ = None
        self.cost_matrix_ = None
        self.has_cost_ = False
        self.plot_ready_ = False
        self.plot_ = None
        self.plot_title_ = None

    def add_matrix(self, matrix: np.ndarray):
        self.matrix_ = copy.deepcopy(matrix)

    def _color_main_matrix(self, input, color):
        coords = torch.nonzero(input == 1)
        for c in coords:
            if self.matrix_[c[0], c[1]] < color:
                self.matrix_[c[0], c[1]] = color

    @matrix_check
    def add_start(self, start):
        self._color_main_matrix(start, START)

    @matrix_check
    def add_goal(self, goal):
        self._color_main_matrix(goal, GOAL)

    @matrix_check
    def add_solution(self, solution):
        self._color_main_matrix(solution, TRAVERSED)

    @matrix_check
    def add_searched_nodes(self, search_history):
        self._color_main_matrix(search_history, EXPLORED)

    @matrix_check
    def add_cost_matrix(self, cost_matrix):
        self.cost_matrix_ = copy.deepcopy(cost_matrix)
        self.has_cost_ = True

    def add_title(self, title):
        self.plot_title_ = title

    @matrix_check
    def prepare_plot(self):
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
        if self.plot_ready_:
            fig, ax = self.plot_
            plt.figure(fig.number)
            plt.show()
