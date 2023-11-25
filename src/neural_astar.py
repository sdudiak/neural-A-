from typing import Any
from astar import ClassicAstar
import lightning as pl
import encoders
import custom_types
from astar_refactored import Astar
import heuristics
from custom_types import onehottensor2node2d, nodelist2otensor
import torch
from display import Displayer


class NeuralAstar(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = encoders.Unet(1, 1)
        self.astar = Astar(heuristic=heuristics.euclidian, costmap_weight=0.9)
        self.displayer = Displayer()

    def forward(self, matrix_batch, start_batch, goal_batch):

        matrix_batch = matrix_batch.float()
        matrix_batch_squeezed = matrix_batch.unsqueeze(1)  # add dimentsion
        costmaps = self.encoder(matrix_batch_squeezed)

        paths_batch =torch.tensor([])
        visited_nodes_batch =torch.tensor([])
        for matrix, start, goal, costmap in zip(
            matrix_batch, start_batch, goal_batch, costmaps
        ):
            np_costmap = costmap[0].detach().numpy()
            self.displayer.add_matrix(matrix)
            self.displayer.add_cost_matrix(np_costmap)
            self.displayer.prepare_plot()
            self.displayer.draw()
            start_node = onehottensor2node2d(start)
            goal_node = onehottensor2node2d(goal)
            path, searched_nodes = self.astar(matrix, start_node, goal_node, np_costmap)
            path_tensor = nodelist2otensor(matrix.shape[0], path)
            searched_nodes_tensor = nodelist2otensor(matrix.shape[0], searched_nodes)
            paths_batch = torch.cat((paths_batch,path_tensor))
            visited_nodes_batch = torch.cat((visited_nodes_batch,searched_nodes_tensor))

        return paths_batch, visited_nodes_batch
