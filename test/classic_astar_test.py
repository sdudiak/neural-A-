import unittest
import torch
import numpy as np
from src.classic_astar import (
    Astar,
    Node2d,
    PathNotFoundException,
)  

class TestAstar(unittest.TestCase):
    def setUp(self):
        self.astar = Astar(heuristic=self.simple_heuristic)

    def simple_heuristic(self, node, goal):
        # Simple heuristic for testing
        return abs(node.x - goal.x) + abs(node.y - goal.y)

    def test_run_astar_found_path(self):
        matrix = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        start = torch.tensor([0, 0]).view(1, 2)
        goal = torch.tensor([2, 2]).view(1, 2)

        path, searched_nodes = self.astar._run_astar(
            matrix, Node2d(*start[0]), Node2d(*goal[0])
        )

        self.assertEqual(len(path), 3)
        self.assertEqual(len(searched_nodes), 4)

    def test_run_astar_no_path(self):
        matrix = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        start = torch.tensor([0, 0]).view(1, 2)
        goal = torch.tensor([2, 2]).view(1, 2)

        with self.assertRaises(PathNotFoundException):
            self.astar._run_astar(matrix, Node2d(*start[0]), Node2d(*goal[0]))

    def test_run_astar_custom_costmap(self):
        matrix = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        start = torch.tensor([0, 0]).view(1, 2)
        goal = torch.tensor([2, 2]).view(1, 2)
        costmap = np.array([[1, 2, 1], [1, 0, 1], [1, 2, 1]])

        path, searched_nodes = self.astar._run_astar(
            matrix, Node2d(*start[0]), Node2d(*goal[0]), costmap
        )

        self.assertEqual(len(path), 3)
        self.assertEqual(len(searched_nodes), 4)

    def test_run_astar_large_matrix(self):
        matrix_size = 100
        matrix = np.zeros((matrix_size, matrix_size))
        start = torch.tensor([0, 0]).view(1, 2)
        goal = torch.tensor([matrix_size - 1, matrix_size - 1]).view(1, 2)

        path, searched_nodes = self.astar._run_astar(
            matrix, Node2d(*start[0]), Node2d(*goal[0])
        )

        self.assertGreater(len(path), 1)
        self.assertGreater(len(searched_nodes), 1)

    def test_run_astar_custom_heuristic(self):
        matrix = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        start = torch.tensor([0, 0]).view(1, 2)
        goal = torch.tensor([2, 2]).view(1, 2)

        custom_heuristic = (
            lambda node, goal: abs(node.x - goal.x) + abs(node.y - goal.y) + 1
        )
        astar_with_custom_heuristic = Astar(heuristic=custom_heuristic)

        path, searched_nodes = astar_with_custom_heuristic._run_astar(
            matrix, Node2d(*start[0]), Node2d(*goal[0])
        )

        self.assertEqual(len(path), 3)
        self.assertEqual(len(searched_nodes), 4)


if __name__ == "__main__":
    unittest.main()
