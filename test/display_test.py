import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__)))


import unittest
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.display import Displayer, NotConfiguredException


class TestDisplayer(unittest.TestCase):
    plt.interactive(False)

    def setUp(self):
        self.displayer = Displayer()

    def test_add_matrix(self):
        matrix = np.array([[0, 1], [1, 0]])
        self.displayer.add_matrix(matrix)
        self.assertTrue(np.array_equal(self.displayer.matrix_, matrix))

    def test_add_start(self):
        matrix = torch.tensor([[0, 1], [1, 0]])
        start = torch.tensor([[1, 0], [0, 0]])
        expected_result = torch.tensor([[4, 1], [1, 0]])
        self.displayer.add_matrix(matrix)
        self.displayer.add_start(start)
        self.assertTrue(np.array_equal(self.displayer.matrix_, expected_result))

    def test_add_goal(self):
        matrix = torch.tensor([[0, 1], [1, 0]])
        goal = torch.tensor([[0, 0], [0, 1]])
        expected_result = torch.tensor([[0, 1], [1, 5]])
        self.displayer.add_matrix(matrix)
        self.displayer.add_goal(goal)
        self.assertTrue(np.array_equal(self.displayer.matrix_, expected_result))

    def test_add_solution(self):
        matrix = torch.tensor([[0, 1], [1, 0]])
        solution = torch.tensor([[0, 1], [0, 0]])
        expected_result = torch.tensor([[0, 3], [1, 0]])
        self.displayer.add_matrix(matrix)
        self.displayer.add_solution(solution)
        self.assertTrue(np.array_equal(self.displayer.matrix_, expected_result))

    def test_add_searched_nodes(self):
        matrix = torch.tensor([[0, 1], [1, 0]])
        search_history = torch.tensor([[0, 1], [0, 0]])
        expected_result = torch.tensor([[0, 2], [1, 0]])
        self.displayer.add_matrix(matrix)
        self.displayer.add_searched_nodes(search_history)
        self.assertTrue(np.array_equal(self.displayer.matrix_, expected_result))

    def test_add_cost_matrix(self):
        matrix = torch.tensor([[0, 1], [1, 0]])
        cost_matrix = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        expected_result = torch.tensor([[0, 1], [1, 0]])
        self.displayer.add_matrix(matrix)
        self.displayer.add_cost_matrix(cost_matrix)
        self.assertTrue(np.array_equal(self.displayer.matrix_, expected_result))
        self.assertTrue(np.array_equal(self.displayer.cost_matrix_, cost_matrix))

    def test_add_title(self):
        title = "Test Title"
        self.displayer.add_title(title)
        self.assertEqual(self.displayer.plot_title_, title)

    def test_prepare_plot(self):
        matrix = np.array([[0, 1], [1, 0]])
        self.displayer.add_matrix(matrix)
        self.displayer.prepare_plot()
        self.assertTrue(self.displayer.plot_ready_)

    def test_matrix_check_exception(self):
        with self.assertRaises(NotConfiguredException):
            self.displayer.draw()


if __name__ == "__main__":
    unittest.main()
