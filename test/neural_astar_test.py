import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from neural_astar.neural_astar import NeuralAstar


class MockAstar(nn.Module):
    def forward(self, matrix, start, goal, costmap):
        return torch.ones_like(matrix)


class MockDifferentiableAstar(nn.Module):
    def forward(self, matrix, start, goal, costmap):
        return torch.ones_like(matrix) * 0.5


class MockAstarEncoder(nn.Module):
    def forward(self, input_data):
        return torch.ones(input_data.size(0), 1)


class TestNeuralAstar(unittest.TestCase):
    def setUp(self):
        classic_astar = MockAstar()
        differentiable_astar = MockDifferentiableAstar()
        encoder = MockAstarEncoder()
        self.neural_astar = NeuralAstar(
            classic_astar, differentiable_astar, encoder, use_start_goal_data=True
        )
        self.data_loader = DataLoader(
            TensorDataset(
                torch.ones(1, 1, 5, 5), torch.zeros(1, 1, 5, 5), torch.ones(1, 1, 5, 5)
            )
        )

    def test_forward_classic_astar(self):
        result = self.neural_astar.forward(
            *next(iter(self.data_loader)), use_classic_astar=True
        )
        self.assertTrue(torch.all(result == 1.0))

    def test_forward_differentiable_astar(self):
        result = self.neural_astar.forward(*next(iter(self.data_loader)))
        self.assertTrue(torch.all(result == 0.5))


if __name__ == "__main__":
    unittest.main()
