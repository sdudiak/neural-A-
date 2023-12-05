import unittest
import torch
from neural_astar.heuristics import (
    chebyshev,
    euclidean,
    differentiable_euclidean,
    Node2d,
)


class TestDistanceFunctions(unittest.TestCase):
    def test_chebyshev(self):
        p1 = Node2d(1, 2)
        p2 = Node2d(4, 6)
        result = chebyshev(p1, p2)
        self.assertEqual(result, 4)

    def test_euclidean(self):
        p1 = Node2d(1, 2)
        p2 = Node2d(4, 6)
        result = euclidean(p1, p2)
        self.assertAlmostEqual(result, 5.0, places=5)

    def test_differentiable_euclidean(self):
        goal_maps = torch.tensor(
            [[[0, 1], [1, 0]], [[1, 0], [0, 1]]], dtype=torch.float32
        )
        result = differentiable_euclidean(goal_maps)
        expected_result = torch.tensor(
            [[[1.4142, 1.0000], [1.0000, 0.0000]], [[1.4142, 1.0000], [1.0000, 0.0000]]]
        )
        self.assertTrue(torch.allclose(result, expected_result, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
