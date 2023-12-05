#!/usr/bin/env python
import unittest
import torch

from neural_astar.dataset_utils import (
    PathPlanningDataset,
    PathPlaningDataItem,
    name_dataitem,
    calculate_distance,
    generate_random_points,
)

from neural_astar.custom_types import Node2d

from neural_astar.custom_exceptions import PathNotFoundException

import neural_astar.heuristics as heuristics


class TestPathPlanningDataset(unittest.TestCase):
    def setUp(self):
        self.map_size = 10
        self.problems_on_one_map = 2
        self.heuristic = heuristics.euclidean
        self.max_astar_iterations = 100
        self.randomize_points = False
        self.dataset = PathPlanningDataset(
            None,  # Provide a mock map_folder_path
            self.map_size,
            self.problems_on_one_map,
            self.heuristic,
            self.max_astar_iterations,
            self.randomize_points,
        )

    def tearDown(self):
        pass

    def test_dataset_length(self):
        self.assertEqual(len(self.dataset), 0)

    def test_dataset_item_types(self):
        for idx in range(len(self.dataset)):
            dataitem = self.dataset.data[idx]
            self.assertIsInstance(dataitem, PathPlaningDataItem)
            self.assertIsInstance(dataitem.map, torch.Tensor)
            self.assertIsInstance(dataitem.start, torch.Tensor)
            self.assertIsInstance(dataitem.goal, torch.Tensor)
            self.assertIsInstance(dataitem.path, torch.Tensor)

    def test_astar_execution(self):
        for idx in range(len(self.dataset)):
            dataitem = self.dataset.data[idx]
            astar = self.dataset.astar
            try:
                solution, _ = astar._run_astar(
                    dataitem.map, dataitem.start, dataitem.goal
                )
                self.assertIsInstance(solution, list)
            except PathNotFoundException:
                self.fail("Astar should find a path for the provided dataitem")

    def test_display_dataitem(self):
        for idx in range(len(self.dataset)):
            with self.subTest(idx=idx):
                dataitem = self.dataset.data[idx]
                with self.assertRaises(NotImplementedError):
                    self.dataset.display_dataitem_by_idx(idx)

    def test_name_dataitem(self):
        map_name = "map_name.png"
        item_number = 1
        result = name_dataitem(map_name, item_number)
        expected_result = "map_name_1"
        self.assertEqual(result, expected_result)

    def test_calculate_distance(self):
        point1 = Node2d(0, 0)
        point2 = Node2d(3, 4)
        result = calculate_distance(point1, point2)
        expected_result = 5.0
        self.assertEqual(result, expected_result)

    def test_generate_random_points(self):
        matrix = torch.zeros((10, 10))
        seed = 42
        result = generate_random_points(matrix, seed)
        self.assertIsInstance(result, tuple)
        self.assertEqual(result[0], Node2d(1, 9))
        self.assertEqual(result[1], Node2d(4, 0))


if __name__ == "__main__":
    unittest.main()
