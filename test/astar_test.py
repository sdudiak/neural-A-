#!/usr/bin/env python
import unittest

from src.astar import *
from src.custom_types import *

# TODO documentation


class ProblemInstanceTest(unittest.TestCase):
    def test_wrong_types(self):
        with self.assertRaises(InvalidProblemException) as context:
            problem_instance = ProblemInstance(0, 0, 0)
        self.assertEqual(
            str(context.exception), "Problem attruibutes have incorrect types"
        )

    def test_if_2d(self):
        matrix3d = np.array(
            [[[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3], [1, 2, 3]]]
        )
        with self.assertRaises(InvalidProblemException) as context:
            problem_instance = ProblemInstance(matrix3d, Node2d(0, 0), Node2d(0, 0))
        self.assertEqual(
            str(context.exception), "Given problem matrix does not have two dimensions"
        )

    def test_if_in_bounds(self):
        matrix3d = np.array(
            [[[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3], [1, 2, 3]]]
        )
        with self.assertRaises(InvalidProblemException) as context:
            problem_instance = ProblemInstance(matrix3d, Node2d(0, 0), Node2d(0, 0))
        self.assertEqual(
            str(context.exception), "Given problem matrix does not have two dimensions"
        )

    def test_if_in_bounds(self):
        matrix2d = np.array([[0, 0], [0, 0]])
        start_in_bounds = Node2d(0, 0)
        goal_out_of_bounds = Node2d(5, 5)
        with self.assertRaises(InvalidProblemException) as context:
            problem_instance = ProblemInstance(
                matrix2d, start_in_bounds, goal_out_of_bounds
            )
        self.assertEqual(str(context.exception), "Start or Goal node out of map bounds")

    def test_if_binary(self):
        matrix2d = np.array([[0, 0], [0, 2]])
        start = Node2d(0, 0)
        goal = Node2d(1, 1)
        with self.assertRaises(InvalidProblemException) as context:
            problem_instance = ProblemInstance(matrix2d, start, goal)
        self.assertEqual(str(context.exception), "Given matrix is not binary")

    def test_if_different(self):
        matrix2d = np.array([[0, 0], [0, 1]])
        start = Node2d(0, 0)
        goal = Node2d(0, 0)
        with self.assertRaises(InvalidProblemException) as context:
            problem_instance = ProblemInstance(matrix2d, start, goal)
        self.assertEqual(str(context.exception), "Start and Goal are the same point")

    def test_if_proper(self):
        matrix = np.random.choice([0, 1], size=(50, 50))
        s = Node2d(0, 0)
        g = Node2d(49, 49)
        try:
            problem_instance = ProblemInstance(matrix, s, g)
        except IndentationError as e:
            self.fail("Exception should not be thrown here:", e)
