import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
from collections import namedtuple
from astar import ClassicAstar
import random
import math
import os
import re
from custom_types import Node2d, ProblemInstance, node2onehottensor, nodelist2otensor
import heuristics
from display import Displayer
from astar_refactored import Astar
from custom_exceptions import PathNotFoundException


PathPlaningDataItem = namedtuple(
    "PathPlaningDataItem", ["name", "map", "start", "goal", "path"]
)


def name_dataitem(map_name, item_number):
    parts = map_name.split(".")

    # Check if the string contains a dot
    if len(parts) > 1:
        # Take the part before the dot and add the number
        modified_string = parts[0] + "_" + str(item_number)
    else:
        # If there is no dot, simply add the number to the end of the original string
        modified_string = map_name + "_" + str(item_number)

    return modified_string


def png_to_binary_matrix(image_path, target_size, threshold):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(img, 0, threshold, cv2.THRESH_BINARY_INV)
    resized = cv2.resize(binary_image, (target_size, target_size))
    matrix = torch.tensor(resized)
    return resized


def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def generate_random_points(matrix, seed=None):
    if seed is not None:
        random.seed(seed)

    rows, cols = len(matrix), len(matrix[0])
    min_distance = 0.8 * max(rows, cols)  # Minimum distance is 80% of the matrix size

    # Helper function to get a random point in a specific quarter
    def get_random_point(quarter):
        if quarter == 1:
            return random.randint(0, rows // 2 - 1), random.randint(cols // 2, cols - 1)
        elif quarter == 2:
            return random.randint(0, rows // 2 - 1), random.randint(0, cols // 2 - 1)
        elif quarter == 3:
            return random.randint(rows // 2, rows - 1), random.randint(0, cols // 2 - 1)
        elif quarter == 4:
            return random.randint(rows // 2, rows - 1), random.randint(
                cols // 2, cols - 1
            )

    while True:
        start_quarter = random.randint(1, 4)
        goal_quarter = random.randint(1, 4)

        start_point = Node2d(*get_random_point(start_quarter))
        goal_point = Node2d(*get_random_point(goal_quarter))

        # Check if both points are on empty tiles (value = 0), in different quarters,
        # and the distance is at least 80% of the matrix size
        if (
            matrix[start_point.x][start_point.y] == 0
            and matrix[goal_point.x][goal_point.y] == 0
            and start_quarter != goal_quarter
            and calculate_distance(start_point, goal_point) >= min_distance
        ):
            return start_point, goal_point


class PathPlanningDataset(Dataset):
    def __init__(
        self,
        map_folder_path,
        map_size,
        problems_on_one_map,
        heuristic,
        max_astar_iterations:int,
        randomize_points: bool = False,
    ) -> None:
        super().__init__()
        self.data = []
        self.map_folder_path = map_folder_path
        self.map_size = map_size
        self.problems_on_one_map = problems_on_one_map
        self.heuristic = heuristic
        self.randomize_points = randomize_points
        self.max_astar_iterations_ = max_astar_iterations
        self.displayer = Displayer()
        self.prepare_dataset()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        dataitem = self.data[index]
        return dataitem.map, dataitem.start, dataitem.goal, dataitem.path

    def prepare_dataset(self):
        for filename in os.listdir(self.map_folder_path):
            if not filename.endswith(".png"):
                continue
            file_path = os.path.join(self.map_folder_path, filename)

            matrix = png_to_binary_matrix(file_path, self.map_size, 1)
            for seed in range(self.problems_on_one_map):
                if not self.randomize_points:
                    start, goal = generate_random_points(matrix, seed)
                else:
                    start, goal = generate_random_points(matrix,seed=None)
                p = ProblemInstance(
                    input_matrix=matrix, start_node=start, goal_node=goal
                )
                astar = Astar(heuristic=self.heuristic,costmap_weight=1,max_iterations=self.max_astar_iterations_)
                try:
                    solution,_ = astar(matrix,start,goal)
                except PathNotFoundException as e:
                    print("Path not found, skipping: ", e)
                    continue
                name = name_dataitem(filename, seed)
                # Convert to tensorflow types
                matrix = torch.from_numpy(matrix)
                start = node2onehottensor(matrix.shape[0], start)
                goal = node2onehottensor(matrix.shape[0], goal)
                solution = nodelist2otensor(matrix.shape[0], solution)
                data_item = PathPlaningDataItem(name, matrix, start, goal, solution)
                self.data.append(data_item)

    def display_dataitem_by_idx(self, idx):
        dataitem = self.data[idx]
        dp = Displayer()
        dp.add_matrix(dataitem.map)
        dp.add_goal(dataitem.goal)
        dp.add_start(dataitem.start)
        dp.add_solution(dataitem.path)
        dp.add_title(dataitem.name)
        dp.prepare_plot()
        dp.draw()

