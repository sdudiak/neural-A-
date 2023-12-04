import unittest
import torch
from src.custom_types import (
    Node2d,
    node2onehottensor,
    nodelist2otensor,
    onehottensor2node2d,
)


class TestYourModule(unittest.TestCase):

    def test_node2onehottensor(self):
        tensor_shape = 5
        node = Node2d(2, 3)
        result = node2onehottensor(tensor_shape, node)

        # Check if the result is a torch tensor
        self.assertIsInstance(result, torch.Tensor)

        # Check if the shape of the tensor is correct
        self.assertEqual(result.shape, (tensor_shape, tensor_shape))

        # Check if the correct node is set to 1 in the tensor
        self.assertEqual(result[node.x, node.y], 1)

    def test_nodelist2otensor(self):
        tensor_shape = 5
        node_list = [Node2d(1, 2), Node2d(3, 4)]
        result = nodelist2otensor(tensor_shape, node_list)

        # Check if the result is a torch tensor
        self.assertIsInstance(result, torch.Tensor)

        # Check if the shape of the tensor is correct
        self.assertEqual(result.shape, (tensor_shape, tensor_shape))

        # Check if the correct nodes are set to 1 in the tensor
        for node in node_list:
            self.assertEqual(result[node.x, node.y], 1)

    def test_onehottensor2node2d(self):
        tensor_shape = 5
        tensor = torch.zeros((tensor_shape, tensor_shape))
        tensor[2, 3] = 1
        result = onehottensor2node2d(tensor)

        # Check if the result is a Node2d
        self.assertIsInstance(result, Node2d)

        # Check if the coordinates of the Node2d are correct
        self.assertEqual(result.x.item(), 2)
        self.assertEqual(result.y.item(), 3)


if __name__ == "__main__":
    unittest.main()
