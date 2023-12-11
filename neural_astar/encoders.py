#!/usr/bin/env python
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__)))

from typing import Any
import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from custom_exceptions import NotConfiguredException


class AstarEncoder(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the AstarEncoder.

        :param args: Additional arguments.
        :param kwargs: Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.encoder = None

    def add_encoder(self, encoder: Any) -> None:
        """
        Add an encoder to the AstarEncoder.

        :param encoder: The encoder to be added.
        :type encoder: Any
        """
        self.encoder = encoder

    def forward(self, x: Any) -> torch.Tensor:
        """
        Forward pass of the AstarEncoder.

        :param x: Input data.
        :type x: Any

        :return: Output tensor.
        :rtype: torch.Tensor
        """
        if self.encoder is None:
            raise NotConfiguredException("No encoder specified")
        y = torch.sigmoid(self.encoder(x))
        return y


class CNN(pl.LightningModule):
    CHANNELS = [32, 64, 128, 256]

    def __init__(self, input_dim: int, encoder_depth: int) -> None:
        """
        Initialize the CNN.

        :param input_dim: Input dimension.
        :type input_dim: int

        :param encoder_depth: Depth of the encoder.
        :type encoder_depth: int
        """
        super(CNN, self).__init__()
        self.channels = [input_dim] + self.CHANNELS[:encoder_depth] + [1]
        self.encoder = self._build_encoder()

    def _build_encoder(self) -> nn.Sequential:
        """
        Build the encoder for the CNN.

        :return: Sequential module representing the encoder.
        :rtype: nn.Sequential
        """
        blocks = []
        for i in range(len(self.channels) - 1):
            blocks.append(
                nn.Conv2d(
                    self.channels[i],
                    self.channels[i + 1],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            blocks.append(nn.BatchNorm2d(self.channels[i + 1]))
            blocks.append(nn.ReLU())
        return nn.Sequential(*blocks[:-1])

    def forward(self, x: Any) -> torch.Tensor:
        """
        Forward pass of the CNN.

        :param x: Input data.
        :type x: Any

        :return: Output tensor.
        :rtype: torch.Tensor
        """
        return self.encoder(x)
