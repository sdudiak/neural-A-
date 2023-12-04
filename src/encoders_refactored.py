from typing import Any
import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from custom_exceptions import NotConfiguredException


class AstarEncoder(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = None

    def add_encoder(self, encoder):
        self.encoder = encoder

    def forward(self, x):
        if self.encoder is None:
            raise NotConfiguredException("No encoder specified")
        y = torch.sigmoid(self.encoder(x))
        return y


class CNN(pl.LightningModule):
    CHANNELS = [32, 64, 128, 256]

    def __init__(self, input_dim: int, encoder_depth: int):
        super(CNN, self).__init__()
        self.channels = [input_dim] + self.CHANNELS[:encoder_depth] + [1]
        self.encoder = self._build_encoder()

    def _build_encoder(self):
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

    def forward(self, x):
        return self.encoder(x)