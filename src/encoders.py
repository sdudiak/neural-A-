import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import lightning as pl


class EncoderBase(pl.LightningModule):
    def __init__(self, input_dim: int, encoder_depth: int = 4):
        super().__init__()
        self.model = self.construct_encoder(input_dim, encoder_depth)

    def construct_encoder(self, input_dim, encoder_depth) -> nn.Module:
        pass

    def forward(self, x):
        y = torch.sigmoid(self.model(x))
        return y


class Unet(EncoderBase):
    DECODER_CHANNELS = [256, 128, 64, 32, 16]

    def construct_encoder(self, input_dim: int, encoder_depth: int) -> nn.Module:
        """
        Unet encoder

        Args:
            input_dim (int): input dimension
            encoder_depth (int, optional): depth of the encoder.
        """
        decoder_channels = self.DECODER_CHANNELS[:encoder_depth]
        return smp.Unet(
            encoder_name="vgg16_bn",
            encoder_weights=None,
            classes=1,
            in_channels=input_dim,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
        )


class CNN(EncoderBase):
    CHANNELS = [32, 64, 128, 256]

    def construct_encoder(self, input_dim: int, encoder_depth: int) -> nn.Module:
        """
        Simple CNN encoder

        Args:
            input_dim (int): input dimension
            encoder_depth (int, optional): depth of the encoder.
        """
        channels = [input_dim] + self.CHANNELS[:encoder_depth] + [1]
        blocks = []
        for i in range(len(channels) - 1):
            blocks.append(nn.Conv2d(channels[i], channels[i + 1], 3, 1, 1))
            blocks.append(nn.BatchNorm2d(channels[i + 1]))
            blocks.append(nn.ReLU())
        return nn.Sequential(*blocks[:-1])
