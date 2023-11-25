import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class EncoderBase(nn.Module):
    
    def __init__(self, input_dim: int, encoder_depth: int = 4):
        self.requires_grad_ = True
        """
        Base Encoder

        Args:
            input_dim (int): input dimension
            encoder_depth (int, optional): depth of the encoder. Defaults to 4.
        """
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
