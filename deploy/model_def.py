"""
Model builder used for SERVING.

This MUST match the model built in ../model.py so best_model.pth loads 1:1.
It's the same custom ResNet34 U-Net (encoder + UnetDecoder + 1x1 seg head),
segmentation-only — the classification head was disabled during training.
forward() returns mask logits of shape [B, 1, H, W].
"""

import torch.nn as nn
import segmentation_models_pytorch as smp

IMG_SIZE = 256
ENCODER = "resnet34"


class ResUNet(nn.Module):
    """Mirror of the training architecture in ../model.py."""

    def __init__(self):
        super().__init__()

        self.encoder = smp.encoders.get_encoder(
            ENCODER,
            in_channels=1,
            depth=5,
            weights=None,            # weights come from best_model.pth, not ImageNet
        )

        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
        )

        self.seg_head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(features)
        return self.seg_head(decoder_output)


def build_model():
    return ResUNet()
