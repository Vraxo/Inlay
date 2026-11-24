"""Small U-Net segmentation model used by Inlay's local detector."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class InlaySegmenter(nn.Module):
    """Predicts the visible foreground-image region as a single-channel mask."""

    input_size = 384

    def __init__(self) -> None:
        super().__init__()
        self.encoder1 = ConvBlock(3, 24)
        self.encoder2 = ConvBlock(24, 48)
        self.encoder3 = ConvBlock(48, 96)
        self.encoder4 = ConvBlock(96, 160)
        self.bottleneck = ConvBlock(160, 224)
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(224, 160, 2, stride=2)
        self.decoder4 = ConvBlock(320, 160)
        self.up3 = nn.ConvTranspose2d(160, 96, 2, stride=2)
        self.decoder3 = ConvBlock(192, 96)
        self.up2 = nn.ConvTranspose2d(96, 48, 2, stride=2)
        self.decoder2 = ConvBlock(96, 48)
        self.up1 = nn.ConvTranspose2d(48, 24, 2, stride=2)
        self.decoder1 = ConvBlock(48, 24)
        self.output = nn.Conv2d(24, 1, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        first = self.encoder1(value)
        second = self.encoder2(self.pool(first))
        third = self.encoder3(self.pool(second))
        fourth = self.encoder4(self.pool(third))
        center = self.bottleneck(self.pool(fourth))
        decoded = self.decoder4(torch.cat((self.up4(center), fourth), dim=1))
        decoded = self.decoder3(torch.cat((self.up3(decoded), third), dim=1))
        decoded = self.decoder2(torch.cat((self.up2(decoded), second), dim=1))
        decoded = self.decoder1(torch.cat((self.up1(decoded), first), dim=1))
        return self.output(decoded)


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    binary = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = 1 - ((2 * intersection + 1) / (denominator + 1))
    return binary + dice.mean()
