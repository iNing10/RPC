from typing import Sequence

import torch
import torch.nn as nn


class MultiScaleTemporalConv(nn.Module):
    """Multi-scale temporal convolution for ST-GCN features."""

    def __init__(self, channels: int, kernel_size: int = 3, dilations: Sequence[int] = (1, 2, 3)):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        self.channels = channels
        self.dilations = tuple(int(dilation) for dilation in dilations)
        if not self.dilations:
            raise ValueError("dilations must not be empty")
        if any(dilation <= 0 for dilation in self.dilations):
            raise ValueError("dilations must be positive")

        branches = []
        for dilation in self.dilations:
            pad = ((kernel_size - 1) // 2) * dilation
            branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=(kernel_size, 1),
                        padding=(pad, 0),
                        dilation=(dilation, 1),
                        bias=False,
                    ),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.branches = nn.ModuleList(branches)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * len(self.dilations), channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected [B,C,T,V], got {tuple(x.shape)}")
        out = torch.cat([branch(x) for branch in self.branches], dim=1)
        out = self.fuse(out)
        return self.act(out + x)
