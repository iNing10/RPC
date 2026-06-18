"""DA-Flow model components for skeleton-based video anomaly detection.

The implementation follows the paper at a practical level: each flow step uses
ActNorm, invertible channel permutation, and an affine coupling transform whose
conditioner is a GCN followed by the dual attention module (DAM).
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def openpose_edges() -> List[Tuple[int, int]]:
    return [
        (4, 3),
        (3, 2),
        (7, 6),
        (6, 5),
        (13, 12),
        (12, 11),
        (10, 9),
        (9, 8),
        (11, 5),
        (8, 2),
        (5, 1),
        (2, 1),
        (0, 1),
        (15, 0),
        (14, 0),
        (17, 15),
        (16, 14),
    ]


def hop_distance(num_joints: int, edges: List[Tuple[int, int]], max_hop: int) -> Tensor:
    adjacency = torch.zeros(num_joints, num_joints, dtype=torch.float32)
    for i, j in edges:
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0
    hop = torch.full((num_joints, num_joints), float("inf"))
    transfer = [torch.matrix_power(adjacency, d) for d in range(max_hop + 1)]
    arrive = torch.stack(transfer) > 0
    for d in range(max_hop, -1, -1):
        hop[arrive[d]] = float(d)
    return hop


def normalize_digraph(adjacency: Tensor) -> Tensor:
    degree = adjacency.sum(dim=0)
    degree_inv = torch.zeros_like(degree)
    degree_inv[degree > 0] = degree[degree > 0].pow(-1.0)
    return adjacency * degree_inv[None, :]


def normalize_symmetric(adjacency: Tensor) -> Tensor:
    degree = adjacency.sum(dim=0)
    degree_inv_sqrt = torch.zeros_like(degree)
    degree_inv_sqrt[degree > 0] = degree[degree > 0].pow(-0.5)
    return degree_inv_sqrt[:, None] * adjacency * degree_inv_sqrt[None, :]


def skeleton_adjacency(
    num_joints: int = 18,
    strategy: str = "distance",
    max_hop: int = 1,
    norm: str = "digraph",
) -> Tensor:
    """Return ST-GCN-style adjacency partitions for OpenPose-18 skeletons."""
    if num_joints != 18:
        raise ValueError("DA-Flow reproduction expects 18 OpenPose-format joints.")
    edges = [(i, i) for i in range(num_joints)] + openpose_edges()
    hop = hop_distance(num_joints, edges, max_hop=max_hop)
    valid_hops = list(range(max_hop + 1))
    adjacency = torch.zeros(num_joints, num_joints, dtype=torch.float32)
    for h in valid_hops:
        adjacency[hop == h] = 1.0
    if norm == "digraph":
        normalized = normalize_digraph(adjacency)
    elif norm == "symmetric":
        normalized = normalize_symmetric(adjacency)
    else:
        raise ValueError("Adjacency norm must be digraph or symmetric.")

    if strategy == "uniform":
        return normalized.unsqueeze(0)
    if strategy == "distance":
        parts = []
        for h in valid_hops:
            part = torch.zeros_like(normalized)
            part[hop == h] = normalized[hop == h]
            parts.append(part)
        return torch.stack(parts, dim=0)
    if strategy == "spatial":
        center = 1
        parts = []
        for h in valid_hops:
            root = torch.zeros_like(normalized)
            close = torch.zeros_like(normalized)
            further = torch.zeros_like(normalized)
            for i in range(num_joints):
                for j in range(num_joints):
                    if hop[j, i] != h:
                        continue
                    if hop[j, center] == hop[i, center]:
                        root[j, i] = normalized[j, i]
                    elif hop[j, center] > hop[i, center]:
                        close[j, i] = normalized[j, i]
                    else:
                        further[j, i] = normalized[j, i]
            if h == 0:
                parts.append(root)
            else:
                parts.append(root + close)
                parts.append(further)
        return torch.stack(parts, dim=0)
    raise ValueError("Only uniform, distance, and spatial adjacency are supported here.")


class ActNorm(nn.Module):
    """Channel-wise activation normalization with data-dependent init."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.log_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps
        self.initialized = False

    @torch.no_grad()
    def initialize(self, x: Tensor) -> None:
        mean = x.mean(dim=(0, 2, 3), keepdim=True)
        std = x.std(dim=(0, 2, 3), keepdim=True).clamp_min(self.eps)
        self.bias.copy_(-mean)
        self.log_scale.copy_(-torch.log(std))
        self.initialized = True

    def forward(self, x: Tensor, reverse: bool = False) -> Tuple[Tensor, Tensor]:
        if not self.initialized and self.training:
            self.initialize(x)
        _, _, frames, joints = x.shape
        logdet = frames * joints * self.log_scale.sum()
        if reverse:
            y = x * torch.exp(-self.log_scale) - self.bias
            return y, -logdet
        y = (x + self.bias) * torch.exp(self.log_scale)
        return y, logdet


class ChannelReverse(nn.Module):
    """Deterministic channel reversal used by the STG-NF/Glow baseline."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        indices = torch.arange(channels - 1, -1, -1, dtype=torch.long)
        inverse = torch.empty_like(indices)
        inverse[indices] = torch.arange(channels, dtype=torch.long)
        self.register_buffer("indices", indices)
        self.register_buffer("inverse", inverse)

    def forward(self, x: Tensor, reverse: bool = False) -> Tuple[Tensor, Tensor]:
        return x[:, self.inverse if reverse else self.indices], torch.zeros((), device=x.device)


class InvertibleConv1x1(nn.Module):
    """Glow-style trainable 1x1 invertible channel permutation."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        q, _ = torch.linalg.qr(torch.randn(channels, channels))
        if torch.det(q) < 0:
            q[:, 0] *= -1
        self.weight = nn.Parameter(q)

    def forward(self, x: Tensor, reverse: bool = False) -> Tuple[Tensor, Tensor]:
        _, _, frames, joints = x.shape
        weight = torch.inverse(self.weight) if reverse else self.weight
        y = F.conv2d(x, weight.view(weight.shape[0], weight.shape[1], 1, 1))
        logdet = torch.slogdet(self.weight)[1] * frames * joints
        if reverse:
            logdet = -logdet
        return y, logdet


class GraphConv(nn.Module):
    """Distance-partition graph convolution over skeleton joints."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_joints: int,
        strategy: str = "distance",
        max_hop: int = 1,
        adjacency_norm: str = "digraph",
    ) -> None:
        super().__init__()
        adjacency = skeleton_adjacency(
            num_joints,
            strategy=strategy,
            max_hop=max_hop,
            norm=adjacency_norm,
        )
        self.register_buffer("adjacency", adjacency)
        self.proj = nn.Conv2d(in_channels, out_channels * adjacency.shape[0], kernel_size=1)
        self.num_partitions = int(adjacency.shape[0])

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        b, kc, t, v = x.shape
        x = x.view(b, self.num_partitions, kc // self.num_partitions, t, v)
        return torch.einsum("bkctv,kvw->bctw", x, self.adjacency)


class DualAttention(nn.Module):
    """Dual Attention Module from DA-Flow.

    Skeleton attention pools over frames and attends over channel-joint pairs.
    Frame attention pools over joints and attends over frame-channel pairs.
    """

    def __init__(self, channels: int, kernel_size: int = 7) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.skeleton_conv = nn.Conv2d(
            1, 1, kernel_size=(3, kernel_size), padding=(1, pad), bias=False
        )
        self.frame_conv = nn.Conv2d(
            1, 1, kernel_size=(kernel_size, 3), padding=(pad, 1), bias=False
        )
        self.skeleton_bn = nn.BatchNorm2d(1)
        self.frame_bn = nn.BatchNorm2d(1)

    def forward(self, x: Tensor) -> Tensor:
        skel = x.amax(dim=2).unsqueeze(1)
        skel = torch.sigmoid(self.skeleton_bn(self.skeleton_conv(skel)))
        skel = skel.squeeze(1).unsqueeze(2)
        y_skel = x * skel

        frame = x.amax(dim=3).permute(0, 2, 1).unsqueeze(1)
        frame = torch.sigmoid(self.frame_bn(self.frame_conv(frame)))
        frame = frame.squeeze(1).permute(0, 2, 1).unsqueeze(3)
        y_frame = x * frame

        return 0.5 * (y_skel + y_frame) + x


class CouplingConditioner(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_joints: int,
        attention_kernel: int = 7,
        adjacency_strategy: str = "distance",
        max_hop: int = 1,
        adjacency_norm: str = "digraph",
        temporal_kernel: int = 1,
    ) -> None:
        super().__init__()
        output_channels = out_channels * 2
        feature_channels = output_channels if hidden_channels <= 1 else hidden_channels
        layers: List[nn.Module] = [
            GraphConv(
                in_channels,
                feature_channels,
                num_joints,
                strategy=adjacency_strategy,
                max_hop=max_hop,
                adjacency_norm=adjacency_norm,
            ),
        ]
        if temporal_kernel and temporal_kernel > 1:
            pad = temporal_kernel // 2
            layers.extend(
                [
                    nn.BatchNorm2d(feature_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        feature_channels,
                        feature_channels,
                        kernel_size=(temporal_kernel, 1),
                        padding=(pad, 0),
                    ),
                    nn.BatchNorm2d(feature_channels),
                ]
            )
        layers.append(DualAttention(feature_channels, kernel_size=attention_kernel))
        if feature_channels != output_channels:
            layers.append(nn.Conv2d(feature_channels, output_channels, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class AffineCoupling(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        num_joints: int,
        scale_factor: float = 1.5,
        attention_kernel: int = 7,
        adjacency_strategy: str = "distance",
        max_hop: int = 1,
        adjacency_norm: str = "digraph",
        temporal_kernel: int = 1,
        scale_mode: str = "sigmoid",
    ) -> None:
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("AffineCoupling requires an even channel count.")
        self.split = channels // 2
        self.scale_factor = scale_factor
        self.scale_mode = scale_mode
        self.conditioner = CouplingConditioner(
            self.split,
            channels - self.split,
            hidden_channels,
            num_joints,
            attention_kernel=attention_kernel,
            adjacency_strategy=adjacency_strategy,
            max_hop=max_hop,
            adjacency_norm=adjacency_norm,
            temporal_kernel=temporal_kernel,
        )

    def forward(self, x: Tensor, reverse: bool = False) -> Tuple[Tensor, Tensor]:
        x0, x1 = x[:, : self.split], x[:, self.split :]
        shift, scale_raw = self.conditioner(x0).chunk(2, dim=1)
        if self.scale_mode == "sigmoid":
            scale = torch.sigmoid(scale_raw + 2.0) + 1e-6
            log_scale = torch.log(scale)
        else:
            log_scale = torch.tanh(scale_raw) * self.scale_factor
            scale = torch.exp(log_scale)
        if reverse:
            y1 = (x1 - shift) / scale
            logdet = -log_scale.flatten(1).sum(dim=1)
        else:
            y1 = x1 * scale + shift
            logdet = log_scale.flatten(1).sum(dim=1)
        return torch.cat([x0, y1], dim=1), logdet


class FlowStep(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        num_joints: int,
        attention_kernel: int = 7,
        adjacency_strategy: str = "distance",
        max_hop: int = 1,
        adjacency_norm: str = "digraph",
        temporal_kernel: int = 1,
        scale_mode: str = "sigmoid",
        flow_permutation: str = "reverse",
    ) -> None:
        super().__init__()
        self.actnorm = ActNorm(channels)
        if flow_permutation == "reverse":
            self.permutation = ChannelReverse(channels)
        elif flow_permutation == "invconv":
            self.permutation = InvertibleConv1x1(channels)
        else:
            raise ValueError("Flow permutation must be reverse or invconv.")
        self.coupling = AffineCoupling(
            channels,
            hidden_channels,
            num_joints,
            attention_kernel=attention_kernel,
            adjacency_strategy=adjacency_strategy,
            max_hop=max_hop,
            adjacency_norm=adjacency_norm,
            temporal_kernel=temporal_kernel,
            scale_mode=scale_mode,
        )

    def forward(self, x: Tensor, reverse: bool = False) -> Tuple[Tensor, Tensor]:
        if reverse:
            x, ld3 = self.coupling(x, reverse=True)
            x, ld2 = self.permutation(x, reverse=True)
            x, ld1 = self.actnorm(x, reverse=True)
            return x, ld1 + ld2 + ld3
        x, ld1 = self.actnorm(x)
        x, ld2 = self.permutation(x)
        x, ld3 = self.coupling(x)
        return x, ld1 + ld2 + ld3


class DAFlow(nn.Module):
    def __init__(
        self,
        channels: int = 2,
        hidden_channels: int = 1,
        num_joints: int = 18,
        num_steps: int = 8,
        prior_mean: float = 3.0,
        attention_kernel: int = 7,
        adjacency_strategy: str = "distance",
        max_hop: int = 1,
        adjacency_norm: str = "digraph",
        temporal_kernel: int = 1,
        scale_mode: str = "sigmoid",
        flow_permutation: str = "reverse",
    ) -> None:
        super().__init__()
        self.channels = channels
        self.num_joints = num_joints
        self.prior_mean = prior_mean
        self.steps = nn.ModuleList(
            [
                FlowStep(
                    channels,
                    hidden_channels,
                    num_joints,
                    attention_kernel=attention_kernel,
                    adjacency_strategy=adjacency_strategy,
                    max_hop=max_hop,
                    adjacency_norm=adjacency_norm,
                    temporal_kernel=temporal_kernel,
                    scale_mode=scale_mode,
                    flow_permutation=flow_permutation,
                )
                for _ in range(num_steps)
            ]
        )

    def forward(self, x: Tensor, reverse: bool = False) -> Tuple[Tensor, Tensor]:
        logdet: Tensor | float
        logdet = 0.0
        steps = reversed(self.steps) if reverse else self.steps
        for step in steps:
            x, ld = step(x, reverse=reverse)
            logdet = logdet + ld
        return x, logdet

    def log_prob(self, x: Tensor) -> Tensor:
        z, logdet = self.forward(x)
        return self.log_prob_from_latent(z, logdet)

    def log_prob_from_latent(self, z: Tensor, logdet: Tensor | float) -> Tensor:
        log_base = -0.5 * (
            (z - self.prior_mean).pow(2) + math.log(2.0 * math.pi)
        ).flatten(1).sum(dim=1)
        if not torch.is_tensor(logdet):
            logdet = torch.as_tensor(logdet, device=z.device, dtype=z.dtype)
        if logdet.ndim == 0:
            logdet = logdet.expand_as(log_base)
        return log_base + logdet

    def latent_log_prob(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z, logdet = self.forward(x)
        return z, self.log_prob_from_latent(z, logdet)

    def nll(self, x: Tensor, bits_per_dim: bool = False) -> Tensor:
        loss = -self.log_prob(x)
        if bits_per_dim:
            _, c, t, v = x.shape
            loss = loss / (math.log(2.0) * c * t * v)
        return loss.mean()


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
