"""Prototype and reliability scoring helpers for post-hoc RPC evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class CalibrationConfig:
    pool: str = "flatten"
    k: int = 4
    distance: str = "radius"
    norm: str = "standard"
    alpha: float = 0.5
    reliability_mode: str = "none"
    reliability_gamma: float = 1.0
    reliability_floor: float = 0.0

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "CalibrationConfig":
        return cls(
            pool=str(values.get("pool", "flatten")),
            k=int(values.get("k", 4)),
            distance=str(values.get("distance", "radius")),
            norm=str(values.get("norm", "standard")),
            alpha=float(values.get("alpha", 0.5)),
            reliability_mode=str(values.get("reliability_mode", "none")),
            reliability_gamma=float(values.get("reliability_gamma", 1.0)),
            reliability_floor=float(values.get("reliability_floor", 0.0)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "pool": self.pool,
            "k": self.k,
            "distance": self.distance,
            "norm": self.norm,
            "alpha": self.alpha,
            "reliability_mode": self.reliability_mode,
            "reliability_gamma": self.reliability_gamma,
            "reliability_floor": self.reliability_floor,
        }


def pool_latent(z: torch.Tensor, mode: str = "flatten") -> torch.Tensor:
    if mode == "mean":
        return z.mean(dim=(2, 3))
    if mode == "mean_std":
        return torch.cat((z.mean(dim=(2, 3)), z.std(dim=(2, 3), unbiased=False)), dim=1)
    if mode == "rms":
        return torch.sqrt(torch.mean(z * z, dim=(2, 3)).clamp_min(1e-12))
    if mode == "flatten":
        return z.flatten(start_dim=1)
    raise ValueError(f"Unsupported latent pooling mode: {mode}")


def torch_kmeans(
    x: torch.Tensor,
    k: int,
    num_iters: int = 30,
    seed: int = 0,
    sample_size: int = 50000,
) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, got shape {tuple(x.shape)}")
    if x.shape[0] == 0:
        raise ValueError("Cannot run k-means with no samples")

    if sample_size > 0 and x.shape[0] > sample_size:
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)
        indices = torch.randperm(x.shape[0], generator=generator, device=x.device)[:sample_size]
        x = x[indices]

    num_samples = x.shape[0]
    num_centers = min(int(k), num_samples)
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    init_idx = torch.randperm(num_samples, generator=generator, device=x.device)[:num_centers]
    centers = x[init_idx].clone()

    for _ in range(num_iters):
        distances = torch.cdist(x, centers, p=2)
        assignments = torch.argmin(distances, dim=1)
        new_centers = centers.clone()
        for center_idx in range(num_centers):
            mask = assignments == center_idx
            if mask.any():
                new_centers[center_idx] = x[mask].mean(dim=0)
        if torch.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers
    return centers.detach()


def compute_stats(values: torch.Tensor, norm: str = "standard") -> dict[str, torch.Tensor]:
    values = values.float()
    if norm == "standard":
        return {
            "mean": values.mean().detach().cpu(),
            "std": values.std(unbiased=False).clamp_min(1e-6).detach().cpu(),
        }
    if norm == "robust":
        median = values.median()
        mad = (values - median).abs().median().clamp_min(1e-6)
        return {
            "mean": median.detach().cpu(),
            "std": (mad * 1.4826).clamp_min(1e-6).detach().cpu(),
        }
    raise ValueError(f"Unsupported normalization: {norm}")


def zscore(values: torch.Tensor, stats: dict[str, torch.Tensor]) -> torch.Tensor:
    mean = _as_tensor(stats["mean"], values.device, values.dtype)
    std = _as_tensor(stats["std"], values.device, values.dtype).clamp_min(1e-6)
    return (values - mean) / std


def get_keypoint_reliability(full_pose: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if full_pose.shape[1] < 3:
        segment = torch.ones(full_pose.shape[0], device=full_pose.device, dtype=full_pose.dtype)
        flat = torch.ones_like(z).flatten(start_dim=1)
        return segment, flat
    confidence = full_pose[:, 2].float().clamp(0.0, 1.0)
    segment = confidence.mean(dim=(1, 2))
    flat = confidence.unsqueeze(1).expand(-1, z.shape[1], -1, -1).flatten(start_dim=1)
    return segment, flat


def reliability_gate(values: torch.Tensor, cfg: CalibrationConfig) -> torch.Tensor:
    return values.float().clamp(0.0, 1.0).pow(cfg.reliability_gamma).clamp_min(cfg.reliability_floor)


def euclidean_distance(h: torch.Tensor, centers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    distances = torch.cdist(h.float(), centers.float(), p=2)
    return torch.min(distances, dim=1)


def weighted_euclidean_distance(
    h: torch.Tensor,
    centers: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    distances = _weighted_squared_distances(h.float(), centers.float(), weights.float()).sqrt()
    return torch.min(distances, dim=1)


def diag_mahalanobis_distance(
    h: torch.Tensor,
    centers: torch.Tensor,
    variances: torch.Tensor,
    weights: torch.Tensor | None = None,
    chunk_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    out_dist = []
    out_assign = []
    h = h.float()
    centers = centers.float()
    variances = variances.float().clamp_min(1e-6)
    for start in range(0, h.shape[0], chunk_size):
        cur = h[start : start + chunk_size]
        diff2 = (cur[:, None, :] - centers[None, :, :]).pow(2) / variances[None, :, :]
        if weights is not None:
            cur_weights = weights[start : start + chunk_size].float().clamp_min(1e-6)
            diff2 = diff2 * cur_weights[:, None, :]
            distances = diff2.sum(dim=2).sqrt()
        else:
            distances = diff2.sum(dim=2).sqrt()
        cur_dist, cur_assign = torch.min(distances, dim=1)
        out_dist.append(cur_dist.cpu())
        out_assign.append(cur_assign.cpu())
    return torch.cat(out_dist, dim=0), torch.cat(out_assign, dim=0)


def calibration_distances(
    train_h: torch.Tensor,
    test_h: torch.Tensor,
    centers: torch.Tensor,
    cfg: CalibrationConfig,
    train_weight: torch.Tensor | None = None,
    test_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    weighted = cfg.reliability_mode == "keypoint_weighted"
    if weighted and cfg.pool != "flatten":
        raise ValueError("keypoint_weighted reliability requires flatten pooling")

    if cfg.distance == "euclidean":
        if weighted:
            train_dist, _ = weighted_euclidean_distance(train_h, centers, _require_weight(train_weight))
            test_dist, _ = weighted_euclidean_distance(test_h, centers, _require_weight(test_weight))
        else:
            train_dist, _ = euclidean_distance(train_h, centers)
            test_dist, _ = euclidean_distance(test_h, centers)
        return train_dist.cpu(), test_dist.cpu()

    if cfg.distance == "radius":
        if weighted:
            train_base, train_assign = weighted_euclidean_distance(train_h, centers, _require_weight(train_weight))
            test_base, test_assign = weighted_euclidean_distance(test_h, centers, _require_weight(test_weight))
        else:
            train_base, train_assign = euclidean_distance(train_h, centers)
            test_base, test_assign = euclidean_distance(test_h, centers)
        radius = torch.zeros(centers.shape[0], dtype=torch.float32)
        train_base_cpu = train_base.cpu()
        train_assign_cpu = train_assign.cpu()
        for idx in range(centers.shape[0]):
            mask = train_assign_cpu == idx
            radius[idx] = train_base_cpu[mask].mean() if mask.any() else train_base_cpu.mean()
        radius = radius.clamp_min(1e-6)
        return train_base_cpu / radius[train_assign_cpu], test_base.cpu() / radius[test_assign.cpu()]

    if cfg.distance == "diag_mahalanobis":
        _euclid_dist, assignments = euclidean_distance(train_h, centers)
        assignments = assignments.cpu()
        train_h_cpu = train_h.float().cpu()
        centers_cpu = centers.float().cpu()
        global_var = train_h_cpu.var(dim=0, unbiased=False).clamp_min(1e-6)
        variances = torch.zeros_like(centers_cpu)
        for idx in range(centers_cpu.shape[0]):
            mask = assignments == idx
            variances[idx] = train_h_cpu[mask].var(dim=0, unbiased=False).clamp_min(1e-6) if mask.any() else global_var
        if weighted:
            train_dist, _ = diag_mahalanobis_distance(train_h_cpu, centers_cpu, variances, _require_weight(train_weight).cpu())
            test_dist, _ = diag_mahalanobis_distance(test_h.float().cpu(), centers_cpu, variances, _require_weight(test_weight).cpu())
        else:
            train_dist, _ = diag_mahalanobis_distance(train_h_cpu, centers_cpu, variances)
            test_dist, _ = diag_mahalanobis_distance(test_h.float().cpu(), centers_cpu, variances)
        return train_dist, test_dist

    raise ValueError(f"Unsupported distance: {cfg.distance}")


def fused_normality(
    nll_z: torch.Tensor,
    dist_z: torch.Tensor,
    segment_reliability: torch.Tensor,
    cfg: CalibrationConfig,
) -> torch.Tensor:
    if cfg.reliability_mode == "keypoint_gate":
        alpha_term = cfg.alpha * reliability_gate(segment_reliability, cfg).cpu()
    else:
        alpha_term = cfg.alpha
    return -(nll_z.cpu() + alpha_term * dist_z.cpu())


def reliability_only_normality(
    nll_z: torch.Tensor,
    segment_reliability: torch.Tensor,
    cfg: CalibrationConfig,
) -> torch.Tensor:
    gate = reliability_gate(segment_reliability, cfg).cpu()
    return -(gate * nll_z.cpu())


def _weighted_squared_distances(h: torch.Tensor, centers: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.clamp_min(1e-6)
    diff2 = (h[:, None, :] - centers[None, :, :]).pow(2) * weights[:, None, :]
    return diff2.sum(dim=2)


def _require_weight(weight: torch.Tensor | None) -> torch.Tensor:
    if weight is None:
        raise ValueError("Weighted reliability mode requires weights")
    return weight


def _as_tensor(value: Any, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(value, device=device, dtype=dtype)
