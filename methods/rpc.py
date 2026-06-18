#!/usr/bin/env python3
"""RPC post-hoc calibration on fixed baseline checkpoints."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
STGNF_ROOT = ROOT / "baselines" / "stgnf"
DAFLOW_ROOT = ROOT / "baselines" / "daflow"
for module_path in (ROOT, STGNF_ROOT, DAFLOW_ROOT):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from args import init_parser, init_sub_args  # noqa: E402
from common.data import (  # noqa: E402
    aggregate_frame_scores,
    labels_scores_by_video,
    pose_files,
    resolve_dataset,
)
from dataset import get_dataset_and_loader  # noqa: E402
from models.STG_NF.model_pose import STG_NF  # noqa: E402
from pose_dataset import collect_windows, dataset_config, read_raw_frame_labels  # noqa: E402
from run_experiment import (  # noqa: E402
    choose_device as choose_daflow_device,
    load_checkpoint as load_daflow_checkpoint,
    make_model as make_daflow_model,
)
from utils.data_utils import trans_list  # noqa: E402
from utils.scoring_utils import (  # noqa: E402
    apply_dataset_eval_masks,
    get_dataset_scores,
    smooth_scores,
)
from utils.train_utils import init_model_params  # noqa: E402

from rpc_core import (  # noqa: E402
    CalibrationConfig,
    calibration_distances,
    compute_stats,
    fused_normality,
    get_keypoint_reliability,
    pool_latent,
    reliability_gate,
    reliability_only_normality,
    torch_kmeans,
    zscore,
)


BACKBONE_NAMES = {"stg-nf": "STG-NF", "daflow": "DA-Flow"}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser(argv)
    args = parser.parse_args(argv)
    if args.checkpoint is None:
        parser.error("--checkpoint is required")

    set_seed(args.seed)
    result = run_eval(args)

    if getattr(args, "output_json", None):
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(as_jsonable(result), indent=2), encoding="utf-8")
    print(json.dumps(as_jsonable(result), indent=2), flush=True)
    return 0


def build_parser(argv: list[str]) -> argparse.ArgumentParser:
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--backbone", default="stg-nf", choices=sorted(BACKBONE_NAMES))
    known, _unknown = probe.parse_known_args(argv)
    if known.backbone == "stg-nf":
        parser = init_parser()
    elif known.backbone == "daflow":
        parser = build_daflow_parser()
    else:
        raise ValueError(f"Unsupported backbone: {known.backbone}")
    parser.description = __doc__
    add_rpc_args(parser)
    return parser


def add_rpc_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backbone", default="stg-nf", choices=sorted(BACKBONE_NAMES))
    parser.add_argument("--calibration", default="rpc", choices=["r", "pc", "rpc", "all"])
    parser.add_argument("--output-json", "--output_json", default=None)

    parser.add_argument("--rpc-pool", "--rpc_pool", default=None, choices=["flatten", "mean", "mean_std", "rms"])
    parser.add_argument("--rpc-pc-pool", "--rpc_pc_pool", default=None, choices=["flatten", "mean", "mean_std", "rms"])
    parser.add_argument("--rpc-pc-k", "--rpc_pc_k", type=int, default=None)
    parser.add_argument("--rpc-pc-alpha", "--rpc_pc_alpha", type=float, default=None)
    parser.add_argument("--rpc-pc-distance", "--rpc_pc_distance", default=None, choices=["euclidean", "radius", "diag_mahalanobis"])
    parser.add_argument("--rpc-k", "--rpc_k", type=int, default=None)
    parser.add_argument("--rpc-alpha", "--rpc_alpha", type=float, default=None)
    parser.add_argument("--rpc-distance", "--rpc_distance", default=None, choices=["euclidean", "radius", "diag_mahalanobis"])
    parser.add_argument("--rpc-reliability-mode", "--rpc_reliability_mode", default=None, choices=["none", "keypoint_gate", "keypoint_weighted"])
    parser.add_argument("--rpc-reliability-gamma", "--rpc_reliability_gamma", type=float, default=None)
    parser.add_argument("--rpc-reliability-floor", "--rpc_reliability_floor", type=float, default=None)
    parser.add_argument("--rpc-norm", "--rpc_norm", default=None, choices=["standard", "robust"])
    parser.add_argument("--rpc-kmeans-iters", "--rpc_kmeans_iters", type=int, default=30)
    parser.add_argument("--rpc-kmeans-sample-size", "--rpc_kmeans_sample_size", type=int, default=50000)
    parser.add_argument("--rpc-kmeans-seed", "--rpc_kmeans_seed", type=int, default=None)


def build_daflow_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="DA-Flow RPC")
    parser.add_argument("--dataset", default="shanghaitech")
    parser.add_argument("--data-root", "--data_root", default=str(ROOT / "data"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--no-eval", action="store_true")

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=256)
    parser.add_argument("--eval-batch-size", "--eval_batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", "--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr-decay", "--lr_decay", type=float, default=1.0)
    parser.add_argument("--optimizer", default="adam", choices=["adam", "adamax"])
    parser.add_argument("--grad-clip", "--grad_clip", type=float, default=100.0)
    parser.add_argument("--bits-per-dim", "--bits_per_dim", action="store_true")
    parser.add_argument("--eval-every-epoch", "--eval_every_epoch", action="store_true")
    parser.add_argument("--use-best", "--use_best", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", "--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--num-steps", "--num_steps", type=int, default=8)
    parser.add_argument("--hidden-channels", "--hidden_channels", type=int, default=1)
    parser.add_argument("--num-joints", "--num_joints", type=int, default=18)
    parser.add_argument("--prior-mean", "--prior_mean", type=float, default=3.0)
    parser.add_argument("--score-prior-mean", "--score_prior_mean", type=float, default=None)
    parser.add_argument("--attention-kernel", "--attention_kernel", type=int, default=7)
    parser.add_argument("--temporal-kernel", "--temporal_kernel", type=int, default=1)
    parser.add_argument("--adjacency-strategy", "--adjacency_strategy", default="distance", choices=["distance", "uniform", "spatial"])
    parser.add_argument("--max-hop", "--max_hop", type=int, default=1)
    parser.add_argument("--adjacency-norm", "--adjacency_norm", default="digraph", choices=["digraph", "symmetric"])
    parser.add_argument("--scale-mode", "--scale_mode", default="sigmoid", choices=["sigmoid", "tanh"])
    parser.add_argument("--flow-permutation", "--flow_permutation", default="reverse", choices=["reverse", "invconv"])

    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--train-stride", "--train_stride", type=int, default=6)
    parser.add_argument("--test-stride", "--test_stride", type=int, default=1)
    parser.add_argument("--missing-tolerance", "--missing_tolerance", type=int, default=2)
    parser.add_argument("--segment-score-threshold", "--segment_score_threshold", type=float, default=0.0)
    parser.add_argument("--smooth-sigma", "--smooth_sigma", type=int, default=7)
    parser.add_argument("--num-transform", "--num_transform", type=int, default=1)
    parser.add_argument("--max-train-files", "--max_train_files", type=int, default=None)
    parser.add_argument("--max-test-files", "--max_test_files", type=int, default=None)
    parser.add_argument("--max-train-samples", "--max_train_samples", type=int, default=None)
    parser.add_argument("--fast-dev-run", "--fast_dev_run", action="store_true")
    return parser


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    pools = sorted(required_pools(args))
    context = load_backbone_context(args, pools=pools)
    pc_cfg, rpc_cfg = resolve_configs(args)
    metrics = {BACKBONE_NAMES[args.backbone]: context["score_test"]((-context["test_cache"]["nll"]).numpy().squeeze())}
    if args.calibration in ("r", "all"):
        metrics["R"] = evaluate_reliability_only(context["train_cache"], context["test_cache"], context["score_test"], rpc_cfg, args)
        metrics["R_gain"] = metrics["R"] - metrics[BACKBONE_NAMES[args.backbone]]
    if args.calibration in ("pc", "all"):
        metrics["PC"] = evaluate_calibration(context["train_cache"], context["test_cache"], context["score_test"], pc_cfg, args)
        metrics["PC_gain"] = metrics["PC"] - metrics[BACKBONE_NAMES[args.backbone]]
    if args.calibration in ("rpc", "all"):
        metrics["RPC"] = evaluate_calibration(context["train_cache"], context["test_cache"], context["score_test"], rpc_cfg, args)
        metrics["RPC_gain"] = metrics["RPC"] - metrics[BACKBONE_NAMES[args.backbone]]
    if "RPC" in metrics and "PC" in metrics:
        metrics["RPC_vs_PC"] = metrics["RPC"] - metrics["PC"]

    return {
        "dataset": args.dataset,
        "backbone": args.backbone,
        "mode": "eval",
        "calibration": args.calibration,
        "checkpoint": args.checkpoint,
        "metrics": metrics,
        "configs": {
            "pc": pc_cfg.as_dict(),
            "rpc": rpc_cfg.as_dict(),
            "kmeans_iters": args.rpc_kmeans_iters,
            "kmeans_sample_size": args.rpc_kmeans_sample_size,
            "kmeans_seed": args.rpc_kmeans_seed,
        },
    }


def load_backbone_context(args: argparse.Namespace, pools: list[str]) -> dict[str, Any]:
    if args.backbone == "stg-nf":
        return load_stgnf_context(args, pools)
    if args.backbone == "daflow":
        return load_daflow_context(args, pools)
    raise ValueError(f"Unsupported backbone: {args.backbone}")


def load_stgnf_context(args: argparse.Namespace, pools: list[str]) -> dict[str, Any]:
    args, _model_args = init_sub_args(args)
    dataset, loader = get_dataset_and_loader(args, trans_list=trans_list, only_test=False)
    model = build_stgnf_model(args, dataset)
    load_stgnf_checkpoint(model, args.checkpoint, args.device)
    model.to(args.device)
    model.eval()
    train_cache = extract_stgnf_split(model, loader["train"], args, normal_only=True, pools=pools, keep_inputs=False)
    test_cache = extract_stgnf_split(model, loader["test"], args, normal_only=False, pools=pools, keep_inputs=False)

    def score_test(normality: np.ndarray) -> float:
        return auc_stgnf_from_normality(normality, dataset["test"].metadata, args)

    return {
        "model": model,
        "args": args,
        "train_cache": train_cache,
        "test_cache": test_cache,
        "score_test": score_test,
    }


def load_daflow_context(args: argparse.Namespace, pools: list[str]) -> dict[str, Any]:
    if getattr(args, "fast_dev_run", False):
        args.eval_batch_size = min(args.eval_batch_size, 128)
        args.max_train_files = args.max_train_files or 2
        args.max_test_files = args.max_test_files or 3
        args.max_train_samples = args.max_train_samples or 256
    device = choose_daflow_device(args.device)
    cfg = dataset_config(args.dataset)
    model = make_daflow_model(args).to(device)
    load_daflow_checkpoint(model, args.checkpoint, device)
    model.eval()
    train_window = args.window or cfg.train_window
    test_window = args.window or cfg.test_window
    train_files = pose_files(args.data_root, cfg, "train", max_files=args.max_train_files)
    test_files = pose_files(args.data_root, cfg, "test", max_files=args.max_test_files)
    train_windows, train_conf, _train_meta, _train_counts = collect_windows(
        train_files,
        window=train_window,
        stride=args.train_stride,
        num_joints=args.num_joints,
        vid_res=cfg.vid_res,
        missing_tolerance=args.missing_tolerance,
        max_samples=args.max_train_samples,
        seed=args.seed,
        return_conf=True,
        segment_score_threshold=args.segment_score_threshold,
    )
    train_windows, train_conf = apply_daflow_train_transforms(train_windows, train_conf, args.num_transform)
    test_windows, test_conf, test_meta, test_counts = collect_windows(
        test_files,
        window=test_window,
        stride=args.test_stride,
        num_joints=args.num_joints,
        vid_res=cfg.vid_res,
        missing_tolerance=args.missing_tolerance,
        max_samples=None,
        seed=args.seed,
        return_conf=True,
        segment_score_threshold=args.segment_score_threshold,
    )
    if len(train_windows) == 0:
        raise RuntimeError("No DA-Flow train windows were built.")
    if len(test_windows) == 0:
        raise RuntimeError("No DA-Flow test windows were built.")
    for video_id in list(test_counts):
        label_count = len(read_raw_frame_labels(args.data_root, cfg, video_id))
        test_counts[video_id] = max(test_counts[video_id], label_count)
    train_cache = extract_daflow_tensors(
        model,
        torch.from_numpy(train_windows),
        torch.from_numpy(train_conf),
        args,
        device,
        pools,
        keep_inputs=False,
    )
    test_cache = extract_daflow_tensors(
        model,
        torch.from_numpy(test_windows),
        torch.from_numpy(test_conf),
        args,
        device,
        pools,
        keep_inputs=False,
    )

    def score_test(normality: np.ndarray) -> float:
        return auc_daflow_from_normality(normality, test_meta, test_counts, args)

    return {
        "model": model,
        "args": args,
        "device": device,
        "train_cache": train_cache,
        "test_cache": test_cache,
        "score_test": score_test,
    }


def build_stgnf_model(args: argparse.Namespace, dataset: dict[str, Any]) -> STG_NF:
    model_args = init_model_params(args, dataset)
    return STG_NF(**model_args)


def load_stgnf_checkpoint(model: STG_NF, checkpoint_path: str, device: str) -> None:
    map_location = device if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.set_actnorm_init()


@torch.no_grad()
def extract_stgnf_split(
    model: STG_NF,
    loader: torch.utils.data.DataLoader,
    args: argparse.Namespace,
    normal_only: bool,
    pools: list[str],
    keep_inputs: bool,
) -> dict[str, Any]:
    chunks = []
    for data_arr in loader:
        data = [item.to(args.device, non_blocking=True) for item in data_arr]
        full_pose = data[0].float()
        score_seq = data[-2].float()
        label = data[-1].float()
        if normal_only:
            normal_mask = label > 0
            if not normal_mask.any():
                continue
            full_pose = full_pose[normal_mask]
            score_seq = score_seq[normal_mask]
        chunks.append(
            extract_stgnf_tensors(
                model,
                full_pose.detach().cpu(),
                score_seq.detach().cpu(),
                args,
                pools,
                keep_inputs=keep_inputs,
            )
        )
    if not chunks:
        split = "normal train" if normal_only else "test"
        raise RuntimeError(f"No samples extracted for {split} split")
    return concat_caches(chunks)


@torch.no_grad()
def extract_stgnf_tensors(
    model: STG_NF,
    full_pose: torch.Tensor,
    score_seq: torch.Tensor,
    args: argparse.Namespace,
    pools: list[str],
    keep_inputs: bool = True,
) -> dict[str, Any]:
    parts = empty_cache_parts(pools)
    batch_size = int(getattr(args, "batch_size", 256))
    for start in range(0, full_pose.shape[0], batch_size):
        pose = full_pose[start : start + batch_size].to(args.device, non_blocking=True).float()
        score = score_seq[start : start + batch_size].to(args.device, non_blocking=True).float()
        label = torch.ones(pose.shape[0], dtype=torch.float32, device=args.device)
        flow_input = pose if args.model_confidence else pose[:, :2]
        z, nll, _aux = model(x=flow_input, label=label, return_latent=True)
        if args.model_confidence:
            nll = nll * score.amin(dim=-1)
        segment_reliability, flat_reliability = get_keypoint_reliability(pose, z)
        append_cache_parts(parts, nll, segment_reliability, flat_reliability, z, pools)
    cache = finalize_cache_parts(parts)
    if keep_inputs:
        cache["pose"] = full_pose.detach().cpu().float()
        cache["score_seq"] = score_seq.detach().cpu().float()
    return cache


@torch.no_grad()
def extract_daflow_tensors(
    model: torch.nn.Module,
    windows: torch.Tensor,
    confidence: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    pools: list[str],
    keep_inputs: bool,
) -> dict[str, Any]:
    parts = empty_cache_parts(pools)
    old_prior_mean = model.prior_mean
    if args.score_prior_mean is not None:
        model.prior_mean = args.score_prior_mean
    try:
        for start in range(0, windows.shape[0], args.eval_batch_size):
            batch = windows[start : start + args.eval_batch_size].to(device).float()
            conf = confidence[start : start + args.eval_batch_size].to(device).float()
            z, log_prob = model.latent_log_prob(batch)
            nll = -log_prob
            if args.bits_per_dim:
                _n, channels, frames, joints = batch.shape
                nll = nll / (np.log(2.0) * channels * frames * joints)
            segment_reliability = conf.clamp(0.0, 1.0).mean(dim=(1, 2))
            flat_reliability = conf.unsqueeze(1).expand(-1, z.shape[1], -1, -1).flatten(start_dim=1)
            append_cache_parts(parts, nll, segment_reliability, flat_reliability, z, pools)
    finally:
        model.prior_mean = old_prior_mean
    cache = finalize_cache_parts(parts)
    if keep_inputs:
        cache["pose"] = windows.detach().cpu().float()
        cache["confidence"] = confidence.detach().cpu().float()
    return cache


def empty_cache_parts(pools: list[str]) -> dict[str, Any]:
    return {
        "nll": [],
        "segment_reliability": [],
        "flat_reliability": [],
        "pooled": {pool: [] for pool in pools},
    }


def append_cache_parts(
    parts: dict[str, Any],
    nll: torch.Tensor,
    segment_reliability: torch.Tensor,
    flat_reliability: torch.Tensor,
    z: torch.Tensor,
    pools: list[str],
) -> None:
    parts["nll"].append(nll.detach().cpu().float())
    parts["segment_reliability"].append(segment_reliability.detach().cpu().float())
    parts["flat_reliability"].append(flat_reliability.detach().cpu().float())
    for pool in pools:
        parts["pooled"][pool].append(pool_latent(z, mode=pool).detach().cpu().float())


def finalize_cache_parts(parts: dict[str, Any]) -> dict[str, Any]:
    return {
        "nll": torch.cat(parts["nll"], dim=0).float(),
        "segment_reliability": torch.cat(parts["segment_reliability"], dim=0).float(),
        "flat_reliability": torch.cat(parts["flat_reliability"], dim=0).float(),
        "pooled": {pool: torch.cat(values, dim=0).float() for pool, values in parts["pooled"].items()},
    }


def evaluate_reliability_only(
    train_cache: dict[str, Any],
    eval_cache: dict[str, Any],
    scorer: Any,
    cfg: CalibrationConfig,
    args: argparse.Namespace,
) -> float:
    normality = reliability_only_scores(train_cache, eval_cache, cfg, args)
    return scorer(normality.numpy().squeeze())


def evaluate_calibration(
    train_cache: dict[str, Any],
    eval_cache: dict[str, Any],
    scorer: Any,
    cfg: CalibrationConfig,
    args: argparse.Namespace,
) -> float:
    normality = calibration_scores(train_cache, eval_cache, cfg, args)
    return scorer(normality.numpy().squeeze())


def reliability_only_scores(
    train_cache: dict[str, Any],
    eval_cache: dict[str, Any],
    cfg: CalibrationConfig,
    args: argparse.Namespace,
) -> torch.Tensor:
    nll_stats = compute_stats(train_cache["nll"], cfg.norm)
    nll_z = zscore(eval_cache["nll"].to(runtime_device(args)), nll_stats).detach().cpu()
    return reliability_only_normality(nll_z, eval_cache["segment_reliability"], cfg)


def calibration_scores(
    train_cache: dict[str, Any],
    eval_cache: dict[str, Any],
    cfg: CalibrationConfig,
    args: argparse.Namespace,
) -> torch.Tensor:
    train_h = train_cache["pooled"][cfg.pool]
    eval_h = eval_cache["pooled"][cfg.pool]
    centers = torch_kmeans(
        train_h.to(runtime_device(args)),
        k=cfg.k,
        num_iters=args.rpc_kmeans_iters,
        seed=effective_kmeans_seed(args, cfg),
        sample_size=args.rpc_kmeans_sample_size,
    ).cpu()
    train_weight = eval_weight = None
    if cfg.reliability_mode == "keypoint_weighted":
        train_weight = reliability_gate(train_cache["flat_reliability"], cfg)
        eval_weight = reliability_gate(eval_cache["flat_reliability"], cfg)
    train_dist, eval_dist = calibration_distances(train_h, eval_h, centers, cfg, train_weight, eval_weight)
    nll_z = zscore(eval_cache["nll"].to(runtime_device(args)), compute_stats(train_cache["nll"], cfg.norm)).detach().cpu()
    dist_z = zscore(eval_dist.to(runtime_device(args)), compute_stats(train_dist, cfg.norm)).detach().cpu()
    return fused_normality(nll_z, dist_z, eval_cache["segment_reliability"], cfg)


def effective_kmeans_seed(args: argparse.Namespace, cfg: CalibrationConfig) -> int:
    if args.rpc_kmeans_seed is not None:
        return int(args.rpc_kmeans_seed)
    return int(args.seed + cfg.k)


def auc_stgnf_from_normality(normality: np.ndarray, metadata: Any, args: argparse.Namespace) -> float:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        gt_arr, scores_arr, clip_names = get_dataset_scores(normality, metadata, args=args)
        scores_arr = smooth_scores(scores_arr)
        gt_arr, scores_arr = apply_dataset_eval_masks(gt_arr, scores_arr, clip_names, args=args)
    labels = np.concatenate(gt_arr)
    scores = np.concatenate(scores_arr)
    return safe_auc(labels, scores)


def auc_daflow_from_normality(
    normality: np.ndarray,
    meta: list[tuple[str, int, str]],
    frame_counts: dict[str, int],
    args: argparse.Namespace,
) -> float:
    frame_normality = aggregate_frame_scores(meta, normality, frame_counts, reduce="min")
    if resolve_dataset(args.dataset).key == "hr_ubnormal":
        frame_anomaly_scores = {video_id: -scores for video_id, scores in frame_normality.items()}
        _video_ids, gt_arr, scores_arr = labels_scores_by_video(args.data_root, args.dataset, frame_anomaly_scores)
        scores_arr = smooth_score_arrays(scores_arr, args.smooth_sigma)
    else:
        frame_normality = smooth_frame_scores(frame_normality, args.smooth_sigma)
        frame_anomaly_scores = {video_id: -scores for video_id, scores in frame_normality.items()}
        _video_ids, gt_arr, scores_arr = labels_scores_by_video(args.data_root, args.dataset, frame_anomaly_scores)
    return safe_auc(np.concatenate(gt_arr), np.concatenate(scores_arr))


def smooth_frame_scores(frame_scores: dict[str, np.ndarray], sigma: int) -> dict[str, np.ndarray]:
    if sigma <= 0:
        return frame_scores
    smoothed = {}
    for video_id, scores in frame_scores.items():
        values = scores.astype(np.float32, copy=True)
        for sig in range(1, sigma):
            values = gaussian_filter1d(values, sigma=sig)
        smoothed[video_id] = values
    return smoothed


def smooth_score_arrays(scores_arr: list[np.ndarray], sigma: int) -> list[np.ndarray]:
    if sigma <= 0:
        return scores_arr
    smoothed = []
    for scores in scores_arr:
        values = np.asarray(scores, dtype=np.float32).copy()
        for sig in range(1, sigma):
            values = gaussian_filter1d(values, sigma=sig)
        smoothed.append(values)
    return smoothed


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(scores)
    if not finite.all():
        if finite.any():
            scores[scores == np.inf] = scores[finite].max()
            scores[scores == -np.inf] = scores[finite].min()
            scores[~np.isfinite(scores)] = 0.0
        else:
            scores = np.zeros_like(scores)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def resolve_configs(args: argparse.Namespace) -> tuple[CalibrationConfig, CalibrationConfig]:
    pc_values = required_pc_values(args)
    rpc_values = required_rpc_values(args)
    return CalibrationConfig.from_dict(pc_values), CalibrationConfig.from_dict(rpc_values)


def required_pc_values(args: argparse.Namespace) -> dict[str, Any]:
    values = {
        "pool": args.rpc_pc_pool or args.rpc_pool,
        "k": args.rpc_pc_k,
        "distance": args.rpc_pc_distance,
        "norm": args.rpc_norm,
        "alpha": args.rpc_pc_alpha,
        "reliability_mode": "none",
        "reliability_gamma": 1.0,
        "reliability_floor": 0.0,
    }
    missing = [key for key in ("pool", "k", "distance", "norm", "alpha") if values[key] is None]
    if missing:
        raise ValueError(f"Missing PC calibration arguments: {', '.join(missing)}")
    return values


def required_rpc_values(args: argparse.Namespace) -> dict[str, Any]:
    values = {
        "pool": args.rpc_pool,
        "k": args.rpc_k,
        "distance": args.rpc_distance,
        "norm": args.rpc_norm,
        "alpha": args.rpc_alpha,
        "reliability_mode": args.rpc_reliability_mode,
        "reliability_gamma": args.rpc_reliability_gamma,
        "reliability_floor": args.rpc_reliability_floor,
    }
    missing = [key for key, value in values.items() if value is None]
    if missing:
        raise ValueError(f"Missing RPC calibration arguments: {', '.join(missing)}")
    return values


def required_pools(args: argparse.Namespace) -> set[str]:
    pools = {args.rpc_pool, args.rpc_pc_pool or args.rpc_pool}
    missing = {pool for pool in pools if pool is None}
    if missing:
        raise ValueError("Missing --rpc-pool for eval mode.")
    return {str(pool) for pool in pools if pool is not None}


def concat_caches(caches: list[dict[str, Any]]) -> dict[str, Any]:
    caches = [cache for cache in caches if cache_len(cache) > 0]
    if not caches:
        raise RuntimeError("Cannot concatenate an empty cache list.")
    pools = list(caches[0]["pooled"])
    result = {
        "nll": torch.cat([cache["nll"] for cache in caches], dim=0).float(),
        "segment_reliability": torch.cat([cache["segment_reliability"] for cache in caches], dim=0).float(),
        "flat_reliability": torch.cat([cache["flat_reliability"] for cache in caches], dim=0).float(),
        "pooled": {
            pool: torch.cat([cache["pooled"][pool] for cache in caches], dim=0).float()
            for pool in pools
        },
    }
    for key in ("pose", "score_seq", "confidence"):
        if all(key in cache for cache in caches):
            result[key] = torch.cat([cache[key] for cache in caches], dim=0).float()
    return result


def cache_len(cache: dict[str, Any]) -> int:
    return int(cache["nll"].shape[0])


def apply_daflow_train_transforms(windows: np.ndarray, confidence: np.ndarray, num_transform: int) -> tuple[np.ndarray, np.ndarray]:
    if num_transform < 2:
        return windows, confidence
    flipped = windows.copy()
    flipped[:, 0] *= -1.0
    return np.concatenate([windows, flipped], axis=0), np.concatenate([confidence, confidence.copy()], axis=0)


def runtime_device(args: argparse.Namespace) -> torch.device | str:
    if args.backbone == "daflow":
        return choose_daflow_device(args.device)
    return args.device


def as_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(item) for item in value]
    return value


def set_seed(seed: int) -> None:
    if seed == 999:
        seed = torch.initial_seed()
    else:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


if __name__ == "__main__":
    raise SystemExit(main())
