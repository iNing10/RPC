"""Train and evaluate DA-Flow on the provided pose datasets."""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score
from common.data import create_run_dir, make_data_loader

from daflow_model import DAFlow, count_parameters
from pose_dataset import (
    DEFAULT_DATA_ROOT,
    PoseWindowDataset,
    aggregate_frame_scores,
    collect_windows,
    dataset_config,
    labels_scores_by_video,
    pose_files,
    read_raw_frame_labels,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "daflow"
DEFAULT_EVAL_OUTPUT_DIR = PROJECT_ROOT / "eval_outputs" / "daflow"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def as_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(v) for v in value]
    return value


def args_dict(args: argparse.Namespace) -> Dict[str, Any]:
    return {key: as_jsonable(value) for key, value in vars(args).items()}


def setup_logging(run_dir: Path, args: argparse.Namespace, dataset_name: str) -> tuple[logging.Logger, Dict[str, Any]]:
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%b%d_%H%M")
    log_path = log_dir / f"training_{timestamp}.log"

    logger = logging.getLogger(f"daflow.{run_dir}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    training_metrics: Dict[str, Any] = {
        "start_time": datetime.now().isoformat(),
        "args": args_dict(args),
        "epochs": [],
    }

    mode = "Evaluation" if args.checkpoint else "Training"
    logger.info("=" * 60)
    logger.info(f"{mode} Started")
    logger.info(f"Experiment: {run_dir}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info("=" * 60)
    return logger, training_metrics


def save_args(args: argparse.Namespace, run_dir: Path) -> None:
    with (run_dir / "args.json").open("w") as f:
        json.dump(args_dict(args), f)


def checkpoint_state(
    model: DAFlow,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": args,
    }


def save_checkpoint(
    model: DAFlow,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    checkpoint_dir: Path,
    filename: str,
    is_best: bool = False,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / filename
    torch.save(checkpoint_state(model, optimizer, epoch, args), path)
    if is_best:
        shutil.copy(path, checkpoint_dir / "checkpoint.pth.tar")
    return path


def load_checkpoint(model: DAFlow, path: str, device: torch.device) -> Dict[str, Any]:
    state = torch.load(path, map_location=device, weights_only=False)
    state_dict = state.get("state_dict", state.get("model"))
    if state_dict is None:
        raise KeyError(f"Checkpoint {path} does not contain 'state_dict' or legacy 'model'.")
    model.load_state_dict(state_dict)
    return state


def save_training_metrics(training_metrics: Dict[str, Any], run_dir: Path, logger: logging.Logger) -> Path:
    training_metrics["end_time"] = datetime.now().isoformat()
    epochs = training_metrics.get("epochs", [])
    if epochs:
        best_epoch = max(epochs, key=lambda item: item.get("auc", -float("inf")))
        training_metrics["best_auc"] = best_epoch.get("auc")
        training_metrics["best_epoch"] = best_epoch.get("epoch")
    else:
        training_metrics["best_auc"] = None
        training_metrics["best_epoch"] = None

    timestamp = datetime.now().strftime("%b%d_%H%M")
    metrics_path = run_dir / "logs" / f"training_metrics_{timestamp}.json"
    with metrics_path.open("w") as f:
        json.dump(as_jsonable(training_metrics), f, indent=2)
    logger.info(f"Training metrics saved to: {metrics_path}")
    return metrics_path


def log_message(logger: Optional[logging.Logger], message: str) -> None:
    if logger is None:
        print(message, flush=True)
    else:
        logger.info(message)


def make_model(args: argparse.Namespace) -> DAFlow:
    return DAFlow(
        channels=2,
        hidden_channels=args.hidden_channels,
        num_joints=args.num_joints,
        num_steps=args.num_steps,
        prior_mean=args.prior_mean,
        attention_kernel=args.attention_kernel,
        adjacency_strategy=args.adjacency_strategy,
        max_hop=args.max_hop,
        adjacency_norm=args.adjacency_norm,
        temporal_kernel=args.temporal_kernel,
        scale_mode=args.scale_mode,
        flow_permutation=args.flow_permutation,
    )


def train(
    model: DAFlow,
    dataset: PoseWindowDataset,
    args: argparse.Namespace,
    device: torch.device,
    run_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
    training_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    loader = make_data_loader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        seed=args.seed,
        pin_memory=device.type == "cuda",
    )
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamax":
        optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer {args.optimizer!r}")
    model.train()
    history = []
    checkpoint_name = time.strftime("%b%d_%H%M_") + "_checkpoint.pth.tar"
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        log_message(logger, f"Starting Epoch {epoch} / {args.epochs}")
        losses = []
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            loss = model.nll(batch, bits_per_dim=args.bits_per_dim)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        history.append(mean_loss)
        if args.lr_decay != 1.0:
            for group in optimizer.param_groups:
                group["lr"] *= args.lr_decay
        current_lr = optimizer.param_groups[0]["lr"]
        if run_dir is not None:
            save_checkpoint(model, optimizer, epoch, args, run_dir / "checkpoints", checkpoint_name)
        log_message(logger, f"Epoch {epoch}: NLL = {mean_loss:.4f}")
        log_message(logger, "Checkpoint Saved. New LR: {0:.3e}".format(current_lr))
        if training_metrics is not None:
            training_metrics["epochs"].append(
                {
                    "epoch": epoch,
                    "training_time": time.time() - epoch_start_time,
                    "lr": current_lr,
                    "train_nll_loss": mean_loss,
                    "train_total_loss": mean_loss,
                }
            )
    return {"final_nll": history[-1], "best_nll": min(history), "loss_history": history}


@torch.no_grad()
def score_windows(
    model: DAFlow,
    windows: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    scores = []
    for start in range(0, len(windows), args.eval_batch_size):
        batch = torch.from_numpy(windows[start : start + args.eval_batch_size]).to(device)
        log_prob = model.log_prob(batch)
        scores.append(log_prob.detach().cpu().numpy())
    return np.concatenate(scores, axis=0) if scores else np.empty((0,), dtype=np.float32)


def smooth_frame_scores(frame_scores: Dict[str, np.ndarray], sigma: int) -> Dict[str, np.ndarray]:
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


def evaluate(
    model: DAFlow,
    args: argparse.Namespace,
    device: torch.device,
    run_dir: Optional[Path] = None,
) -> Dict[str, object]:
    data_root = Path(args.data_root)
    cfg = dataset_config(args.dataset)
    window = args.window or cfg.test_window
    files = pose_files(data_root, cfg, "test", max_files=args.max_test_files)
    print(f"building eval windows: dataset={cfg.name} files={len(files)}", flush=True)
    windows, meta, frame_counts = collect_windows(
        files,
        window=window,
        stride=args.test_stride,
        num_joints=args.num_joints,
        vid_res=cfg.vid_res,
        missing_tolerance=args.missing_tolerance,
        max_samples=None,
        seed=args.seed,
        segment_score_threshold=args.segment_score_threshold,
    )
    if len(windows) == 0:
        raise RuntimeError("No evaluation windows were built.")
    for video_id in list(frame_counts):
        label_count = len(read_raw_frame_labels(data_root, cfg, video_id))
        frame_counts[video_id] = max(frame_counts[video_id], label_count)
    old_prior_mean = model.prior_mean
    score_prior_mean = getattr(args, "score_prior_mean", None)
    if score_prior_mean is not None:
        model.prior_mean = score_prior_mean
    try:
        window_normality = score_windows(model, windows, args, device)
        scoring_info = {"score_method": "nll"}
    finally:
        model.prior_mean = old_prior_mean
    frame_normality = aggregate_frame_scores(meta, window_normality, frame_counts, reduce="min")
    if cfg.key == "hr_ubnormal":
        frame_scores = {video_id: -scores for video_id, scores in frame_normality.items()}
        _video_ids, gt_arr, scores_arr = labels_scores_by_video(data_root, cfg, frame_scores)
        scores_arr = smooth_score_arrays(scores_arr, args.smooth_sigma)
    else:
        frame_normality = smooth_frame_scores(frame_normality, args.smooth_sigma)
        frame_scores = {video_id: -scores for video_id, scores in frame_normality.items()}
        _video_ids, gt_arr, scores_arr = labels_scores_by_video(data_root, cfg, frame_scores)
    labels = np.concatenate(gt_arr)
    scores = np.concatenate(scores_arr)
    if len(np.unique(labels)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(labels, scores))

    out_dir = run_dir or Path(args.output_dir)
    score_dir = out_dir / "results"
    score_dir.mkdir(parents=True, exist_ok=True)
    score_path = score_dir / "scores.npz"
    np.savez_compressed(
        score_path,
        gt_arr=np.array(gt_arr, dtype=object),
        scores_arr=np.array(scores_arr, dtype=object),
    )

    return {
        "auc": auc,
        "score_prior_mean": score_prior_mean,
        "num_eval_windows": int(len(windows)),
        "num_eval_frames": int(len(labels)),
        "num_eval_anomaly_frames": int(labels.sum()),
        "score_path": str(score_path),
        "scoring": scoring_info,
    }


def run(args: argparse.Namespace) -> Dict[str, object]:
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = choose_device(args.device)
    data_root = Path(args.data_root)
    cfg = dataset_config(args.dataset)
    window = args.window or cfg.train_window

    output_dir = Path(args.output_dir)
    if args.checkpoint and output_dir.expanduser().resolve() == DEFAULT_OUTPUT_DIR:
        output_dir = DEFAULT_EVAL_OUTPUT_DIR
    run_dir = create_run_dir(output_dir, cfg, run_name=args.run_name)
    checkpoint_dir = run_dir / "checkpoints"
    result_dir = run_dir / "results"
    for subdir in (checkpoint_dir, result_dir, run_dir / "logs"):
        subdir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(run_dir)
    args.ckpt_dir = str(checkpoint_dir)
    logger, training_metrics = setup_logging(run_dir, args, cfg.stgnf_name)
    save_args(args, run_dir)

    model = make_model(args).to(device)
    logger.info(f"device={device} parameters={count_parameters(model)}")
    metrics: Dict[str, object] = {
        "dataset": cfg.name,
        "data_root": str(data_root),
        "window": window,
        "train_stride": args.train_stride,
        "test_stride": args.test_stride,
        "parameters": count_parameters(model),
        "args": vars(args).copy(),
    }

    best_eval = None
    best_checkpoint_path = None
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, device)
        metrics["loaded_checkpoint"] = args.checkpoint
        logger.info(f"Checkpoint loaded: {args.checkpoint}")
    else:
        train_files = pose_files(data_root, cfg, "train", max_files=args.max_train_files)
        logger.info(f"building train windows: dataset={cfg.name} files={len(train_files)}")
        train_dataset = PoseWindowDataset(
            train_files,
            window=window,
            stride=args.train_stride,
            num_joints=args.num_joints,
            vid_res=cfg.vid_res,
            missing_tolerance=args.missing_tolerance,
            max_samples=args.max_train_samples,
            seed=args.seed,
            num_transform=args.num_transform,
            segment_score_threshold=args.segment_score_threshold,
            dataset_name=cfg.key,
        )
        logger.info(f"train windows={len(train_dataset)}")
        metrics["num_train_windows"] = len(train_dataset)
        if args.eval_every_epoch:
            metrics["train"] = train_with_epoch_eval(
                model,
                train_dataset,
                args,
                device,
                run_dir,
                logger,
                training_metrics,
            )
            best_eval = metrics["train"].get("best_eval")
            best_checkpoint_path = metrics["train"].get("best_checkpoint")
            if args.use_best and best_checkpoint_path:
                load_checkpoint(model, best_checkpoint_path, device)
        else:
            metrics["train"] = train(model, train_dataset, args, device, run_dir, logger, training_metrics)
        save_training_metrics(training_metrics, run_dir, logger)
        if best_checkpoint_path:
            metrics["best_checkpoint"] = best_checkpoint_path

    if not args.no_eval:
        metrics["eval"] = evaluate(model, args, device, run_dir)
        auc = float(metrics["eval"]["auc"])
        num_frames = int(metrics["eval"]["num_eval_frames"])
        logger.info(f"AUC={auc:.4f}")
        print("\nEvaluation complete")
        print(f"  Dataset: {cfg.stgnf_name}")
        print(f"  Frame AUC: {auc:.6f}")
        print(f"  Frames: {num_frames}")
        print()
    return metrics


def train_with_epoch_eval(
    model: DAFlow,
    dataset: PoseWindowDataset,
    args: argparse.Namespace,
    device: torch.device,
    run_dir: Path,
    logger: logging.Logger,
    training_metrics: Dict[str, Any],
) -> Dict[str, object]:
    loader = make_data_loader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        seed=args.seed,
        pin_memory=device.type == "cuda",
    )
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamax":
        optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer {args.optimizer!r}")

    history = []
    epoch_evals = []
    best_auc = -float("inf")
    best_eval = None
    best_checkpoint = None
    time_str = time.strftime("%b%d_%H%M_")
    checkpoint_name = time_str + "_checkpoint.pth.tar"
    best_checkpoint_name = time_str + "_best_checkpoint.pth.tar"
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        logger.info(f"Starting Epoch {epoch} / {args.epochs}")
        model.train()
        losses = []
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            loss = model.nll(batch, bits_per_dim=args.bits_per_dim)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        history.append(mean_loss)
        save_checkpoint(model, optimizer, epoch, args, run_dir / "checkpoints", checkpoint_name)
        if args.lr_decay != 1.0:
            for group in optimizer.param_groups:
                group["lr"] *= args.lr_decay
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch {epoch}: NLL = {mean_loss:.4f}")
        logger.info("Checkpoint Saved. New LR: {0:.3e}".format(current_lr))
        logger.info(f"Evaluating epoch {epoch}...")
        epoch_eval = evaluate(model, args, device, run_dir)
        epoch_evals.append({"epoch": epoch, **epoch_eval})
        logger.info(f"Epoch {epoch}: AUC = {epoch_eval['auc'] * 100:.2f}%")
        epoch_metrics = {
            "epoch": epoch,
            "auc": epoch_eval["auc"],
            "is_best": epoch_eval["auc"] > best_auc,
            "training_time": time.time() - epoch_start_time,
            "lr": current_lr,
            "train_nll_loss": mean_loss,
            "train_total_loss": mean_loss,
        }
        training_metrics["epochs"].append(epoch_metrics)
        if epoch_eval["auc"] > best_auc:
            best_auc = float(epoch_eval["auc"])
            best_eval = epoch_eval
            best_checkpoint = save_checkpoint(
                model,
                optimizer,
                epoch,
                args,
                run_dir / "checkpoints",
                best_checkpoint_name,
                is_best=True,
            )
            logger.info(f"New best AUC: {best_auc * 100:.2f}% (Epoch {epoch})")
            logger.info(f"Best model saved: {best_checkpoint_name}")
        else:
            logger.info(f"Current AUC: {epoch_eval['auc'] * 100:.2f}%, Best AUC: {best_auc * 100:.2f}%")

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETED")
    logger.info(f"Best model: Epoch {int(max(epoch_evals, key=lambda item: item['auc'])['epoch']) if epoch_evals else 0} with AUC = {best_auc * 100:.2f}%")
    if best_checkpoint is not None:
        logger.info(f"Best model saved as: {best_checkpoint_name}")
    logger.info("=" * 60)

    return {
        "final_nll": history[-1],
        "best_nll": min(history),
        "loss_history": history,
        "epoch_evals": epoch_evals,
        "best_epoch": int(max(epoch_evals, key=lambda item: item["auc"])["epoch"]) if epoch_evals else None,
        "best_auc": best_auc,
        "best_eval": best_eval,
        "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="shanghaitech",
        help="shanghaitech/stc, shanghaitech-hr, ubnormal, or hr-ubnormal",
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint", default=None, help="Evaluate an existing checkpoint.")
    parser.add_argument("--no-eval", action="store_true")

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-decay", type=float, default=1.0)
    parser.add_argument("--optimizer", default="adam", choices=["adam", "adamax"])
    parser.add_argument("--grad-clip", type=float, default=100.0)
    parser.add_argument("--bits-per-dim", action="store_true")
    parser.add_argument("--eval-every-epoch", action="store_true")
    parser.add_argument("--use-best", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--hidden-channels", type=int, default=1)
    parser.add_argument("--num-joints", type=int, default=18)
    parser.add_argument("--prior-mean", type=float, default=3.0)
    parser.add_argument(
        "--score-prior-mean",
        type=float,
        default=None,
        help="Optional prior mean used only for evaluation scoring.",
    )
    parser.add_argument("--attention-kernel", type=int, default=7)
    parser.add_argument("--temporal-kernel", type=int, default=1)
    parser.add_argument("--adjacency-strategy", default="distance", choices=["distance", "uniform", "spatial"])
    parser.add_argument("--max-hop", type=int, default=1)
    parser.add_argument("--adjacency-norm", default="digraph", choices=["digraph", "symmetric"])
    parser.add_argument("--scale-mode", default="sigmoid", choices=["sigmoid", "tanh"])
    parser.add_argument("--flow-permutation", default="reverse", choices=["reverse", "invconv"])

    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--train-stride", type=int, default=6)
    parser.add_argument("--test-stride", type=int, default=1)
    parser.add_argument("--missing-tolerance", type=int, default=2)
    parser.add_argument(
        "--segment-score-threshold",
        type=float,
        default=0.0,
        help="Drop windows whose mean tracked-person detection score is at or below this value.",
    )
    parser.add_argument("--smooth-sigma", type=int, default=7)
    parser.add_argument("--num-transform", type=int, default=1)
    parser.add_argument("--max-train-files", type=int, default=None)
    parser.add_argument("--max-test-files", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)

    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Override settings for a quick end-to-end smoke run.",
    )
    args = parser.parse_args(argv)
    if args.fast_dev_run:
        args.epochs = 1
        args.batch_size = min(args.batch_size, 64)
        args.eval_batch_size = min(args.eval_batch_size, 128)
        args.max_train_files = args.max_train_files or 2
        args.max_test_files = args.max_test_files or 3
        args.max_train_samples = args.max_train_samples or 256
    return args


def main(argv: Optional[list[str]] = None) -> Dict[str, object]:
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
