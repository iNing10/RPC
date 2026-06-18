"""Shared dataset registry, path handling, labels, and DataLoader helpers."""

from __future__ import annotations

import os
import csv
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    folder: str
    stgnf_name: str
    daflow_name: str
    train_window: int
    test_window: int
    vid_res: Tuple[int, int]

    @property
    def name(self) -> str:
        return self.key


DATASET_SPECS = {
    "shanghaitech": DatasetSpec("shanghaitech", "ShanghaiTech", "ShanghaiTech", "shanghaitech", 24, 24, (856, 480)),
    "shanghaitech_hr": DatasetSpec(
        "shanghaitech_hr", "ShanghaiTech", "ShanghaiTech-HR", "shanghaitech-hr", 24, 24, (856, 480)
    ),
    "ubnormal": DatasetSpec("ubnormal", "UBnormal", "UBnormal", "ubnormal", 4, 4, (1080, 720)),
    "hr_ubnormal": DatasetSpec("hr_ubnormal", "UBnormal", "HR-UBnormal", "hr-ubnormal", 4, 4, (1080, 720)),
}

ALIASES = {
    "shanghaitech": "shanghaitech",
    "stc": "shanghaitech",
    "shanghaitech-hr": "shanghaitech_hr",
    "shanghaitech_hr": "shanghaitech_hr",
    "hr-shanghaitech": "shanghaitech_hr",
    "hr_stc": "shanghaitech_hr",
    "hr-stc": "shanghaitech_hr",
    "stc-hr": "shanghaitech_hr",
    "ubnormal": "ubnormal",
    "ubrnormal": "ubnormal",
    "hr-ubnormal": "hr_ubnormal",
    "hr_ubnormal": "hr_ubnormal",
    "ubnormal-hr": "hr_ubnormal",
}

SHANGHAITECH_HR_SKIP = {(1, 130), (1, 135), (1, 136), (6, 144), (6, 145), (12, 152)}
HR_UBNORMAL_KEYS = {"hr_ubnormal"}
UBNORMAL_KEYS = {"ubnormal", *HR_UBNORMAL_KEYS}


def resolve_dataset(name: str | DatasetSpec) -> DatasetSpec:
    if isinstance(name, DatasetSpec):
        return name
    key = ALIASES.get(str(name).lower())
    if key is None:
        valid = ", ".join(sorted(ALIASES))
        raise ValueError(f"Unknown dataset {name!r}. Valid aliases: {valid}")
    return DATASET_SPECS[key]


def dataset_config(name: str | DatasetSpec) -> DatasetSpec:
    """Compatibility alias used by DA-Flow."""
    return resolve_dataset(name)


def dataset_slug(name: str | DatasetSpec) -> str:
    return resolve_dataset(name).key


def stgnf_dataset_name(name: str | DatasetSpec) -> str:
    return resolve_dataset(name).stgnf_name


def daflow_dataset_name(name: str | DatasetSpec) -> str:
    return resolve_dataset(name).daflow_name


def dataset_root(data_root: str | Path, dataset: str | DatasetSpec) -> Path:
    return Path(data_root).expanduser().resolve() / resolve_dataset(dataset).folder


def pose_dir(data_root: str | Path, dataset: str | DatasetSpec, split: str) -> Path:
    return dataset_root(data_root, dataset) / "pose" / split


def video_dir(data_root: str | Path, dataset: str | DatasetSpec, split: str) -> Path:
    rel = "train/images" if split == "train" else "test/frames"
    return dataset_root(data_root, dataset) / rel


def as_dir_string(path: str | Path) -> str:
    text = str(path)
    return text if text.endswith(os.sep) else text + os.sep


def run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_run_dir(output_root: str | Path, dataset: str | DatasetSpec, run_name: str | None = None) -> Path:
    name = run_name or run_timestamp()
    run_dir = Path(output_root).expanduser().resolve() / dataset_slug(dataset) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_data_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool = True,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        worker_init_fn=seed_worker,
        generator=generator,
    )


class BasePoseDataset(Dataset):
    def __init__(self, dataset: str | DatasetSpec) -> None:
        super().__init__()
        self.spec = resolve_dataset(dataset)


def video_id_from_pose_path(path: str | Path) -> str:
    return Path(path).name.replace("_alphapose_tracked_person.json", "")


def is_shanghaitech_hr_skip(video_id: str) -> bool:
    try:
        scene_text, clip_text = video_id.split("_")[:2]
        return (int(scene_text), int(clip_text)) in SHANGHAITECH_HR_SKIP
    except (ValueError, IndexError):
        return False


def split_name_from_path(path: str | Path) -> str:
    parts = [part.lower() for part in Path(path).parts]
    if "test" in parts or "testing" in parts:
        return "test"
    if "validation" in parts or "val" in parts:
        return "validation"
    return "train"


def normalize_ubnormal_clip_name(name: str | Path) -> str:
    base = os.path.basename(str(name)).strip()
    base = re.sub(r"\.(json|txt|npy)$", "", base)
    match = re.match(r"^(abnormal|normal)_scene_(\d+)_scenario(.+?)(?:_alphapose.*|_tracks.*|_annotations.*)?$", base)
    if not match:
        return base
    clip_type, scene_id, scenario = match.groups()
    return f"{clip_type}_scene_{int(scene_id)}_scenario{scenario}"


def hr_ubnormal_mask_split(split: str) -> str:
    split_text = split.lower()
    if split_text in {"test", "testing"}:
        return "testing"
    if split_text in {"validation", "val", "validating"}:
        return "validating"
    raise ValueError(f"HR-UBnormal masks are only defined for validation/test splits, got {split!r}.")


def hr_ubnormal_mask_dir(data_root: str | Path, split: str) -> Path:
    return (
        Path(data_root).expanduser().resolve()
        / "UBnormal"
        / "hr_bool_masks"
        / hr_ubnormal_mask_split(split)
        / "test_frame_mask"
    )


def _hr_ubnormal_manifest_candidates(data_root: str | Path, split: str) -> List[Path]:
    mask_dir = hr_ubnormal_mask_dir(data_root, split)
    return [
        mask_dir / "manifest.csv",
        mask_dir.parent / "manifest.csv",
        Path(data_root).expanduser().resolve() / "UBnormal" / "hr_bool_masks" / hr_ubnormal_mask_split(split) / "manifest.csv",
    ]


def _resolve_manifest_mask_path(raw_path: str, manifest_path: Path, mask_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    candidates = [
        manifest_path.parent / path,
        mask_dir / path,
        Path(mask_dir.parents[2]) / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_hr_ubnormal_mask_manifest(data_root: str | Path, split: str) -> Dict[str, Path]:
    mask_dir = hr_ubnormal_mask_dir(data_root, split)
    for manifest_path in _hr_ubnormal_manifest_candidates(data_root, split):
        if not manifest_path.exists():
            continue
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            rows = csv.DictReader(handle)
            if not rows.fieldnames:
                raise ValueError(f"HR-UBnormal manifest has no header: {manifest_path}")
            video_col = next((name for name in ("video_id", "clip_id", "clip", "name") if name in rows.fieldnames), None)
            mask_col = next((name for name in ("mask_path", "mask", "path", "file", "filename") if name in rows.fieldnames), None)
            if video_col is None or mask_col is None:
                raise ValueError(
                    "HR-UBnormal manifest must contain a video column "
                    "(video_id/clip_id/clip/name) and a mask column (mask_path/mask/path/file/filename): "
                    f"{manifest_path}"
                )
            mapping: Dict[str, Path] = {}
            for row in rows:
                raw_video_id = (row.get(video_col) or "").strip()
                raw_mask = (row.get(mask_col) or "").strip()
                if not raw_video_id or not raw_mask:
                    continue
                mask_path = _resolve_manifest_mask_path(raw_mask, manifest_path, mask_dir)
                mapping[raw_video_id] = mask_path
                mapping[normalize_ubnormal_clip_name(raw_video_id)] = mask_path
            return mapping
    return {}


def load_hr_ubnormal_eval_mask(
    data_root: str | Path,
    split: str,
    video_id: str | Path,
    n_frames: int,
) -> np.ndarray:
    mask_dir = hr_ubnormal_mask_dir(data_root, split)
    raw_video_id = Path(str(video_id)).name
    normalized_id = normalize_ubnormal_clip_name(raw_video_id)
    direct_candidates = [
        mask_dir / f"{raw_video_id}.npy",
        mask_dir / f"{normalized_id}.npy",
    ]
    manifest = _load_hr_ubnormal_mask_manifest(data_root, split)
    manifest_candidate = manifest.get(raw_video_id) or manifest.get(normalized_id)
    candidates = ([manifest_candidate] if manifest_candidate is not None else []) + direct_candidates
    for candidate in candidates:
        if candidate.exists():
            mask = np.load(candidate).astype(bool).reshape(-1)
            if len(mask) != n_frames:
                raise ValueError(
                    "HR-UBnormal mask length mismatch for "
                    f"{raw_video_id}: mask={len(mask)} frames, score/label={n_frames} frames, path={candidate}"
                )
            return mask

    expected = ", ".join(str(path) for path in direct_candidates)
    manifests = ", ".join(str(path) for path in _hr_ubnormal_manifest_candidates(data_root, split))
    raise FileNotFoundError(
        "Official HR-UBnormal frame mask was not found. "
        f"Video={raw_video_id}, split={split}. Expected exact mask at one of [{expected}] "
        f"or a manifest at one of [{manifests}]."
    )


def apply_hr_ubnormal_eval_mask(
    data_root: str | Path,
    split: str,
    video_id: str | Path,
    labels: np.ndarray,
    scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(labels) != len(scores):
        raise ValueError(
            f"Cannot apply HR-UBnormal mask to {video_id}: label length {len(labels)} != score length {len(scores)}."
        )
    mask = load_hr_ubnormal_eval_mask(data_root, split, video_id, len(labels))
    return labels[mask], scores[mask]


def is_ubnormal_family(dataset: str | DatasetSpec) -> bool:
    return resolve_dataset(dataset).key in UBNORMAL_KEYS


def filter_pose_files_for_protocol(
    files: Sequence[Path],
    dataset: str | DatasetSpec,
    data_root: str | Path,
    split: str,
) -> List[Path]:
    spec = resolve_dataset(dataset)
    filtered = list(files)
    if spec.key == "shanghaitech_hr" and split == "test":
        filtered = [path for path in filtered if not is_shanghaitech_hr_skip(video_id_from_pose_path(path))]
    return filtered


def pose_files(
    data_root: str | Path,
    dataset: str | DatasetSpec,
    split: str,
    max_files: Optional[int] = None,
) -> List[Path]:
    files = sorted(pose_dir(data_root, dataset, split).glob("*_alphapose_tracked_person.json"))
    files = filter_pose_files_for_protocol(files, dataset, data_root, split)
    if max_files is not None:
        files = files[:max_files]
    return files


def list_pose_json_names(
    pose_root: str | Path,
    dataset: str | DatasetSpec,
    data_root: str | Path,
    split: str | None = None,
    specific_clip: int | None = None,
) -> List[str]:
    root = Path(pose_root)
    split = split or split_name_from_path(root)
    files = sorted(root.glob("*_alphapose_tracked_person.json"))
    files = filter_pose_files_for_protocol(files, dataset, data_root, split)
    if specific_clip is not None:
        files = [files[specific_clip]]
    return [path.name for path in files]


def keypoints17_to_openpose18(keypoints: np.ndarray) -> np.ndarray:
    if keypoints.shape[-2] != 17:
        return keypoints
    neck = 0.5 * (keypoints[..., 5, :] + keypoints[..., 6, :])
    keypoints = np.concatenate([keypoints, neck[..., None, :]], axis=-2)
    order = np.asarray([0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3])
    return keypoints[..., order, :]


def _coerce_keypoints(values: Sequence[float], num_joints: int) -> Tuple[np.ndarray, np.ndarray]:
    keypoints = np.asarray(values, dtype=np.float32).reshape(-1, 3)
    if keypoints.shape[0] == 17 and num_joints == 18:
        keypoints = keypoints17_to_openpose18(keypoints)
    coords = np.zeros((num_joints, 2), dtype=np.float32)
    conf = np.zeros((num_joints,), dtype=np.float32)
    count = min(num_joints, keypoints.shape[0])
    coords[:count] = keypoints[:count, :2]
    conf[:count] = keypoints[:count, 2]
    return coords, conf


def load_tracks(path: str | Path, num_joints: int = 18) -> Tuple[List[Dict[str, np.ndarray]], int]:
    with Path(path).open("r") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} is not a tracked-person pose JSON dictionary.")

    tracks: List[Dict[str, np.ndarray]] = []
    global_max = -1
    for track_id, frames_dict in raw.items():
        items = []
        for frame_text, rec in frames_dict.items():
            if "keypoints" not in rec:
                continue
            frame = int(frame_text)
            coords, conf = _coerce_keypoints(rec["keypoints"], num_joints)
            score = float(rec.get("scores", float(conf.mean())))
            items.append((frame, coords, conf, score))
            global_max = max(global_max, frame)
        if not items:
            continue
        items.sort(key=lambda item: item[0])
        tracks.append(
            {
                "track_id": str(track_id),
                "frames": np.asarray([item[0] for item in items], dtype=np.int64),
                "coords": np.stack([item[1] for item in items], axis=0),
                "conf": np.stack([item[2] for item in items], axis=0),
                "scores": np.asarray([item[3] for item in items], dtype=np.float32),
            }
        )
    return tracks, global_max + 1


def read_raw_frame_labels(data_root: str | Path, dataset: str | DatasetSpec, video_id: str) -> np.ndarray:
    spec = resolve_dataset(dataset)
    root = dataset_root(data_root, spec)
    if spec.key in ("shanghaitech", "shanghaitech_hr"):
        path = root / "gt" / "test_frame_mask" / f"{video_id}.npy"
        labels = np.load(path).astype(np.float32)
        labels = (labels > 0.5).astype(np.uint8)
    elif spec.key in UBNORMAL_KEYS:
        path = root / "gt" / f"{video_id}_tracks.txt"
        with path.open("rb") as f:
            sig = f.read(6)
        if sig.startswith(b"\x93NUMPY"):
            normal_mask = np.load(path).astype(np.float32)
            labels = (normal_mask <= 0.5).astype(np.uint8)
        else:
            intervals = []
            max_end = 0
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    continue
                value, start, end = int(float(parts[0])), int(float(parts[1])), int(float(parts[2]))
                intervals.append((value, start, end))
                max_end = max(max_end, end)
            labels = np.zeros((max_end + 1,), dtype=np.uint8)
            for value, start, end in intervals:
                if value:
                    labels[start : end + 1] = 1
    else:
        raise ValueError(f"Labels are not implemented for {spec.key}.")
    return labels.astype(np.uint8)


def load_frame_labels(data_root: str | Path, dataset: str | DatasetSpec, video_id: str, n_frames: int) -> np.ndarray:
    try:
        labels = read_raw_frame_labels(data_root, dataset, video_id)
    except FileNotFoundError:
        spec = resolve_dataset(dataset)
        if spec.key not in UBNORMAL_KEYS:
            raise
        labels = np.zeros((n_frames,), dtype=np.uint8)
    if len(labels) < n_frames:
        pad = np.zeros((n_frames - len(labels),), dtype=np.uint8)
        labels = np.concatenate([labels, pad], axis=0)
    elif len(labels) > n_frames:
        labels = labels[:n_frames]
    return labels.astype(np.uint8)


def aggregate_frame_scores(
    meta: Sequence[Tuple[str, int, str]],
    window_scores: np.ndarray,
    frame_counts: Dict[str, int],
    fill_value: Optional[float] = None,
    reduce: str = "min",
) -> Dict[str, np.ndarray]:
    scores = {
        video_id: np.full((count,), np.nan, dtype=np.float32)
        for video_id, count in frame_counts.items()
    }
    for (video_id, frame_idx, _track_id), score in zip(meta, window_scores):
        if video_id not in scores or frame_idx < 0 or frame_idx >= len(scores[video_id]):
            continue
        current = scores[video_id][frame_idx]
        if np.isnan(current):
            scores[video_id][frame_idx] = float(score)
        elif reduce == "min" and score < current:
            scores[video_id][frame_idx] = float(score)
        elif reduce == "max" and score > current:
            scores[video_id][frame_idx] = float(score)

    finite_chunks = [v[np.isfinite(v)] for v in scores.values() if np.isfinite(v).any()]
    finite_values = np.concatenate(finite_chunks) if finite_chunks else np.empty((0,))
    if fill_value is None:
        if len(finite_values) == 0:
            fill_value = 0.0
        elif reduce == "min":
            fill_value = float(finite_values.max())
        else:
            fill_value = float(finite_values.min())
    for video_id, values in scores.items():
        values[~np.isfinite(values)] = fill_value
        scores[video_id] = values
    return scores


def labels_and_scores(
    data_root: str | Path,
    dataset: str | DatasetSpec,
    frame_scores: Dict[str, np.ndarray],
    split: str = "test",
) -> Tuple[np.ndarray, np.ndarray]:
    _video_ids, labels, scores = labels_scores_by_video(data_root, dataset, frame_scores, split=split)
    return np.concatenate(labels), np.concatenate(scores)


def labels_scores_by_video(
    data_root: str | Path,
    dataset: str | DatasetSpec,
    frame_scores: Dict[str, np.ndarray],
    split: str = "test",
) -> Tuple[List[str], List[np.ndarray], List[np.ndarray]]:
    spec = resolve_dataset(dataset)
    video_ids: List[str] = []
    labels: List[np.ndarray] = []
    scores: List[np.ndarray] = []
    for video_id in sorted(frame_scores):
        score = frame_scores[video_id]
        label = load_frame_labels(data_root, spec, video_id, len(score))
        if spec.key == "hr_ubnormal":
            label, score = apply_hr_ubnormal_eval_mask(data_root, split, video_id, label, score)
            if len(label) == 0:
                continue
        video_ids.append(video_id)
        labels.append(label)
        scores.append(score.astype(np.float32))
    return video_ids, labels, scores
