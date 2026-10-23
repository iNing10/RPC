"""MSAD-HR tracked-pose preprocessing and original-timeline evaluation."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import Dataset

from common.data import keypoints17_to_openpose18

WINDOW = 24
MSAD_HR_VIDEOS = frozenset(
    f"{category}_{index}"
    for category, count in (
        ("Assault", 15),
        ("Fighting", 13),
        ("People_falling", 43),
        ("Robbery", 29),
        ("Shooting", 17),
        ("Traffic_accident", 39),
        ("Vandalism", 12),
    )
    for index in range(1, count + 1)
)


def interpolate_runs(frames, keypoints, max_missing=24):
    if not len(frames):
        return []
    if np.any(np.diff(frames) <= 0):
        raise ValueError("Track frames must be unique and increasing")
    runs = []
    breaks = np.flatnonzero(np.diff(frames) - 1 > max_missing) + 1
    for indices in np.split(np.arange(len(frames)), breaks):
        observed, poses = frames[indices], keypoints[indices]
        timeline = np.arange(observed[0], observed[-1] + 1, dtype=np.int64)
        right = np.searchsorted(observed, timeline).clip(0, len(observed) - 1)
        left = np.maximum(right - 1, 0)
        denominator = observed[right] - observed[left]
        fraction = np.divide(timeline - observed[left], denominator,
                             out=np.zeros(len(timeline), dtype=np.float32), where=denominator != 0)
        filled = poses[left] + fraction[:, None, None] * (poses[right] - poses[left])
        filled[observed - observed[0]] = poses
        runs.append((timeline, filled.astype(np.float32)))
    return runs


def load_runs(path, train, confidence_threshold=0.4, max_missing=24):
    tracks = json.loads(Path(path).read_text())
    if not isinstance(tracks, dict):
        raise ValueError(f"Expected tracked-person JSON: {path}")
    runs = []
    maximum_frame = -1
    for records in tracks.values():
        observations = []
        for frame_text, record in records.items():
            if "keypoints" not in record:
                continue
            frame = int(frame_text)
            if frame < 0:
                raise ValueError(f"Negative frame index in {path}")
            maximum_frame = max(maximum_frame, frame)
            pose = np.asarray(record["keypoints"], dtype=np.float32).reshape(-1, 3)
            if pose.shape != (17, 3):
                raise ValueError(f"Expected COCO17 keypoints in {path}")
            if not np.isfinite(pose).all() or (train and float(pose[:, 2].mean()) < confidence_threshold):
                continue
            pose[:, 2] = pose[:, 2].clip(0, 1)
            observations.append((frame, pose))
        if observations:
            observations.sort(key=lambda item: item[0])
            runs.extend(interpolate_runs(
                np.asarray([item[0] for item in observations], dtype=np.int64),
                np.stack([item[1] for item in observations]), max_missing=max_missing))
    return runs, maximum_frame


def normalize_windows(pose, resolution):
    coords = pose[..., :2] / np.asarray(resolution, dtype=np.float32)
    scale = coords[..., 1].std(axis=(1, 2), keepdims=True).clip(1e-6)
    coords = (coords - coords.mean(axis=(1, 2), keepdims=True)) / scale[..., None]
    result = np.concatenate([coords, pose[..., 2:3]], axis=-1).transpose(0, 3, 1, 2)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite normalized skeleton window")
    return result.astype(np.float32)


def prepare_split(data, prepared, split, *, window=24, stride=None, confidence_threshold=0.4, max_missing=24):
    data, prepared = Path(data), Path(prepared)
    prepared.mkdir(parents=True, exist_ok=True)
    with (prepared / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _prepare_split(data, prepared, split, window, stride, confidence_threshold, max_missing)


def _prepare_split(data, prepared, split, window, stride, confidence_threshold, max_missing):
    stride = stride if stride is not None else (6 if split == "train" else 1)
    if window < 1 or stride < 1 or max_missing < 0:
        raise ValueError("Window/stride must be positive and interpolation gap nonnegative")
    catalog = data / "MSAD_evaluation_tools" / f"{split}_videos.csv"
    with catalog.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 360 or len({row["video"] for row in rows}) != 360:
        raise ValueError(f"Expected 360 unique {split} videos")
    digest = hashlib.sha256(catalog.read_bytes() + b"msad-hr-coco17-conf04-gap24-v1")
    digest.update(json.dumps([window, stride, confidence_threshold, max_missing]).encode())
    for row in rows:
        digest.update(row["pose_relative_path"].encode())
        with (data / "pose" / row["pose_relative_path"]).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    fingerprint = digest.hexdigest()
    target = prepared / split
    marker = target / "complete.json"
    if marker.exists():
        if json.loads(marker.read_text()).get("fingerprint") != fingerprint:
            raise ValueError(f"Incompatible preprocessing cache: {target}; choose a new --prepared-dir")
        if all((target / name).is_file() for name in ("pose.npy", "videos.npy", "centers.npy", "catalog.json")):
            return
        raise ValueError(f"Incomplete preprocessing cache: {target}; choose a new --prepared-dir")
    target.mkdir(parents=True, exist_ok=True)
    counts = []
    for video_index, row in enumerate(rows):
        runs, maximum = load_runs(data / "pose" / row["pose_relative_path"], split == "train",
                                 confidence_threshold, max_missing)
        count = int(row["num_frames"])
        resolution = [int(row["width"]), int(row["height"])]
        if count <= 0 or min(resolution) <= 0 or (split == "test" and maximum >= count):
            raise ValueError(f"Invalid video timeline/resolution: {row['video']}")
        parts, centers = [], []
        for frames, keypoints in runs:
            starts = np.arange(0, max(0, len(frames) - window + 1), stride)
            pose18 = keypoints17_to_openpose18(keypoints)
            for offset in range(0, len(starts), 4096):
                batch = starts[offset:offset + 4096, None] + np.arange(window)[None, :]
                parts.append(normalize_windows(pose18[batch], resolution))
                centers.append(frames[batch[:, 0]] + window // 2)
        pose = np.concatenate(parts) if parts else np.empty((0, 3, window, 18), dtype=np.float32)
        center = np.concatenate(centers).astype(np.int32) if centers else np.empty(0, dtype=np.int32)
        np.savez(target / f"{video_index:03d}.npz", pose=pose, centers=center)
        counts.append(len(pose))
        if (video_index + 1) % 30 == 0:
            print(f"{split}: prepared {video_index + 1}/360 videos", flush=True)
    total = sum(counts)
    if not total:
        raise ValueError(f"No complete {split} pose windows")
    arrays = {
        "pose": np.lib.format.open_memmap(target / "pose.npy", mode="w+", dtype=np.float32,
                                          shape=(total, 3, window, 18)),
        "videos": np.lib.format.open_memmap(target / "videos.npy", mode="w+", dtype=np.int32, shape=(total,)),
        "centers": np.lib.format.open_memmap(target / "centers.npy", mode="w+", dtype=np.int32, shape=(total,)),
    }
    cursor = 0
    for video_index, count in enumerate(counts):
        shard = target / f"{video_index:03d}.npz"
        with np.load(shard) as archive:
            end = cursor + count
            arrays["pose"][cursor:end] = archive["pose"]
            arrays["videos"][cursor:end] = video_index
            arrays["centers"][cursor:end] = archive["centers"]
            cursor = end
        shard.unlink()
    for array in arrays.values():
        array.flush()
    (target / "catalog.json").write_text(json.dumps(rows, indent=2) + "\n")
    marker.write_text(json.dumps({"fingerprint": fingerprint, "windows": total}) + "\n")


class Windows(Dataset):
    def __init__(self, prepared, split, transforms=1):
        self.pose = np.load(prepared / split / "pose.npy", mmap_mode="r")
        self.transforms = transforms

    def __len__(self):
        return len(self.pose) * self.transforms

    def __getitem__(self, index):
        pose = np.array(self.pose[index % len(self.pose)], copy=True)
        if index >= len(self.pose):
            pose[0] *= -1
        return pose


class STGNFDataset(Windows):
    def __init__(self, prepared, split, transforms=1):
        super().__init__(prepared, split, transforms)
        self.metadata = np.column_stack([
            np.load(prepared / split / "videos.npy"),
            np.load(prepared / split / "centers.npy"),
        ])

    def __getitem__(self, index):
        pose = super().__getitem__(index)
        return [pose, index // len(self.pose), pose[2].mean(axis=1), np.float32(1)]


class DAFlowDataset(Windows):
    def __getitem__(self, index):
        return super().__getitem__(index)[:2]


def prepared_split(data_root, prepared, split, **kwargs):
    from common.data import dataset_root

    data = dataset_root(data_root, "msad-hr")
    prepared = Path(prepared)
    prepare_split(data, prepared, split, **kwargs)
    return prepared


def split_arrays(data_root, prepared, split, **kwargs):
    prepared = prepared_split(data_root, prepared, split, **kwargs)
    pose = np.load(prepared / split / "pose.npy", mmap_mode="c")
    return pose[:, :2], pose[:, 2]


class FrameScorer:
    def __init__(self, prepared, data, backbone, smooth_sigma=None):
        self.rows = json.loads((prepared / "test/catalog.json").read_text())
        self.counts = np.asarray([int(row["num_frames"]) for row in self.rows])
        self.offsets = np.concatenate([[0], np.cumsum(self.counts)])
        videos = np.load(prepared / "test/videos.npy")
        centers = np.load(prepared / "test/centers.npy")
        if (centers < 0).any() or (centers >= self.counts[videos]).any():
            raise ValueError("Window centers exceed the original video timeline")
        self.indices = self.offsets[videos] + centers
        tools = data / "MSAD_evaluation_tools"
        hr_names = MSAD_HR_VIDEOS
        if len(hr_names) != 168 or not hr_names <= {r["video"] for r in self.rows}:
            raise ValueError("Invalid reconstructed MSAD-HR protocol list")
        labels = [np.load(tools / "gt" / f"{row['video']}.npy") for row in self.rows]
        if any(y.shape != (count,) or not np.isin(y, [0, 1]).all()
               for y, count in zip(labels, self.counts)):
            raise ValueError("Labels must be binary and match each original video length")
        self.labels = np.concatenate(labels)
        self.hr = np.concatenate([np.full(count, row["video"] in hr_names, dtype=bool)
                                  for row, count in zip(self.rows, self.counts)])
        self.sigmas = range(1, smooth_sigma if smooth_sigma is not None else (7 if backbone == "stg-nf" else 8))

    def frames(self, anomaly):
        values = np.asarray(anomaly)
        if values.shape != self.indices.shape or not np.isfinite(values).all() or not len(values):
            raise ValueError("Expected one finite anomaly score per test window")
        scores = np.full(self.offsets[-1], -np.inf, dtype=np.float64)
        np.maximum.at(scores, self.indices, values)
        scores[~np.isfinite(scores)] = scores[np.isfinite(scores)].min()
        for start, stop in zip(self.offsets[:-1], self.offsets[1:]):
            for sigma in self.sigmas:
                scores[start:stop] = gaussian_filter1d(scores[start:stop], sigma=sigma)
        return scores

    def evaluate(self, anomaly):
        scores = self.frames(anomaly)[self.hr]
        labels = self.labels[self.hr]
        return {"auroc": float(roc_auc_score(labels, scores)),
                "ap": float(average_precision_score(labels, scores)),
                "num_frames": len(labels), "num_anomaly_frames": int(labels.sum())}

    def arrays(self, anomaly):
        frames = self.frames(anomaly)
        names, labels, scores = [], [], []
        for row, start, stop in zip(self.rows, self.offsets[:-1], self.offsets[1:]):
            if self.hr[start]:
                names.append(row["video"])
                labels.append(self.labels[start:stop])
                scores.append(frames[start:stop])
        return names, labels, scores


def frame_arrays(normality, data_root, prepared, backbone, smooth_sigma=None):
    from common.data import dataset_root

    scorer = FrameScorer(Path(prepared), dataset_root(data_root, "msad-hr"), backbone, smooth_sigma)
    return scorer.arrays(-np.asarray(normality))
