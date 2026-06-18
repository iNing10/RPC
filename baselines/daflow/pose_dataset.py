"""DA-Flow pose-window dataset built on the shared RPC-VAD data layer."""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from common.data import (
    BasePoseDataset,
    DEFAULT_DATA_ROOT,
    aggregate_frame_scores,
    dataset_config,
    labels_scores_by_video,
    load_tracks,
    pose_files,
    read_raw_frame_labels,
    video_id_from_pose_path,
)


def normalize_segment(
    coords: np.ndarray,
    vid_res: Tuple[int, int],
    eps: float = 1e-6,
) -> Optional[np.ndarray]:
    """Normalize one pose segment to [C, T, V]."""
    if not np.isfinite(coords).all():
        return None
    norm = np.asarray(vid_res, dtype=np.float32)
    normalized = coords / norm[None, None, :]
    center = normalized.mean(axis=(0, 1), keepdims=True)
    y_std = normalized[..., 1].std()
    scale = float(max(y_std, eps))
    normalized = (normalized - center) / scale
    return normalized.transpose(2, 0, 1).astype(np.float32)


def is_continuous_enough(frames: np.ndarray, start_key: int, window: int, missing_tolerance: int) -> bool:
    expected = set(range(int(start_key), int(start_key) + window))
    actual = set(int(v) for v in frames)
    return len(expected.intersection(actual)) >= window - missing_tolerance


def iter_track_windows(
    tracks: Sequence[Dict[str, np.ndarray]],
    window: int,
    stride: int,
    vid_res: Tuple[int, int],
    missing_tolerance: int = 2,
    include_conf: bool = False,
    segment_score_threshold: float = 0.0,
) -> Iterator[Tuple[np.ndarray, ...]]:
    for track in tracks:
        frames = track["frames"]
        coords = track["coords"]
        confs = track["conf"]
        scores = track["scores"]
        if len(frames) < window:
            continue
        num_segments = int(np.ceil((len(frames) - window) / stride))
        if num_segments <= 0 and len(frames) >= window:
            num_segments = 1
        for seg_idx in range(num_segments):
            start = seg_idx * stride
            end = start + window
            if end > len(frames):
                continue
            if segment_score_threshold > 0.0 and float(scores[start:end].mean()) <= segment_score_threshold:
                continue
            start_key = int(frames[start])
            if not is_continuous_enough(frames, start_key, window, missing_tolerance):
                continue
            segment = normalize_segment(coords[start:end], vid_res=vid_res)
            if segment is None:
                continue
            frame_idx = start_key + int(window / 2)
            if include_conf:
                yield segment, confs[start:end].astype(np.float32), frame_idx, str(track["track_id"])
            else:
                yield segment, frame_idx, str(track["track_id"])


def collect_windows(
    files: Sequence,
    window: int,
    stride: int,
    num_joints: int = 18,
    vid_res: Tuple[int, int] = (856, 480),
    missing_tolerance: int = 2,
    max_samples: Optional[int] = None,
    seed: int = 0,
    return_conf: bool = False,
    segment_score_threshold: float = 0.0,
) -> Tuple[np.ndarray, ...]:
    samples: List[np.ndarray] = []
    conf_samples: List[np.ndarray] = []
    meta: List[Tuple[str, int, str]] = []
    frame_counts: Dict[str, int] = {}

    for path in files:
        video_id = video_id_from_pose_path(path)
        tracks, n_frames = load_tracks(path, num_joints=num_joints)
        frame_counts[video_id] = n_frames
        for item in iter_track_windows(
            tracks,
            window=window,
            stride=stride,
            vid_res=vid_res,
            missing_tolerance=missing_tolerance,
            include_conf=return_conf,
            segment_score_threshold=segment_score_threshold,
        ):
            if return_conf:
                segment, conf_segment, frame_idx, track_id = item
                conf_samples.append(conf_segment)
            else:
                segment, frame_idx, track_id = item
            samples.append(segment)
            meta.append((video_id, int(frame_idx), str(track_id)))

    if not samples:
        empty = np.empty((0, 2, window, num_joints), dtype=np.float32)
        if return_conf:
            empty_conf = np.empty((0, window, num_joints), dtype=np.float32)
            return empty, empty_conf, meta, frame_counts
        return empty, meta, frame_counts

    array = np.stack(samples, axis=0).astype(np.float32)
    conf_array = np.stack(conf_samples, axis=0).astype(np.float32) if return_conf else None
    if max_samples is not None and len(array) > max_samples:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(array), size=max_samples, replace=False))
        array = array[indices]
        if return_conf and conf_array is not None:
            conf_array = conf_array[indices]
        meta = [meta[i] for i in indices]
    if return_conf:
        return array, conf_array, meta, frame_counts
    return array, meta, frame_counts


class PoseWindowDataset(BasePoseDataset):
    def __init__(
        self,
        files: Sequence,
        window: int,
        stride: int,
        num_joints: int = 18,
        vid_res: Tuple[int, int] = (856, 480),
        missing_tolerance: int = 2,
        max_samples: Optional[int] = None,
        seed: int = 0,
        num_transform: int = 1,
        segment_score_threshold: float = 0.0,
        dataset_name: str = "shanghaitech",
    ) -> None:
        super().__init__(dataset_name)
        self.samples, self.meta, self.frame_counts = collect_windows(
            files,
            window=window,
            stride=stride,
            num_joints=num_joints,
            vid_res=vid_res,
            missing_tolerance=missing_tolerance,
            max_samples=max_samples,
            seed=seed,
            segment_score_threshold=segment_score_threshold,
        )
        if len(self.samples) == 0:
            raise RuntimeError("No pose windows were built from the requested files.")
        if num_transform >= 2:
            flipped = self.samples.copy()
            flipped[:, 0] *= -1.0
            self.samples = np.concatenate([self.samples, flipped], axis=0)
            self.meta = self.meta + self.meta

    def __len__(self) -> int:
        return int(self.samples.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(self.samples[index])
