import csv
import os
import re
from pathlib import Path

import numpy as np


HR_UBNORMAL_DATASETS = {"HR-UBnormal", "UBnormal-HR"}
UBNORMAL_FAMILY_DATASETS = {"UBnormal", *HR_UBNORMAL_DATASETS}


def is_hr_ubnormal(dataset):
    return dataset in HR_UBNORMAL_DATASETS


def is_ubnormal_family(dataset):
    return dataset in UBNORMAL_FAMILY_DATASETS


def dataset_root_name(dataset):
    if is_ubnormal_family(dataset):
        return "UBnormal"
    return dataset


def normalize_ubnormal_clip_name(name):
    base = os.path.basename(str(name)).strip()
    base = re.sub(r"\.(json|txt|npy)$", "", base)
    match = re.match(r"^(abnormal|normal)_scene_(\d+)_scenario(.+?)(?:_alphapose.*|_tracks.*|_annotations.*)?$", base)
    if not match:
        return base
    clip_type, scene_id, scenario = match.groups()
    return f"{clip_type}_scene_{int(scene_id)}_scenario{scenario}"


def hr_ubnormal_mask_split(split):
    split_text = str(split).lower()
    if split_text in {"test", "testing"}:
        return "testing"
    if split_text in {"validation", "val", "validating"}:
        return "validating"
    raise ValueError(f"HR-UBnormal masks are only defined for validation/test splits, got {split!r}.")


def hr_ubnormal_mask_dir(data_dir, split):
    return Path(data_dir) / "UBnormal" / "hr_bool_masks" / hr_ubnormal_mask_split(split) / "test_frame_mask"


def _manifest_candidates(data_dir, split):
    mask_dir = hr_ubnormal_mask_dir(data_dir, split)
    return [
        mask_dir / "manifest.csv",
        mask_dir.parent / "manifest.csv",
        Path(data_dir) / "UBnormal" / "hr_bool_masks" / hr_ubnormal_mask_split(split) / "manifest.csv",
    ]


def _resolve_manifest_mask_path(raw_path, manifest_path, mask_dir):
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    candidates = [
        manifest_path.parent / path,
        mask_dir / path,
        mask_dir.parents[2] / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_mask_manifest(data_dir, split):
    mask_dir = hr_ubnormal_mask_dir(data_dir, split)
    for manifest_path in _manifest_candidates(data_dir, split):
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
            mapping = {}
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


def load_hr_ubnormal_eval_mask(data_dir, split, video_id, n_frames):
    mask_dir = hr_ubnormal_mask_dir(data_dir, split)
    raw_video_id = Path(str(video_id)).name
    normalized_id = normalize_ubnormal_clip_name(raw_video_id)
    direct_candidates = [
        mask_dir / f"{raw_video_id}.npy",
        mask_dir / f"{normalized_id}.npy",
    ]
    manifest = _load_mask_manifest(data_dir, split)
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
    manifests = ", ".join(str(path) for path in _manifest_candidates(data_dir, split))
    raise FileNotFoundError(
        "Official HR-UBnormal frame mask was not found. "
        f"Video={raw_video_id}, split={split}. Expected exact mask at one of [{expected}] "
        f"or a manifest at one of [{manifests}]."
    )


def apply_hr_ubnormal_eval_mask(data_dir, split, video_id, labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(labels) != len(scores):
        raise ValueError(
            f"Cannot apply HR-UBnormal mask to {video_id}: label length {len(labels)} != score length {len(scores)}."
        )
    mask = load_hr_ubnormal_eval_mask(data_dir, split, video_id, len(labels))
    return labels[mask], scores[mask]


def split_name_from_path(path):
    parts = [part.lower() for part in Path(path).parts]
    if "test" in parts or "testing" in parts:
        return "test"
    if "validation" in parts or "val" in parts:
        return "validation"
    return "train"


def filter_hr_ubnormal_files(dataset, data_dir, split, filenames):
    if not is_hr_ubnormal(dataset):
        return list(filenames)
    return list(filenames)
