"""MSAD-HR registry, data adapters, scoring, and cache contracts."""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.data import resolve_dataset
from common.msad import DAFlowDataset, FrameScorer, MSAD_HR_VIDEOS, STGNFDataset, Windows
from common.msad import interpolate_runs, load_runs, normalize_windows, prepare_split
from methods.rpc_core import CalibrationConfig, calibration_distances, diag_mahalanobis_distance


class MSADTests(unittest.TestCase):
    def test_all_bundled_configs_parse(self):
        script = """
import importlib
import json
from pathlib import Path
import run
for path in sorted(Path('configs').rglob('*.json')):
    config = json.loads(path.read_text())
    backend = config['baseline']
    run._prepare_backend(run.BASELINES[backend]['path'])
    argv = run._expand_config_args(config, backend)
    module = importlib.import_module(run.BASELINES[backend]['module'])
    if backend == 'rpc':
        module.build_parser(argv).parse_args(argv)
    elif backend == 'stg-nf':
        module.init_parser().parse_args(argv)
    else:
        module.parse_args(argv)
    print(path)
"""
        result = subprocess.run([sys.executable, "-B", "-c", script], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(result.stdout.splitlines()), len(list((ROOT / "configs").rglob("*.json"))))

    def test_shared_mahalanobis_matches_direct_formula(self):
        generator = torch.Generator().manual_seed(42)
        train = torch.randn(73, 9, generator=generator)
        test = torch.randn(31, 9, generator=generator)
        centers = train[:4].clone()
        assignment = torch.cdist(train, centers).argmin(1)
        variances = torch.stack([train[assignment == i].var(0, unbiased=False).clamp_min(1e-6)
                                 for i in range(len(centers))])
        config = CalibrationConfig(distance="diag_mahalanobis")
        expected = [((h[:, None] - centers[None]).square() / variances[None]).sum(2).sqrt().min(1).values
                    for h in (train, test)]
        for actual, reference in zip(calibration_distances(train, test, centers, config), expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        if torch.cuda.is_available():
            for actual, reference in zip(calibration_distances(train, test, centers, config,
                                                              distance_device="cuda:0"), expected):
                torch.testing.assert_close(actual, reference)
        weights = torch.rand(test.shape, generator=generator)
        weighted = ((test[:, None] - centers[None]).square() / variances[None] * weights[:, None]).sum(2).sqrt().min(1).values
        actual, _ = diag_mahalanobis_distance(test, centers, variances, weights, chunk_size=7)
        torch.testing.assert_close(actual, weighted, rtol=0, atol=0)

    def test_configs_use_shared_backends_and_parsers(self):
        import run

        self.assertNotIn("msad-hr", run.BASELINES)
        self.assertEqual(resolve_dataset("MSAD-HR").key, "msad_hr")
        for folder, baseline in (("stgnf", "stg-nf"), ("daflow", "daflow")):
            for kind in ("baseline", "rpc"):
                relative = f"configs/{folder}/msad_hr.json" if kind == "baseline" else f"configs/rpc/{folder}_msad_hr.json"
                config = json.loads((ROOT / relative).read_text())
                self.assertEqual(config["baseline"], baseline if kind == "baseline" else "rpc")
                self.assertEqual(resolve_dataset(config["args"]["dataset"]).key, "msad_hr")
                result = subprocess.run([sys.executable, "-B", "run.py", "--config", relative, "--help"],
                                        cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_gap_limit_and_no_extrapolation(self):
        frames = np.array([2, 27, 53])
        poses = np.zeros((3, 17, 3), dtype=np.float32)
        poses[1], poses[2] = 1, 2
        runs = interpolate_runs(frames, poses)
        self.assertEqual([len(run[0]) for run in runs], [26, 1])
        self.assertEqual(runs[0][0][0], 2)
        np.testing.assert_array_equal(runs[0][1][0], poses[0])
        np.testing.assert_array_equal(runs[0][1][-1], poses[1])
        np.testing.assert_allclose(runs[0][1][1], 0.04)
        with self.assertRaises(ValueError):
            interpolate_runs(np.array([2, 2]), poses[:2])

    def test_confidence_filter_precedes_interpolation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pose.json"
            pose = np.ones((17, 3), dtype=np.float32)
            records = {}
            for frame, confidence in [(0, 0.8), (1, 0.1), (2, 0.6)]:
                pose[:, 2] = confidence
                records[str(frame)] = {"keypoints": pose.flatten().tolist()}
            path.write_text(json.dumps({"person": records}))
            train, _ = load_runs(path, True)
            test, _ = load_runs(path, False)
            np.testing.assert_allclose(train[0][1][1, :, 2], 0.7)
            np.testing.assert_allclose(test[0][1][1, :, 2], 0.1)

    def test_normalization_preserves_confidence(self):
        pose = np.random.default_rng(42).random((3, 24, 18, 3), dtype=np.float32)
        normalized = normalize_windows(pose, (1920, 1080))
        self.assertEqual(normalized.shape, (3, 3, 24, 18))
        np.testing.assert_array_equal(normalized[:, 2], pose[..., 2])
        np.testing.assert_allclose(normalized[:, :2].mean((2, 3)), 0, atol=1e-6)

    def test_parallel_prepare_adapters_and_cache_invalidation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tools = root / "MSAD_evaluation_tools"
            tools.mkdir()
            (root / "pose").mkdir()
            records = {str(i): {"keypoints": [[float(i + j), float(j), 0.9] for j in range(17)]}
                       for i in range(30)}
            source = root / "pose/person.json"
            source.write_text(json.dumps({"1": records}))
            catalog = tools / "train_videos.csv"
            with catalog.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["video", "pose_relative_path", "num_frames", "width", "height"])
                writer.writeheader()
                writer.writerows({"video": f"v{i}", "pose_relative_path": "person.json",
                                  "num_frames": 30, "width": 100, "height": 100} for i in range(360))
            prepared = root / "prepared"
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(prepare_split, root, prepared, "train") for _ in range(2)]
                for future in futures:
                    future.result()
            dataset = Windows(prepared, "train", 2)
            self.assertEqual(len(dataset), 1440)
            np.testing.assert_array_equal(np.load(prepared / "train/centers.npy")[:2], [12, 18])
            np.testing.assert_array_equal(dataset[720][0], -dataset[0][0])
            np.testing.assert_array_equal(dataset[720][1:], dataset[0][1:])
            stgnf, daflow = STGNFDataset(prepared, "train", 2), DAFlowDataset(prepared, "train", 2)
            np.testing.assert_array_equal(stgnf[720][0][:2], daflow[720])
            self.assertEqual(stgnf[720][-1], 1)
            original = source.read_text()
            records["0"]["keypoints"][0][0] += 50
            source.write_text(json.dumps({"1": records}))
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                prepare_split(root, prepared, "train")
            source.write_text(original)
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                prepare_split(root, prepared, "train", stride=1)
            catalog.write_text(catalog.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                prepare_split(root, prepared, "train")

    def test_max_aggregation_global_fill_and_hr_mask(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            test = root / "prepared/test"
            tools = root / "data/MSAD_evaluation_tools"
            test.mkdir(parents=True)
            (tools / "gt").mkdir(parents=True)
            names = sorted(MSAD_HR_VIDEOS) + [f"v{i}" for i in range(168, 360)]
            rows = [{"video": name, "num_frames": 30} for name in names]
            (test / "catalog.json").write_text(json.dumps(rows))
            np.save(test / "videos.npy", [0, 0, 0, 359])
            np.save(test / "centers.npy", [12, 12, 13, 12])
            for row in rows:
                np.save(tools / "gt" / f"{row['video']}.npy", np.tile([0, 1], 15))
            scorer = FrameScorer(root / "prepared", root / "data", "stg-nf")
            scorer.sigmas = []
            frames = scorer.frames([1, 2, 3, -5])
            self.assertEqual(frames[12], 2)
            self.assertEqual(frames[13], 3)
            self.assertEqual(frames[0], -5)
            names, labels, scores = scorer.arrays([1, 2, 3, -5])
            self.assertEqual(len(names), 168)
            np.testing.assert_array_equal(np.concatenate(scores), frames[scorer.hr])
            np.testing.assert_array_equal(np.concatenate(labels), scorer.labels[scorer.hr])
            with self.assertRaises(ValueError):
                scorer.frames([np.nan] * 4)


if __name__ == "__main__":
    unittest.main()
