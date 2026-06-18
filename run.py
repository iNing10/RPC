#!/usr/bin/env python3
"""Unified entry point for RPC-VAD baselines and methods."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any


sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent
BASELINES = {
    "stg-nf": {
        "name": "STG-NF",
        "path": ROOT / "baselines" / "stgnf",
        "module": "train_eval",
        "entry": "main",
    },
    "daflow": {
        "name": "DA-Flow",
        "path": ROOT / "baselines" / "daflow",
        "module": "run_experiment",
        "entry": "main",
    },
    "rpc": {
        "name": "RPC post-hoc calibration",
        "path": ROOT / "methods",
        "module": "rpc",
        "entry": "main",
    },
}
STORE_FALSE_FLAGS = {
    "stg-nf": {"global_pose_segs"},
    "daflow": set(),
    "rpc": {"global_pose_segs"},
}
PATH_ARGS = {
    "checkpoint",
    "data-root",
    "data_dir",
    "dump_scores_dir",
    "exp_dir",
    "output-dir",
    "output-json",
    "pose_path_test",
    "pose_path_train",
    "pose_path_train_abnormal",
    "vid_path_test",
    "vid_path_train",
}
LOCAL_MODULE_PREFIXES = (
    "args",
    "dataset",
    "daflow_model",
    "pose_dataset",
    "rpc",
    "rpc_core",
    "run_experiment",
    "train_eval",
    "models",
    "utils",
)


def _dispatch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run baselines or RPC methods from one command.",
        add_help=False,
    )
    parser.add_argument("--baseline", choices=sorted(BASELINES), help="Baseline backend to run.")
    parser.add_argument("--config", type=str, help="JSON config file with baseline and backend args.")
    parser.add_argument("--list-baselines", action="store_true", help="List available backends.")
    parser.add_argument("--list-configs", action="store_true", help="List bundled config files.")
    parser.add_argument("--print-config", action="store_true", help="Print expanded backend command and exit.")
    return parser


def _print_main_help() -> None:
    names = ", ".join(f"{key} ({info['name']})" for key, info in BASELINES.items())
    print("Usage:")
    print("  python run.py --config configs/stgnf/shanghaitech.json")
    print("  python run.py --config configs/rpc/stgnf_shanghaitech.json")
    print("  python run.py --baseline stg-nf [STG-NF args]")
    print("  python run.py --baseline daflow [DA-Flow args]")
    print("  python run.py --baseline rpc [RPC args]")
    print()
    print(f"Available baselines: {names}")
    print()
    print("Use `python run.py --baseline stg-nf --help` or")
    print("`python run.py --baseline daflow --help` for backend-specific options.")
    print()
    print("Use `python run.py --list-configs` to show bundled fixed configs.")


def _list_configs() -> None:
    config_root = ROOT / "configs"
    for path in sorted(config_root.rglob("*.json")):
        rel = path.relative_to(ROOT)
        try:
            with path.open() as f:
                config = json.load(f)
            baseline = config.get("baseline", "")
            description = config.get("description", "")
        except (OSError, json.JSONDecodeError):
            baseline = ""
            description = ""
        suffix = f"  [{baseline}]" if baseline else ""
        print(f"{rel}{suffix}")
        if description:
            print(f"  {description}")


def _resolve_config_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _load_config(path_text: str) -> dict[str, Any]:
    path = _resolve_config_path(path_text)
    with path.open() as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    config["_config_path"] = str(path)
    return config


def _resolve_project_path(value: str) -> str:
    expanded = os.path.expandvars(value).replace("${PROJECT_ROOT}", str(ROOT))
    expanded = expanded.replace("${WORKSPACE_ROOT}", str(ROOT.parent))
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return str(path.resolve())


def _format_arg_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def _expand_dict_args(args: dict[str, Any], baseline: str) -> list[str]:
    expanded: list[str] = []
    store_false = STORE_FALSE_FLAGS.get(baseline, set())
    for key, value in args.items():
        if value is None:
            continue
        option = key if key.startswith("--") else f"--{key}"
        bare_key = option[2:]

        if bare_key in store_false:
            if value is False:
                expanded.append(option)
            continue
        if isinstance(value, bool):
            if value:
                expanded.append(option)
            continue

        if bare_key in PATH_ARGS and isinstance(value, str):
            value = _resolve_project_path(value)

        expanded.extend([option, _format_arg_value(value)])
    return expanded


def _expand_config_args(config: dict[str, Any], baseline: str) -> list[str]:
    raw_args = config.get("args", config.get("params", {}))
    if isinstance(raw_args, list):
        expanded = [str(item) for item in raw_args]
    elif isinstance(raw_args, dict):
        expanded = _expand_dict_args(raw_args, baseline)
    else:
        raise ValueError("Config field `args` must be an object or a list.")

    extra_args = config.get("extra_args", [])
    if extra_args:
        if not isinstance(extra_args, list):
            raise ValueError("Config field `extra_args` must be a list.")
        expanded.extend(str(item) for item in extra_args)
    return expanded


def _prepare_backend(path: Path) -> None:
    for name in list(sys.modules):
        if name in LOCAL_MODULE_PREFIXES or name.startswith("models.") or name.startswith("utils."):
            sys.modules.pop(name, None)

    backend_paths = {str(info["path"]) for info in BASELINES.values()}
    sys.path[:] = [entry for entry in sys.path if entry not in backend_paths]
    sys.path.insert(0, str(path))


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    has_baseline = any(arg == "--baseline" or arg.startswith("--baseline=") for arg in argv)
    has_config = any(arg == "--config" or arg.startswith("--config=") for arg in argv)
    if not argv or any(arg in ("-h", "--help") for arg in argv) and not has_baseline and not has_config:
        _print_main_help()
        return 0

    parser = _dispatch_parser()
    dispatch_args, backend_args = parser.parse_known_args(argv)
    if dispatch_args.list_baselines:
        for key, info in BASELINES.items():
            print(f"{key}\t{info['name']}")
        return 0
    if dispatch_args.list_configs:
        _list_configs()
        return 0

    config_args: list[str] = []
    config_baseline = None
    if dispatch_args.config:
        config = _load_config(dispatch_args.config)
        config_baseline = config.get("baseline", config.get("method"))
        if config_baseline is None:
            raise ValueError(f"Config is missing required `baseline` or `method`: {dispatch_args.config}")
        if config_baseline not in BASELINES:
            raise ValueError(f"Unknown baseline in config: {config_baseline!r}")
        if dispatch_args.baseline and dispatch_args.baseline != config_baseline:
            raise ValueError(
                f"Config baseline {config_baseline!r} does not match --baseline {dispatch_args.baseline!r}."
            )
        config_args = _expand_config_args(config, config_baseline)

    baseline = dispatch_args.baseline or config_baseline
    if baseline is None:
        _print_main_help()
        return 2

    backend_args = config_args + backend_args
    if dispatch_args.print_config:
        print(f"baseline: {baseline}")
        print("backend args:")
        print(" ".join(backend_args))
        return 0

    info = BASELINES[baseline]
    _prepare_backend(info["path"])
    module = importlib.import_module(info["module"])
    entry = getattr(module, info["entry"])
    result = entry(backend_args)
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
