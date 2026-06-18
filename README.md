# RPC: Reliability-Aware Prototype Calibration for Frozen Pose-Flow Video Anomaly Detection

This repository is the official implementation of RPC for frozen pose-flow video anomaly detection backbones.
It keeps only the paths needed to train/evaluate the two baselines and run RPC on four datasets:

- ShanghaiTech
- ShanghaiTech-HR
- UBnormal
- UBnormal-HR

## Environment

```bash
conda create -n rpc-vad python=3.12 -y
conda activate rpc-vad
pip install -r requirements.txt
```

## Datasets

The prepared datasets are available from [STG-NF](https://github.com/orhir/STG-NF). Place or symlink them under `data/`.

Expected layout:

```text
data/
  ShanghaiTech/
    pose/
      train/
      test/
    gt/
      test_frame_mask/
  UBnormal/
    pose/
      train/
      validation/
      test/
    gt/
```

## Configs

| Backbone | Dataset | Baseline config | RPC config |
|---|---|---|---|
| STG-NF | ShanghaiTech | `configs/stgnf/shanghaitech.json` | `configs/rpc/stgnf_shanghaitech.json` |
| STG-NF | ShanghaiTech-HR | `configs/stgnf/shanghaitech_hr.json` | `configs/rpc/stgnf_shanghaitech_hr.json` |
| STG-NF | UBnormal | `configs/stgnf/ubnormal.json` | `configs/rpc/stgnf_ubnormal.json` |
| STG-NF | UBnormal-HR | `configs/stgnf/ubnormal_hr.json` | `configs/rpc/stgnf_ubnormal_hr.json` |
| DA-Flow | ShanghaiTech | `configs/daflow/shanghaitech.json` | `configs/rpc/daflow_shanghaitech.json` |
| DA-Flow | ShanghaiTech-HR | `configs/daflow/shanghaitech_hr.json` | `configs/rpc/daflow_shanghaitech_hr.json` |
| DA-Flow | UBnormal | `configs/daflow/ubnormal.json` | `configs/rpc/daflow_ubnormal.json` |
| DA-Flow | UBnormal-HR | `configs/daflow/ubnormal_hr.json` | `configs/rpc/daflow_ubnormal_hr.json` |

## Test Baseline Checkpoints

Run any baseline config directly:

```bash
python run.py --config configs/stgnf/shanghaitech.json
python run.py --config configs/daflow/ubnormal_hr.json
```

## Test RPC

Run any RPC config directly:

```bash
python run.py --config configs/rpc/stgnf_shanghaitech.json
python run.py --config configs/rpc/daflow_ubnormal.json
```

RPC writes JSON outputs to `eval_outputs/rpc/`.

## Train

Clear the config checkpoint path from the command line to train from scratch.

STG-NF:

```bash
python run.py --config configs/stgnf/shanghaitech.json --checkpoint '' --exp_dir train_outputs/stgnf
```

DA-Flow:

```bash
python run.py --config configs/daflow/shanghaitech.json --checkpoint '' --output-dir train_outputs/daflow
```

## Acknowledgements

Thanks to [STG-NF](https://github.com/orhir/STG-NF).
