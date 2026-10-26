"""
MCC5-THU 三轴 → RGB GASF/GADF 图像（逻辑与 1.4_generate_hust_gaf_rgb_images.py 一致）。

B 点（whole_file，论文 MCC5 统一口径）：按 train/val/test 分桶封顶 1050/225/225。
C 点（temporal 备用）：每类全局最多 1500 张。
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset_split import build_idx_to_split_whole_file  # noqa: E402
from utils.gaf_utils import gaf_transform  # noqa: E402
from utils.project_paths import (  # noqa: E402
    FILE_RANGES_MCC5,
    GAF_INPUT_MCC5_AXES,
    GAF_RGB_MCC5,
    MCC5_GAF_SIZE,
    MCC5_META_JSON,
    ensure_mcc5_dirs,
)

GAF_SIZE = MCC5_GAF_SIZE
MAX_IMAGES_PER_LABEL = 1500
SPLIT_CAPS = {"train": 1050, "val": 225, "test": 225}
KEEP_MOD = 8
KEEP_PATTERN = {0, 3, 6}
CLEAR_OUTPUT_DIRS = True
SIGNAL_GAIN = 3.0
SIGNED_POWER = 0.7
MATRIX_CONTRAST_STRETCH = True
STRETCH_LOW_PERC = 2.0
STRETCH_HIGH_PERC = 98.0


def resample_1d(signal: np.ndarray, out_len: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    in_len = signal.shape[0]
    if in_len == out_len:
        return signal
    x_old = np.arange(in_len, dtype=np.float32)
    x_new = np.linspace(0, in_len - 1, out_len, dtype=np.float32)
    return np.interp(x_new, x_old, signal).astype(np.float32)


def signed_power(x: np.ndarray, p: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if p == 1.0:
        return x
    return np.sign(x) * (np.abs(x) ** p)


def contrast_stretch_01(x01: np.ndarray, low_perc: float, high_perc: float) -> np.ndarray:
    lo, hi = np.percentile(x01, [low_perc, high_perc])
    if hi <= lo:
        return x01.astype(np.float32)
    y = np.clip(x01, lo, hi)
    return ((y - lo) / (hi - lo + 1e-8)).astype(np.float32)


def to_rgb_gaf(sig3: np.ndarray, method: str) -> np.ndarray:
    mats = []
    for ch in range(3):
        s = resample_1d(sig3[ch], GAF_SIZE)
        s = signed_power(s, SIGNED_POWER)
        if SIGNAL_GAIN != 1.0:
            s = np.clip(s * float(SIGNAL_GAIN), -1.0, 1.0)
        m = gaf_transform(s, method=method, normalize=False).astype(np.float32)
        m01 = (m + 1.0) * 0.5
        if MATRIX_CONTRAST_STRETCH:
            m01 = contrast_stretch_01(m01, STRETCH_LOW_PERC, STRETCH_HIGH_PERC)
        mats.append(m01)
    return np.clip(np.stack(mats, axis=-1), 0.0, 1.0)


def load_npy_dict(path: Path) -> dict:
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        return obj.item()
    if isinstance(obj, dict):
        return obj
    return obj.item()


def load_whole_file_idx_to_split() -> Dict[int, str]:
    meta = json.loads(MCC5_META_JSON.read_text(encoding="utf-8"))
    file_labels = {int(x["file_id"]): int(x["label"]) for x in meta.get("files", [])}
    ranges = []
    with FILE_RANGES_MCC5.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ranges.append((int(row["file_id"]), int(row["start_idx"]), int(row["end_idx"])))
    return build_idx_to_split_whole_file(ranges, file_labels)


def estimate_target(
    labels: np.ndarray,
    idx_to_split: Optional[Dict[int, str]] = None,
) -> int:
    kept_ratio = len(KEEP_PATTERN) / KEEP_MOD if KEEP_MOD and KEEP_PATTERN else 1.0
    n = labels.shape[0]
    if idx_to_split is None:
        _, cnts = np.unique(labels, return_counts=True)
        total = 0
        for c in cnts.tolist():
            if MAX_IMAGES_PER_LABEL is None:
                total += int(int(c) * kept_ratio)
            else:
                total += min(int(c), MAX_IMAGES_PER_LABEL)
        return total

    per_bucket: Dict[Tuple[int, str], int] = {}
    for idx in range(n):
        if KEEP_MOD and KEEP_PATTERN and (idx % KEEP_MOD) not in KEEP_PATTERN:
            continue
        lab = int(labels[idx])
        split = idx_to_split.get(idx, "train")
        key = (lab, split)
        cap = SPLIT_CAPS.get(split, 0)
        if per_bucket.get(key, 0) < cap:
            per_bucket[key] = per_bucket.get(key, 0) + 1
    return sum(per_bucket.values())


def generate_branch(input_npy: Path, out_dir: Path, exp_point: str = "b") -> None:
    data = load_npy_dict(input_npy)
    signals = np.asarray(data["signals"], dtype=np.float32)
    labels = np.asarray(data["labels"], dtype=int)

    use_split_cap = exp_point.lower() == "b"
    idx_to_split = load_whole_file_idx_to_split() if use_split_cap else None

    gasf_root = out_dir / "gasf"
    gadf_root = out_dir / "gadf"
    if CLEAR_OUTPUT_DIRS and out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    gasf_root.mkdir(parents=True, exist_ok=True)
    gadf_root.mkdir(parents=True, exist_ok=True)

    per_label: Dict[int, int] = {}
    per_split: Dict[Tuple[int, str], int] = {}
    target = estimate_target(labels, idx_to_split)
    mode = "whole_file split caps" if use_split_cap else "global per-label cap"
    print(f"[MCC5-GAF-RGB] exp_point={exp_point.upper()} mode={mode} target≈{target}")

    done = 0
    t0 = time.time()
    for idx in range(signals.shape[0]):
        if KEEP_MOD and KEEP_PATTERN and (idx % KEEP_MOD) not in KEEP_PATTERN:
            continue
        lab = int(labels[idx])

        if use_split_cap:
            split = idx_to_split.get(idx, "train")  # type: ignore[union-attr]
            key = (lab, split)
            cap = SPLIT_CAPS.get(split, 0)
            if per_split.get(key, 0) >= cap:
                continue
            per_split[key] = per_split.get(key, 0) + 1
        elif MAX_IMAGES_PER_LABEL is not None:
            c = per_label.get(lab, 0)
            if c >= MAX_IMAGES_PER_LABEL:
                continue
            per_label[lab] = c + 1

        rgb_gasf = to_rgb_gaf(signals[idx], method="s")
        rgb_gadf = to_rgb_gaf(signals[idx], method="d")

        (gasf_root / str(lab)).mkdir(parents=True, exist_ok=True)
        (gadf_root / str(lab)).mkdir(parents=True, exist_ok=True)
        gasf_path = gasf_root / str(lab) / f"gasf_rgb_{lab}_{idx:05d}.png"
        gadf_path = gadf_root / str(lab) / f"gadf_rgb_{lab}_{idx:05d}.png"
        plt.imsave(gasf_path, rgb_gasf, origin="lower")
        plt.imsave(gadf_path, rgb_gadf, origin="lower")

        done += 1
        if done % 50 == 0 or done == target:
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-6)
            eta = (target - done) / max(rate, 1e-6)
            print(f"[MCC5-GAF-RGB] {done}/{target} | {rate:.2f} img/s | ETA {eta/60:.1f} min")

    if use_split_cap:
        for lab in sorted({int(x) for x in labels.tolist()}):
            counts = {s: per_split.get((lab, s), 0) for s in SPLIT_CAPS}
            print(f"  label {lab}: train={counts['train']} val={counts['val']} test={counts['test']}")

    print(f"[MCC5-GAF-RGB] done -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MCC5 GAF-RGB images")
    parser.add_argument(
        "--exp-point",
        choices=("b", "c"),
        default="b",
        help="B=whole_file 分桶封顶（论文口径）；C=temporal 全局 1500/类",
    )
    args = parser.parse_args()

    ensure_mcc5_dirs()
    if not GAF_INPUT_MCC5_AXES.is_file():
        raise FileNotFoundError(f"请先运行 1.1_preprocess_mcc5_gaf.py: {GAF_INPUT_MCC5_AXES}")
    generate_branch(GAF_INPUT_MCC5_AXES, GAF_RGB_MCC5, exp_point=args.exp_point)


if __name__ == "__main__":
    main()
