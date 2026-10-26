"""
CWRU GAF v2 预处理：去均值 + 全局分位数缩放（与 HUST 一致，不再使用中值滤波残差）。

输入：processed_v2/final_dataset.csv（2048 点 + label）
输出：processed_v2/gaf_preprocessed_cwru.npy
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.preprocess import preprocess_gaf_1d  # noqa: E402
from utils.project_paths import (  # noqa: E402
    CWRU_SEGMENT_LENGTH,
    FINAL_DATASET_CWRU_V2,
    GAF_PREPROCESSED_CWRU_V2,
    ROBUST_PHIGH,
    ROBUST_PLOW,
    ensure_v2_dirs,
)

SEGMENT_LENGTH = CWRU_SEGMENT_LENGTH
RAW_DATA_PATH = FINAL_DATASET_CWRU_V2
OUTPUT_PATH = GAF_PREPROCESSED_CWRU_V2


def _load_final_dataset_csv(path: Path, segment_length: int) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import pandas as pd

        df = pd.read_csv(path, header=None)
        arr = df.to_numpy(dtype=np.float32)
    except Exception:
        arr = np.loadtxt(path, delimiter=",", dtype=np.float32)

    expected_cols = segment_length + 1
    if arr.shape[1] != expected_cols:
        raise ValueError(f"Expected {expected_cols} cols, got {arr.shape[1]}")
    signals = arr[:, :segment_length].astype(np.float32, copy=False)
    labels = arr[:, segment_length].astype(np.int64, copy=False)
    return signals, labels


def main() -> None:
    ensure_v2_dirs()
    signals, labels = _load_final_dataset_csv(RAW_DATA_PATH, SEGMENT_LENGTH)
    n, l = signals.shape
    if l != SEGMENT_LENGTH:
        raise RuntimeError(f"Unexpected segment length {l}, expected {SEGMENT_LENGTH}")

    scaled, meta = preprocess_gaf_1d(signals, plow=ROBUST_PLOW, phigh=ROBUST_PHIGH)
    meta["segment_length"] = int(SEGMENT_LENGTH)
    meta["pipeline"] = "cwru_v2_demean_robust_scale"

    np.save(OUTPUT_PATH, {"signals": scaled, "labels": labels, "meta": meta})

    uniq, cnt = np.unique(labels, return_counts=True)
    dist = {int(u): int(c) for u, c in zip(uniq, cnt)}
    print(f"[CWRU-GAF v2] Saved: {OUTPUT_PATH}")
    print(f"[CWRU-GAF v2] N={n}, shape={scaled.shape}, label_dist={dist}")
    print(f"[CWRU-GAF v2] meta lo/hi=({meta['lo']:.6f}, {meta['hi']:.6f})")


if __name__ == "__main__":
    main()
