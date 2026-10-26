"""
3.4_visualize_cwru_raw_waveforms.py

可视化 CWRU 切片后的原始波形，用于直观对比 CWRU vs HUST 的时域差异。

读取：
- data/processed/final_dataset_cwru.csv
  每行结构: t_0..t_{L-1}, label（无表头，一维信号）

输出：
- data/diagnostics/cwru_waveforms_raw/label_<k>/wave_<k>_<row_idx>.png
"""

from __future__ import annotations

import os
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np


import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.project_paths import (  # noqa: E402
    CWRU_SEGMENT_LENGTH,
    FINAL_DATASET_CWRU_V2,
    PROCESSED_V2,
    ensure_v2_dirs,
)

RAW_CSV = str(FINAL_DATASET_CWRU_V2)
OUT_DIR = str(PROCESSED_V2 / "diagnostics" / "cwru_waveforms_raw_v2")

SEGMENT_LENGTH = CWRU_SEGMENT_LENGTH
MAX_PLOTS_PER_LABEL = 16
RANDOM_SEED = 42


def iter_rows(csv_path: str):
    """逐行读取 CSV，返回 (row_index, signal[L], label)"""
    expected = SEGMENT_LENGTH + 1
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            arr = np.fromstring(line, sep=",", dtype=np.float32)
            if arr.size != expected:
                raise ValueError(f"第 {i} 行列数异常：期望 {expected}，实际 {arr.size}")
            sig = arr[:SEGMENT_LENGTH]
            lab = int(round(float(arr[-1])))
            yield i, sig, lab


def reservoir_sample_per_label(csv_path: str, k: int, seed: int):
    """
    每类最多保留 k 条样本（reservoir sampling），避免一次性加载全部。
    返回: dict[label] -> list[(row_idx, signal)]
    """
    rng = random.Random(seed)
    state: dict[int, tuple[int, list]] = {}

    for row_idx, sig, lab in iter_rows(csv_path):
        seen, samples = state.get(lab, (0, []))
        seen += 1

        if len(samples) < k:
            samples.append((row_idx, sig.copy()))
        else:
            j = rng.randrange(seen)
            if j < k:
                samples[j] = (row_idx, sig.copy())

        state[lab] = (seen, samples)

    return {lab: samples for lab, (seen, samples) in state.items()}


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    return float(np.sqrt(np.mean(x * x)))


def plot_one(out_path: Path, sig: np.ndarray, label: int, idx: int) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.plot(sig, lw=0.8)
    ax.set_title(f"CWRU | Label {label} | sample idx {idx} | rms={rms(sig):.4g}")
    ax.set_xlabel("sample")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    per_label_samples = reservoir_sample_per_label(RAW_CSV, MAX_PLOTS_PER_LABEL, RANDOM_SEED)
    labels = sorted(per_label_samples.keys())
    print(f"[CWRU-WAVE] labels found: {labels} | max_plots_per_label={MAX_PLOTS_PER_LABEL}")

    for lab in labels:
        samples = per_label_samples.get(lab, [])
        print(f"[CWRU-WAVE] label {lab}: plotting {len(samples)}")
        for row_idx, sig in samples:
            out_path = Path(OUT_DIR) / f"label_{lab}" / f"wave_{lab}_{row_idx:05d}.png"
            plot_one(out_path, sig, lab, row_idx)

    print(f"[CWRU-WAVE] done. outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()

