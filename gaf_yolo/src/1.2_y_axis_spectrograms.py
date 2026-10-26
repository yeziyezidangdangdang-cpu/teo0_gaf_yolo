"""
生成 HUST-bearing Y 轴单通道 STFT 频谱图（灰度），用于单轴子模型 + late fusion。

输出：
- data/spec_y/<label>/spec_y_<label>_<idx>.png
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import utils.project_paths as pp  # noqa: E402

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft


INPUT_NPY = str(pp.STFT_INPUT_HUST)
OUTPUT_DIR = str(pp.SPEC_Y)

FS = 51200.0
N_PER_SEG = 256
N_OVERLAP = 128

MAX_IMAGES_PER_LABEL = 50


def stft_mag_db(x: np.ndarray) -> np.ndarray:
    f, t, Zxx = stft(x, fs=FS, nperseg=N_PER_SEG, noverlap=N_OVERLAP)
    mag = np.abs(Zxx)
    mag_db = 20 * np.log10(mag + 1e-8)
    return mag_db.astype(np.float32)


def normalize_01(mag_db: np.ndarray) -> np.ndarray:
    lo = float(mag_db.min())
    hi = float(mag_db.max())
    if hi <= lo:
        return np.zeros_like(mag_db, dtype=np.float32)
    return ((mag_db - lo) / (hi - lo + 1e-8)).astype(np.float32)


def main() -> None:
    npy_obj = np.load(INPUT_NPY, allow_pickle=True)
    data = npy_obj.item() if isinstance(npy_obj, np.ndarray) and npy_obj.dtype == object else npy_obj

    signals = np.asarray(data["signals"], dtype=np.float32)  # (N,3,L)
    labels = np.asarray(data["labels"], dtype=int)
    N = signals.shape[0]

    print(f"[Y-STFT] signals shape={signals.shape}, labels shape={labels.shape}")

    out_root = Path(OUTPUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    per_label_counts: dict[int, int] = {}

    for idx in range(N):
        y_axis = signals[idx, 1]  # (L,)
        lab = int(labels[idx])

        if MAX_IMAGES_PER_LABEL is not None:
            cnt = per_label_counts.get(lab, 0)
            if cnt >= MAX_IMAGES_PER_LABEL:
                continue
            per_label_counts[lab] = cnt + 1

        spec_db = stft_mag_db(y_axis)
        spec_norm = normalize_01(spec_db)

        lab_dir = out_root / str(lab)
        lab_dir.mkdir(parents=True, exist_ok=True)
        out_path = lab_dir / f"spec_y_{lab}_{idx:05d}.png"
        plt.imsave(out_path, spec_norm, origin="lower", cmap="viridis")

        if (idx + 1) % 500 == 0:
            print(f"[Y-STFT] 已生成 {idx + 1} 张 Y 轴 STFT 频谱图")

    print("[Y-STFT] Y 轴 STFT 频谱图生成完成。")


if __name__ == "__main__":
    main()

