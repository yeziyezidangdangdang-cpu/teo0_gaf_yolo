"""
1.2_generate_cwru_spectrograms.py

基于 CWRU 预处理结果生成 STFT 频谱图（灰度），用于 YOLOv8 分类（方案 A 在 CWRU 上的版本）。

输入：
- data/processed/stft_input_cwru.npy
  - signals: (N, L)  一维归一化信号（1024 点）
  - labels:  (N,)

输出：
- data/spectrograms_cwru/<label>/spec_cwru_<label>_<idx>.png
  - 单通道 STFT 幅度谱，归一化到 [0,1] 后以灰度形式保存。
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


INPUT_NPY = str(pp.PROCESSED / 'stft_input_cwru.npy')
OUTPUT_DIR = str(pp.SPECTROGRAMS_CWRU)

FS = 12000.0  # CWRU 常用采样率（可根据实际情况调整）
# 为避免 YOLO 将图像判为 “too small (<10 pixels)”，需要保证 STFT 结果在频率和时间维度上都足够大。
# 对 1024 点信号，nperseg=256, noverlap=192 -> ~129x17 时频图，频率分辨率 46.9 Hz，满足 YOLO 最小尺寸。
N_PER_SEG = 256
N_OVERLAP = 192  # hop=64 -> ~13 time bins，确保时间维度>=10 满足 YOLO

# 每个标签最多生成多少张频谱图（测试阶段可设为 200，想全量可改为 None）
MAX_IMAGES_PER_LABEL = 200

# 归一化策略：True=全局 dB 裁剪（保留类间能量差异），False=每图 min-max（易"压扁"）
USE_GLOBAL_DB_NORM = True
DB_MIN, DB_MAX = -50.0, 0.0  # 裁剪区间，正常类能量低会偏暗，故障类偏亮


def stft_mag_db(x: np.ndarray) -> np.ndarray:
    f, t, Zxx = stft(x, fs=FS, nperseg=N_PER_SEG, noverlap=N_OVERLAP)
    mag = np.abs(Zxx)
    mag_db = 20 * np.log10(mag + 1e-8)
    return mag_db.astype(np.float32)


def normalize_01(mag_db: np.ndarray, global_range: bool = False) -> np.ndarray:
    """归一化到 [0,1]。global_range=True 时用固定 dB 区间，保留类间能量差异。"""
    if global_range:
        lo, hi = DB_MIN, DB_MAX
        mag_clip = np.clip(mag_db, lo, hi)
        return ((mag_clip - lo) / (hi - lo + 1e-8)).astype(np.float32)
    lo, hi = float(mag_db.min()), float(mag_db.max())
    if hi <= lo:
        return np.zeros_like(mag_db, dtype=np.float32)
    return ((mag_db - lo) / (hi - lo + 1e-8)).astype(np.float32)


def main() -> None:
    npy_obj = np.load(INPUT_NPY, allow_pickle=True)
    data = npy_obj.item() if isinstance(npy_obj, np.ndarray) and npy_obj.dtype == object else npy_obj

    signals = np.asarray(data["signals"], dtype=np.float32)  # (N,L)
    labels = np.asarray(data["labels"], dtype=int)
    N = signals.shape[0]

    print(f"[CWRU-STFT] signals shape={signals.shape}, labels shape={labels.shape}")

    out_root = Path(OUTPUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    per_label_counts: dict[int, int] = {}

    for idx in range(N):
        sig = signals[idx]
        lab = int(labels[idx])

        if MAX_IMAGES_PER_LABEL is not None:
            cnt = per_label_counts.get(lab, 0)
            if cnt >= MAX_IMAGES_PER_LABEL:
                continue
            per_label_counts[lab] = cnt + 1

        spec_db = stft_mag_db(sig)
        spec_norm = normalize_01(spec_db, global_range=USE_GLOBAL_DB_NORM)

        lab_dir = out_root / str(lab)
        lab_dir.mkdir(parents=True, exist_ok=True)
        out_path = lab_dir / f"spec_cwru_{lab}_{idx:05d}.png"
        plt.imsave(out_path, spec_norm, origin="lower", cmap="viridis")

        if (idx + 1) % 500 == 0:
            print(f"[CWRU-STFT] 已处理 {idx + 1}/{N} 段信号")

    print("[CWRU-STFT] 频谱图生成完成。")


if __name__ == "__main__":
    main()

