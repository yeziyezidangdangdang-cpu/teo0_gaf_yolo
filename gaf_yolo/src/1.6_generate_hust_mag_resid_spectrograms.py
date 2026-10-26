"""
生成 HUST-bearing 单轴幅值 + 高通残差双通道 STFT 频谱图（简洁增强方案）：

- 从 1.1_preprocess_custom.py 输出的 stft_input_hust.npy 读取 (N, 3, L) 信号（已做每轴 z-score）
- 先合成三轴幅值：s_mag = sqrt(X^2 + Y^2 + Z^2)
- 再构造两条 1D 信号并分别做 STFT：
  - 通道 1（R）：原始幅值 s_mag 的 STFT（总体能量视角）
  - 通道 2（G）：高通残差 s_res = s_mag - moving_average(s_mag) 的 STFT（冲击/高频视角）
- 通道 3（B）：直接复用通道 1（方便以 3 通道 PNG 形式喂给 YOLO）

输出目录：
- data/spectrograms_mag_resid/<label>/spec_mr_<label>_<idx>.png

说明：
- 这是在「单轴幅值」基础上的最小增强：只额外引入一个高通视角，用来放大轻微故障（1/3 类）的差异。
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
OUTPUT_DIR = str(pp.SPECTROGRAMS_MAG_RESID)

FS = 51200.0
N_PER_SEG = 256
N_OVERLAP = 128

MAX_IMAGES_PER_LABEL = 50

# 残差平滑窗口长度（样本数）：越大，高通越“温和”
RESID_SMOOTH_WIN = 256


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    """简单移动平均，保持 same 长度。"""
    if win <= 1:
        return x.astype(np.float32)
    x = np.asarray(x, dtype=np.float32)
    pad = win // 2
    xp = np.pad(x, (pad, win - 1 - pad), mode="reflect")
    kernel = np.ones(win, dtype=np.float32) / float(win)
    y = np.convolve(xp, kernel, mode="valid")
    return y.astype(np.float32)


def stft_mag_db(x: np.ndarray) -> np.ndarray:
    f, t, Zxx = stft(x, fs=FS, nperseg=N_PER_SEG, noverlap=N_OVERLAP)
    mag = np.abs(Zxx)
    mag_db = 20 * np.log10(mag + 1e-8)
    return mag_db.astype(np.float32)


def normalize_01(mag_db: np.ndarray) -> np.ndarray:
    """对单张 (F,T) dB 幅度谱做简单 min-max 拉伸到 [0,1]。"""
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
    N, _, L = signals.shape

    print(f"[MAG-RESID] signals shape={signals.shape}, labels shape={labels.shape}")

    out_root = Path(OUTPUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    per_label_counts: dict[int, int] = {}

    for idx in range(N):
        sig_axes = signals[idx]  # (3,L)
        lab = int(labels[idx])

        if MAX_IMAGES_PER_LABEL is not None:
            cnt = per_label_counts.get(lab, 0)
            if cnt >= MAX_IMAGES_PER_LABEL:
                continue
            per_label_counts[lab] = cnt + 1

        x, y, z = sig_axes[0], sig_axes[1], sig_axes[2]
        mag = np.sqrt(x * x + y * y + z * z).astype(np.float32)  # (L,)

        # 原始幅值视角
        v_orig = mag

        # 高通残差视角
        smooth = moving_average(mag, RESID_SMOOTH_WIN)
        v_resid = mag - smooth

        # STFT + per-image 归一化
        m_orig = normalize_01(stft_mag_db(v_orig))
        m_resid = normalize_01(stft_mag_db(v_resid))

        # 构造 3 通道：R=orig, G=resid, B=orig
        h, w = m_orig.shape
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        rgb[:, :, 0] = m_orig
        rgb[:, :, 1] = m_resid
        rgb[:, :, 2] = m_orig

        lab_dir = out_root / str(lab)
        lab_dir.mkdir(parents=True, exist_ok=True)
        out_path = lab_dir / f"spec_mr_{lab}_{idx:05d}.png"
        plt.imsave(out_path, rgb, origin="lower")

        if (idx + 1) % 500 == 0:
            print(f"[MAG-RESID] 已生成 {idx + 1} 张幅值+残差 STFT 频谱图")

    print("[MAG-RESID] 幅值+残差 STFT 频谱图生成完成。")


if __name__ == "__main__":
    main()

