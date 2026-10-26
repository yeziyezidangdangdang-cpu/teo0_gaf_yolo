"""
生成 HUST-bearing 的「多视角」三通道 STFT 频谱图（创新方案）：

- 从 1.1 预处理后的 stft_input_hust.npy 读取 (N, 3, L) 信号
- 先合成三轴幅值：s_mag = sqrt(X^2 + Y^2 + Z^2)
- 再构造三种视角并分别做 STFT，堆成 RGB：
  - R 通道：原始幅值 s_mag 的 STFT（总体能量视角）
  - G 通道：高通残差 s_res = s_mag - moving_average(s_mag) 的 STFT（冲击/高频视角）
  - B 通道：Hilbert 包络 |H(s_mag)| 的 STFT（调制/包络视角）

输出目录：
- data/spectrograms_mv/<label>/spec_mv_<label>_<idx>.png

说明：
- 主线 1.2 保持简单的多轴 STFT（X/Y/Z→RGB），本脚本作为「多视角增强版」供对比实验使用。
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
from scipy.signal import stft, hilbert


INPUT_NPY = str(pp.STFT_INPUT_HUST)
OUTPUT_DIR = str(pp.PROCESSED.parent / 'spectrograms_mv')

FS = 51200.0
N_PER_SEG = 256
N_OVERLAP = 128

MAX_IMAGES_PER_LABEL = 50

# 残差平滑窗口（样本点数），控制「高通」强度
RESID_SMOOTH_WIN = 256


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    """简单移动平均，边缘使用 pad 模拟 same 长度。"""
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


def normalize_view(mag_db: np.ndarray) -> np.ndarray:
    """
    对单个视角的 (F,T) dB 进行简单百分位拉伸到 [0,1]。
    不做跨样本全局缩放，保持实现简单、易解释。
    """
    p1, p99 = np.percentile(mag_db, [1.0, 99.0])
    if p99 <= p1:
        p1, p99 = float(mag_db.min()), float(mag_db.max() + 1e-6)
    x = np.clip(mag_db, p1, p99)
    x01 = (x - p1) / (p99 - p1 + 1e-8)
    return x01.astype(np.float32)


def main() -> None:
    npy_obj = np.load(INPUT_NPY, allow_pickle=True)
    data = npy_obj.item() if isinstance(npy_obj, np.ndarray) and npy_obj.dtype == object else npy_obj

    signals = np.asarray(data["signals"], dtype=np.float32)  # (N,3,L)
    labels = np.asarray(data["labels"], dtype=int)
    N, _, L = signals.shape

    print(f"[MV-STFT] signals shape={signals.shape}, labels shape={labels.shape}")

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

        # 三种视角
        # 1) 原始幅值
        v_orig = mag

        # 2) 高通残差
        smooth = moving_average(mag, RESID_SMOOTH_WIN)
        v_resid = mag - smooth

        # 3) Hilbert 包络
        v_env = np.abs(hilbert(mag)).astype(np.float32)

        # 分别做 STFT 并归一化到 [0,1]
        views = []
        for v in (v_orig, v_resid, v_env):
            m_db = stft_mag_db(v)
            m01 = normalize_view(m_db)
            views.append(m01)

        rgb = np.stack(views, axis=-1)  # (F,T,3)

        lab_dir = out_root / str(lab)
        lab_dir.mkdir(parents=True, exist_ok=True)
        out_path = lab_dir / f"spec_mv_{lab}_{idx:05d}.png"
        plt.imsave(out_path, rgb, origin="lower")

        if (idx + 1) % 500 == 0:
            print(f"[MV-STFT] 已生成 {idx + 1} 张多视角 STFT 频谱图")

    print("[MV-STFT] 多视角 STFT 频谱图生成完成。")


if __name__ == "__main__":
    main()

