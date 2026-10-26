"""
生成 HUST-bearing 三通道 GAF 图像（X/Y/Z -> RGB），用于 YOLOv8 分类对比 STFT。

输入（由 1.1_preprocess_custom.py 生成）：
- data/processed/gaf_input_hust_axes.npy        （原波形三通道，全局鲁棒缩放到 [-1,1]）
- data/processed/gaf_input_hust_axes_env.npy    （可选：包络三通道，全局鲁棒缩放到 [-1,1]）

输出：
- data/gaf_rgb/gasf/<label>/*.png
- data/gaf_rgb/gadf/<label>/*.png

可选包络输出（若 USE_ENVELOPE=True 且文件存在）：
- data/gaf_rgb_env/gasf/<label>/*.png
- data/gaf_rgb_env/gadf/<label>/*.png
"""

from __future__ import annotations

import os
import time
from pathlib import Path
import sys
import shutil

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.gaf_utils import gaf_transform  # noqa: E402
import utils.project_paths as pp  # noqa: E402

INPUT_AXES_NPY = str(pp.GAF_INPUT_HUST_AXES)
INPUT_ENV_AXES_NPY = str(pp.GAF_INPUT_HUST_AXES_ENV)

OUTPUT_DIR = str(pp.GAF_RGB_HUST)
OUTPUT_DIR_ENV = str(pp.GAF_RGB_ENV_HUST)

# 速度/分辨率折中：GAF 边长（会把 2048 点重采样到该长度）
GAF_SIZE = 256

# 每个标签最多生成多少张图（结合下采样策略，可适当降低重复度）
MAX_IMAGES_PER_LABEL = 1500

# 为减少高度重叠片段带来的重复 GAF 图像，这里引入一个“索引下采样模式”：
# - 例：KEEP_MOD = 8, KEEP_PATTERN = {0,3,6} 表示仅当 idx % 8 ∈ {0,3,6} 时才生成图像，
#   约保留 3/8 ≈ 37.5% 的片段，其余片段直接跳过。
KEEP_MOD = 8
KEEP_PATTERN = {0, 3, 6}

# 是否额外生成包络版（需要 INPUT_ENV_AXES_NPY 存在）
USE_ENVELOPE = True

# 生成前是否清空输出目录（避免混入旧版本图像）
CLEAR_OUTPUT_DIRS = True

# --- 关键：让 GAF 不“卡在 -1 或 0.5”附近的增强 ---
# 对输入信号做非线性增益，使多数样本（尤其 0/1/3）远离 0，从而避免 GASF 大面积趋近 -1
SIGNAL_GAIN = 3.0          # >1 会把小幅值推向两端（随后 clip 回 [-1,1]）
SIGNED_POWER = 0.7         # <1 会放大小幅值（|x|^p），进一步减少“全都挤在 0 附近”

# 对 GAF 矩阵做轻量对比度拉伸（提升颜色层次与细节）
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
    """
    x01: [0,1]，按百分位裁剪后再拉伸回 [0,1]（每张图每通道独立）。
    """
    lo, hi = np.percentile(x01, [low_perc, high_perc])
    if hi <= lo:
        return x01.astype(np.float32)
    y = np.clip(x01, lo, hi)
    return ((y - lo) / (hi - lo + 1e-8)).astype(np.float32)


def to_rgb_gaf(sig3: np.ndarray, method: str) -> np.ndarray:
    """
    sig3: (3, L) in [-1, 1]
    return: (H, W, 3) in [0, 1]
    """
    mats = []
    for ch in range(3):
        s = resample_1d(sig3[ch], GAF_SIZE)
        # 关键：对信号做非线性增益，让多数样本离开 0（避免 GASF 大面积趋近 -1）
        s = signed_power(s, SIGNED_POWER)
        if SIGNAL_GAIN != 1.0:
            s = np.clip(s * float(SIGNAL_GAIN), -1.0, 1.0)
        # 重要：normalize=False，保留 1.1 的全局缩放差异
        m = gaf_transform(s, method=method, normalize=False).astype(np.float32)  # [-1,1]
        m01 = (m + 1.0) * 0.5  # -> [0,1]
        if MATRIX_CONTRAST_STRETCH:
            m01 = contrast_stretch_01(m01, STRETCH_LOW_PERC, STRETCH_HIGH_PERC)
        mats.append(m01)
    rgb = np.stack(mats, axis=-1)  # (H,W,3)
    return np.clip(rgb, 0.0, 1.0)


def load_npy_dict(path: str) -> dict:
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        return obj.item()
    if isinstance(obj, dict):
        return obj
    return obj.item()


def generate_branch(input_npy: str, out_dir: str) -> None:
    data = load_npy_dict(input_npy)
    signals = np.asarray(data["signals"], dtype=np.float32)  # (N,3,L)
    labels = np.asarray(data["labels"], dtype=int)

    N = signals.shape[0]
    print(f"[GAF-RGB] input={input_npy}")
    print(f"[GAF-RGB] signals={signals.shape}, labels={labels.shape}, GAF_SIZE={GAF_SIZE}")
    print(
        f"[GAF-RGB] enhance: gain={SIGNAL_GAIN}, signed_power={SIGNED_POWER}, "
        f"contrast_stretch={MATRIX_CONTRAST_STRETCH}({STRETCH_LOW_PERC}-{STRETCH_HIGH_PERC}%)"
    )

    gasf_root = Path(out_dir) / "gasf"
    gadf_root = Path(out_dir) / "gadf"
    if CLEAR_OUTPUT_DIRS:
        if Path(out_dir).exists():
            shutil.rmtree(out_dir, ignore_errors=True)
    gasf_root.mkdir(parents=True, exist_ok=True)
    gadf_root.mkdir(parents=True, exist_ok=True)

    per_label = {}
    target = 0
    uniq, cnts = np.unique(labels, return_counts=True)
    for c in cnts.tolist():
        target += min(int(c), int(MAX_IMAGES_PER_LABEL)) if MAX_IMAGES_PER_LABEL is not None else int(c)

    t0 = time.time()
    done = 0

    for idx in range(N):
        # 下采样：按索引模式丢弃一部分高度重叠片段，减少重复图像
        if KEEP_MOD is not None and KEEP_PATTERN and (idx % KEEP_MOD) not in KEEP_PATTERN:
            continue

        lab = int(labels[idx])
        if MAX_IMAGES_PER_LABEL is not None:
            c = per_label.get(lab, 0)
            if c >= MAX_IMAGES_PER_LABEL:
                continue
            per_label[lab] = c + 1

        sig3 = signals[idx]  # (3,L)

        rgb_gasf = to_rgb_gaf(sig3, method="s")
        rgb_gadf = to_rgb_gaf(sig3, method="d")

        (gasf_root / str(lab)).mkdir(parents=True, exist_ok=True)
        (gadf_root / str(lab)).mkdir(parents=True, exist_ok=True)

        gasf_path = gasf_root / str(lab) / f"gasf_rgb_{lab}_{idx:05d}.png"
        gadf_path = gadf_root / str(lab) / f"gadf_rgb_{lab}_{idx:05d}.png"

        plt.imsave(gasf_path, rgb_gasf, origin="lower")
        plt.imsave(gadf_path, rgb_gadf, origin="lower")

        done += 1
        if done % 10 == 0 or done == target:
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-6)
            eta_s = (target - done) / max(rate, 1e-6)
            print(f"[GAF-RGB] {done}/{target} | {rate:.2f} img/s | ETA {eta_s/60:.1f} min")

    print(f"[GAF-RGB] done. output -> {out_dir}")


def main() -> None:
    generate_branch(INPUT_AXES_NPY, OUTPUT_DIR)

    if USE_ENVELOPE:
        if os.path.exists(INPUT_ENV_AXES_NPY):
            generate_branch(INPUT_ENV_AXES_NPY, OUTPUT_DIR_ENV)
        else:
            print(f"[GAF-RGB] skip envelope: not found {INPUT_ENV_AXES_NPY}")


if __name__ == "__main__":
    main()

