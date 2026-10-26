"""
生成 HUST-bearing 的结构增强版 GASF / GADF 图像（Structure-aware Enhanced GAF, SA-GAF）。

基于 1.1_preprocess_custom.py 为 HUST 输出的 gaf_preprocessed_hust.npy：
- 每条样本为 2048 点单通道幅值序列（由三轴归一化信号合成）
- 为了避免 2048×2048 GAF 过慢，本脚本会先把每条序列重采样到较小长度（默认 256）
- 对重采样后的序列计算 GASF / GADF
- 在 GAF 矩阵上做轻量的结构增强（对比度拉伸 + unsharp mask 边缘增强）
- 归一化到 [0, 1] 后，使用 'jet' 伪彩色保存

输出目录（不覆盖 STFT 频谱图）：
- GASF 增强图：data/images_enhanced/gasf/<label>/gasf_<label>_<idx>.png
- GADF 增强图：data/images_enhanced/gadf/<label>/gadf_<label>_<idx>.png
"""

import os
import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve

from utils.gaf_utils import gaf_transform


INPUT_NPY = str(pp.GAF_PREPROCESSED_HUST)
OUTPUT_DIR = str(pp.IMAGES_ENHANCED)
SIGNAL_LENGTH = 2048  # 与 HUST SEGMENT_LENGTH 一致

# GAF 图像边长（重采样后的序列长度）；越小越快
GAF_SIZE = 256

# 每个标签最多生成多少组 SA-GAF 图像用于验证
MAX_IMAGES_PER_LABEL = 50


def normalize_to_01(matrix: np.ndarray) -> np.ndarray:
    """将任意实数矩阵归一化到 [0, 1]。"""
    mat_min = float(matrix.min())
    mat_max = float(matrix.max())
    if mat_max == mat_min:
        return np.zeros_like(matrix, dtype=np.float32)
    return (matrix - mat_min) / (mat_max - mat_min)


def local_contrast_stretch(matrix: np.ndarray, low_perc: float = 2.0, high_perc: float = 98.0) -> np.ndarray:
    """
    简单局部对比度拉伸：裁剪到 [low_perc, high_perc] 百分位，再线性映射到 [0, 1]。
    避免极端异常值“压扁”大部分像素。
    """
    lo, hi = np.percentile(matrix, [low_perc, high_perc])
    if hi <= lo:
        return normalize_to_01(matrix)
    clipped = np.clip(matrix, lo, hi)
    return (clipped - lo) / (hi - lo + 1e-8)


def conv2d_reflect_fast(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """使用 scipy.ndimage.convolve 的快速反射边界卷积。"""
    return convolve(image.astype(np.float32), kernel.astype(np.float32), mode="reflect")


def resample_1d(signal: np.ndarray, out_len: int) -> np.ndarray:
    """线性插值重采样到 out_len（不依赖 scipy.signal，速度快且稳定）。"""
    signal = np.asarray(signal, dtype=np.float32)
    in_len = signal.shape[0]
    if in_len == out_len:
        return signal
    x_old = np.arange(in_len, dtype=np.float32)
    x_new = np.linspace(0, in_len - 1, out_len, dtype=np.float32)
    return np.interp(x_new, x_old, signal).astype(np.float32)


def enhance_gaf_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    对单张 GAF 矩阵做结构增强：
    1) 对比度拉伸
    2) unsharp mask 边缘增强
    """
    # 1) 初步归一化到 [0, 1]
    base = normalize_to_01(matrix)
    # 2) 百分位裁剪 + 重新归一化，提升局部对比度
    stretched = local_contrast_stretch(base)
    # 3) unsharp mask：原图 + 边缘
    #    使用经典锐化核，相当于原图 - 轻微模糊，再做一次加权
    sharpen_kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )
    sharpened = conv2d_reflect_fast(stretched, sharpen_kernel)
    # 4) 防止数值越界，再归一化到 [0, 1]
    enhanced = normalize_to_01(sharpened)
    return enhanced


def load_signals_and_labels() -> Tuple[np.ndarray, np.ndarray]:
    """从 gaf_preprocessed_cwru.npy 中读取 signals / labels 并做基本检查。"""
    try:
        npy_obj = np.load(INPUT_NPY, allow_pickle=True)

        data = None
        try:
            if isinstance(npy_obj, np.ndarray) and npy_obj.dtype == object and npy_obj.shape == ():
                data = npy_obj.item()
            elif isinstance(npy_obj, dict):
                data = npy_obj
            elif isinstance(npy_obj, np.lib.npyio.NpzFile):
                data = {
                    "signals": npy_obj["signals"],
                    "labels": npy_obj["labels"],
                }
            else:
                data = npy_obj.item()
        except Exception:
            pass

        signals = np.asarray(data["signals"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=int)

        if signals.ndim != 2 or signals.shape[1] != SIGNAL_LENGTH:
            raise ValueError(f"signals shape {signals.shape} 不符合预期 (N, {SIGNAL_LENGTH})")
        if labels.ndim != 1 or labels.shape[0] != signals.shape[0]:
            raise ValueError(
                f"labels shape {labels.shape} 与 signals.shape[0]={signals.shape[0]} 不匹配"
            )

        return signals, labels
    except Exception as e:
        raise


def main():
    signals, labels = load_signals_and_labels()

    gasf_root = os.path.join(OUTPUT_DIR, "gasf")
    gadf_root = os.path.join(OUTPUT_DIR, "gadf")
    os.makedirs(gasf_root, exist_ok=True)
    os.makedirs(gadf_root, exist_ok=True)

    n = len(signals)
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_hist = {int(lab): int(cnt) for lab, cnt in zip(unique_labels.tolist(), counts.tolist())}
    if MAX_IMAGES_PER_LABEL is None:
        target_total = int(n)
    else:
        target_total = int(sum(min(int(cnt), int(MAX_IMAGES_PER_LABEL)) for cnt in counts.tolist()))

    print(f"[SA-GAF] 总样本数: {n}")
    print(f"[SA-GAF] 标签分布: {label_hist}")
    print(f"[SA-GAF] GAF_SIZE={GAF_SIZE}（重采样后边长），每类最多 {MAX_IMAGES_PER_LABEL} 张，预计生成: {target_total} 张/分支")

    per_label_counts = {}
    done = 0
    t0 = time.time()

    for idx in range(n):
        signal = signals[idx]
        label = int(labels[idx])

        # 每个标签只生成前 MAX_IMAGES_PER_LABEL 张增强图像
        if MAX_IMAGES_PER_LABEL is not None:
            cnt = per_label_counts.get(label, 0)
            if cnt >= MAX_IMAGES_PER_LABEL:
                continue
            per_label_counts[label] = cnt + 1

        # 重采样降低 GAF 计算复杂度
        signal_small = resample_1d(signal, GAF_SIZE)

        # 生成基础 GASF / GADF
        gasf = gaf_transform(signal_small, method="s")
        gadf = gaf_transform(signal_small, method="d")

        # 结构增强
        gasf_enh = enhance_gaf_matrix(gasf)
        gadf_enh = enhance_gaf_matrix(gadf)

        gasf_dir = os.path.join(gasf_root, str(label))
        gadf_dir = os.path.join(gadf_root, str(label))
        os.makedirs(gasf_dir, exist_ok=True)
        os.makedirs(gadf_dir, exist_ok=True)

        gasf_path = os.path.join(gasf_dir, f"gasf_{label}_{idx:05d}.png")
        gadf_path = os.path.join(gadf_dir, f"gadf_{label}_{idx:05d}.png")

        plt.imsave(gasf_path, gasf_enh, cmap="jet")
        plt.imsave(gadf_path, gadf_enh, cmap="jet")

        done += 1
        if done % 10 == 0 or done == target_total:
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-6)
            eta_s = (target_total - done) / max(rate, 1e-6)
            print(f"[SA-GAF] 进度: {done}/{target_total} | {rate:.2f} img/s | ETA {eta_s/60:.1f} min")

    print("[SA-GAF] GASF / GADF 增强图像生成完成！")


if __name__ == "__main__":
    main()

