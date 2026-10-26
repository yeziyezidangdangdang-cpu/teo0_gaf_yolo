# src/3_generate_gaf_images.py
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import utils.project_paths as pp  # noqa: E402
"""
生成三通道 STFT 频谱图（HUST-bearing，多轴共通道，主线方案）：
- 从 1.1 预处理后的 stft_input_hust.npy 读取 (N, 3, L) 信号
- 对 X/Y/Z 三轴分别计算 STFT 幅度谱（dB），堆成 RGB 三通道图像

说明：
- 这是「主线」STFT 方案，保持相对简单、直观，方便与你的新多视角方案做对比。
- 归一化策略：每张图内部，分别对三个通道按各自 min/max 归一化到 [0, 1]。
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import stft


# STFT 输入：1.1 预处理输出
INPUT_NPY = str(pp.STFT_INPUT_HUST)
OUTPUT_DIR = str(pp.SPECTROGRAMS)

FS = 51200.0  # HUST 采样率（Hz）
N_PER_SEG = 256
N_OVERLAP = 128

# 每个标签最多生成多少张频谱图（与 GAF 一致：每类 1500 张即可）
MAX_IMAGES_PER_LABEL = 1500

# 索引下采样（与 GAF 同理）：减少滑动窗口高度重叠带来的重复样本
# KEEP_MOD=8, KEEP_PATTERN={0,3,6} 表示仅当 idx % 8 in {0,3,6} 时才生成，约保留 37.5%
KEEP_MOD = 8
KEEP_PATTERN = {0, 3, 6}


def main():
    # 载入 STFT 输入
    npy_obj = np.load(INPUT_NPY, allow_pickle=True)
    data = npy_obj.item() if isinstance(npy_obj, np.ndarray) and npy_obj.dtype == object else npy_obj

    signals = np.asarray(data["signals"], dtype=np.float32)  # (N, 3, L)
    labels = np.asarray(data["labels"], dtype=int)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    per_label_counts = {}

    for idx in range(len(signals)):
        # 索引下采样：仅保留部分片段，降低相邻片段高度重叠导致的冗余
        if KEEP_MOD is not None and (idx % KEEP_MOD) not in KEEP_PATTERN:
            continue

        sig_axes = signals[idx]  # (3, L)
        lab = int(labels[idx])

        if MAX_IMAGES_PER_LABEL is not None:
            cnt = per_label_counts.get(lab, 0)
            if cnt >= MAX_IMAGES_PER_LABEL:
                continue
            per_label_counts[lab] = cnt + 1

        # 分别对 X/Y/Z 三轴计算 STFT 幅度谱（dB）
        mag_db_channels = []
        for ch in range(3):
            f, t, Zxx = stft(sig_axes[ch], fs=FS, nperseg=N_PER_SEG, noverlap=N_OVERLAP)
            mag = np.abs(Zxx)
            mag_db = 20 * np.log10(mag + 1e-8)
            mag_db_channels.append(mag_db)

        # 堆成 (F, T, 3) 的三通道矩阵
        stack = np.stack(mag_db_channels, axis=-1)

        # 每张图按通道归一化到 [0, 1]
        ch_min = stack.min(axis=(0, 1), keepdims=True)
        ch_max = stack.max(axis=(0, 1), keepdims=True)
        stack_norm = (stack - ch_min) / (ch_max - ch_min + 1e-8)

        lab_dir = os.path.join(OUTPUT_DIR, str(lab))
        os.makedirs(lab_dir, exist_ok=True)
        out_path = os.path.join(lab_dir, f"spec_{lab}_{idx:05d}.png")

        # 直接作为 RGB 图像保存
        plt.imsave(out_path, stack_norm, origin="lower")

        if (idx + 1) % 500 == 0:
            print(f"[STFT-main] 已生成 {idx + 1} 张三通道 STFT 频谱图")

    print("[STFT-main] 三通道 STFT 频谱图生成完成！")


if __name__ == "__main__":
    main()