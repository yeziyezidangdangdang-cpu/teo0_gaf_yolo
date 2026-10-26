import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# === 配置区 ===
# 当前默认：SUBF + AREEMD + 1.1 之后的预处理结果
# 如需对 CWRU 做相同频谱可视化，只需把路径改成对应的 gaf_preprocessed_cwru.npy
INPUT_NPY = Path(r"D:\TEP_2_repro\gaf_yolo\data\processed\gaf_preprocessed_subf.npy")

# 输出频谱图目录
OUTPUT_DIR = Path(r"D:\TEP_2_repro\gaf_yolo\data\diagnostics\spectra_preprocessed")

# 每个标签最多画多少条频谱
MAX_PLOTS_PER_LABEL = 10

# 采样率（Hz），SUBF 为 1000 Hz；若改为 CWRU，请同步修改
FS = 1000.0


def load_signals_and_labels():
    if not INPUT_NPY.exists():
        raise FileNotFoundError(f"找不到预处理结果: {INPUT_NPY}")

    npy_obj = np.load(INPUT_NPY, allow_pickle=True)

    data = None
    if isinstance(npy_obj, np.ndarray) and npy_obj.dtype == object and npy_obj.shape == ():
        data = npy_obj.item()
    elif isinstance(npy_obj, dict):
        data = npy_obj
    elif isinstance(npy_obj, np.lib.npyio.NpzFile):
        data = {"signals": npy_obj["signals"], "labels": npy_obj["labels"]}
    else:
        data = npy_obj.item()

    signals = np.asarray(data["signals"], dtype=float)
    labels = np.asarray(data["labels"], dtype=int)

    if signals.ndim != 2:
        raise ValueError(f"signals 维度异常: {signals.shape}")
    if labels.ndim != 1 or labels.shape[0] != signals.shape[0]:
        raise ValueError("labels 与 signals 行数不匹配")

    return signals, labels


def compute_spectrum(signal: np.ndarray):
    """计算单段信号的单边幅度谱（对数尺度）。"""
    L = signal.size
    # 去均值，减少直流
    sig = signal - np.mean(signal)
    fft_vals = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(L, d=1.0 / FS)
    mag = np.abs(fft_vals)
    # 避免 log(0)
    mag_log = 20 * np.log10(mag + 1e-8)
    return freqs, mag_log


def main():
    signals, labels = load_signals_and_labels()
    n, L = signals.shape

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    unique_labels = np.unique(labels)
    print(f"预处理信号形状: {signals.shape}，标签种类: {unique_labels.tolist()}")

    per_label_counts = {int(lab): 0 for lab in unique_labels}

    for idx in range(n):
        sig = signals[idx]
        lab = int(labels[idx])

        if per_label_counts[lab] >= MAX_PLOTS_PER_LABEL:
            continue

        per_label_counts[lab] += 1

        freqs, mag_log = compute_spectrum(sig)

        plt.figure(figsize=(6, 3))
        plt.plot(freqs, mag_log, linewidth=0.8)
        plt.title(f"Spectrum (label {lab}, sample {idx})")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.tight_layout()

        out_path = OUTPUT_DIR / f"label_{lab}_idx_{idx:05d}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

    print("🎉 预处理信号频谱图已生成。")
    print(f"   输出目录: {OUTPUT_DIR}")
    for lab in sorted(per_label_counts.keys()):
        print(f"   Label {lab}: 绘制 {per_label_counts[lab]} 条")


if __name__ == "__main__":
    main()

