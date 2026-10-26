import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# === 配置区 ===
# 1.1_preprocess_custom.py 输出的 NPY（SUBF + AREEMD 之后）
INPUT_NPY = Path(r"D:\TEP_2_repro\gaf_yolo\data\processed\gaf_preprocessed_cwru.npy")

# 保存时域波形图的目录
OUTPUT_DIR = Path(r"D:\TEP_2_repro\gaf_yolo\data\diagnostics\signals_preprocessed")

# 每个标签最多画多少条信号
MAX_PLOTS_PER_LABEL = 10


def load_signals_and_labels():
    if not INPUT_NPY.exists():
        raise FileNotFoundError(f"找不到预处理结果: {INPUT_NPY}")

    npy_obj = np.load(INPUT_NPY, allow_pickle=True)

    data = None
    # 兼容 np.save(dict) / np.savez
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


def main():
    signals, labels = load_signals_and_labels()
    n, L = signals.shape

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    unique_labels = np.unique(labels)
    print(f"预处理信号形状: {signals.shape}，标签种类: {unique_labels.tolist()}")

    per_label_counts = {int(lab): 0 for lab in unique_labels}

    # 遍历全部样本，但对每个标签只画前 MAX_PLOTS_PER_LABEL 条
    for idx in range(n):
        sig = signals[idx]
        lab = int(labels[idx])

        if per_label_counts[lab] >= MAX_PLOTS_PER_LABEL:
            continue

        per_label_counts[lab] += 1

        plt.figure(figsize=(6, 3))
        plt.plot(sig, linewidth=0.8)
        plt.title(f"Label {lab} - sample {idx}")
        plt.xlabel("Sample index")
        plt.ylabel("Amplitude (preprocessed)")
        plt.tight_layout()

        out_path = OUTPUT_DIR / f"label_{lab}_idx_{idx:05d}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

    print("🎉 预处理信号波形图已生成。")
    print(f"   输出目录: {OUTPUT_DIR}")
    for lab in sorted(per_label_counts.keys()):
        print(f"   Label {lab}: 绘制 {per_label_counts[lab]} 条")


if __name__ == "__main__":
    main()

