import os

import matplotlib.pyplot as plt
import numpy as np

from utils.gaf_utils import gaf_transform


INPUT_NPY = r"D:\TEP_2_repro\gaf_yolo\data\processed\gaf_preprocessed_vibration_machines_xyz.npy"
OUTPUT_DIR = r"D:\TEP_2_repro\gaf_yolo\data\images_vibration_machines_xyz_rgb"

SIGNAL_LENGTH = 128
MAX_IMAGES_PER_LABEL = 50


def gaf_to_01(gaf: np.ndarray) -> np.ndarray:
    """
    先做固定 [-1,1]→[0,1] 映射，保证不同样本之间的能量/幅值差异仍然体现在亮度范围内，
    然后再做轻微的局部对比度拉伸，让图像不至于“全黑发灰”。
    这里的拉伸是在 GAF 空间内做的，而不是直接对原始信号做逐样本 min-max，
    这样既能提高可视化效果，又不会完全抹掉类间差异。
    """
    base = np.clip((gaf + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)

    # 百分位裁剪 + 线性拉伸，增强对比度。
    # 注意：我们只对单张图内部做适度拉伸（2%~98%），避免极端值主导亮度，
    # 但保留整体明暗层次的差异。
    lo, hi = np.percentile(base, [2.0, 98.0])
    if hi <= lo:
        return base
    enhanced = np.clip((base - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return enhanced


def load_xyz_and_labels():
    npy_obj = np.load(INPUT_NPY, allow_pickle=True)
    data = npy_obj.item() if isinstance(npy_obj, np.ndarray) else npy_obj

    signals_xyz = np.asarray(data["signals_xyz"], dtype=np.float32)  # (N,3,128)
    labels = np.asarray(data["labels"], dtype=int)

    if signals_xyz.ndim != 3 or signals_xyz.shape[1] != 3 or signals_xyz.shape[2] != SIGNAL_LENGTH:
        raise ValueError(f"signals_xyz shape {signals_xyz.shape} 不符合预期 (N,3,{SIGNAL_LENGTH})")
    if labels.ndim != 1 or labels.shape[0] != signals_xyz.shape[0]:
        raise ValueError("labels 与 signals_xyz 数量不一致")

    return signals_xyz, labels


def main():
    signals_xyz, labels = load_xyz_and_labels()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    per_label_counts = {}
    n = signals_xyz.shape[0]
    print(f"[XYZ-RGB GAF] 总样本数: {n}")

    for idx in range(n):
        label = int(labels[idx])
        cnt = per_label_counts.get(label, 0)
        if cnt >= MAX_IMAGES_PER_LABEL:
            continue
        per_label_counts[label] = cnt + 1

        x = signals_xyz[idx, 0]
        y = signals_xyz[idx, 1]
        z = signals_xyz[idx, 2]

        # 三轴分别做 GASF，然后拼成 RGB（保留方向信息）
        rx = gaf_to_01(gaf_transform(x, method="s", normalize=False))
        gy = gaf_to_01(gaf_transform(y, method="s", normalize=False))
        bz = gaf_to_01(gaf_transform(z, method="s", normalize=False))

        rgb = np.stack([rx, gy, bz], axis=-1)  # (L,L,3) in [0,1]

        out_dir = os.path.join(OUTPUT_DIR, str(label))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"xyz_rgb_{label}_{idx:05d}.png")

        plt.imsave(out_path, rgb)

    print("[XYZ-RGB GAF] 小体量 RGB GAF 图像生成完成！")


if __name__ == "__main__":
    main()

