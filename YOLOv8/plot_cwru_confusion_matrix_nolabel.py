"""生成 CWRU / 多轴 GAF（HUST）双分支评估混淆矩阵的无字版本（纯热力图，无轴标签/标题/格内数值，保留刻度线与 colorbar）。

从各评估目录读取 confusion_matrix_normalized.csv，分别保存为 confusion_matrix_nolabel.png。
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

DIRS = [
    Path(r"D:\TEP_2_repro\YOLOv8\runs\classify\runs\cwru_cls\cwru_dual_eval_enh"),  # CWRU SA-GAF
    Path(r"D:\TEP_2_repro\YOLOv8\runs\cwru_cls\cwru_dual_eval"),  # CWRU GAF
    Path(r"D:\TEP_2_repro\YOLOv8\runs\hust_gaf_cls\eval_dual_mean"),  # 多轴 GAF (HUST 双分支简单平均)
]


def plot_nolabel(cm_norm: np.ndarray, out_path: Path, figsize=(4, 4)):
    """绘制无字版混淆矩阵：热力图 + 坐标轴刻度线(0,1,2,3) + 右侧 colorbar 及刻度，无轴标签/标题/格内数值。"""
    plot_mat = np.nan_to_num(cm_norm.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(plot_mat, cmap="Blues", vmin=0, vmax=1)

    # 保留刻度线，刻度不显示数字
    n = plot_mat.shape[0]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # 保留右侧条状 colorbar 及刻度 (0.0 ~ 1.0)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    fig.tight_layout(pad=0.1)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    for d in DIRS:
        norm_csv = d / "confusion_matrix_normalized.csv"
        if not norm_csv.exists():
            print(f"Skip (no file): {norm_csv}")
            continue
        cm = np.loadtxt(norm_csv, delimiter=",")
        out_path = d / "confusion_matrix_nolabel.png"
        plot_nolabel(cm, out_path)


if __name__ == "__main__":
    main()
