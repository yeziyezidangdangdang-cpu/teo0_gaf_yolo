"""
Chapter 3 split-comparison figures (with-text + nolabel versions).

Outputs -> D:\\TEP_2_repro\\paper_draw\\
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

DRAW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DRAW.parent / "YOLOv8"))
import repro_paths as pp  # noqa: E402

RUNS = pp.YOLO_ROOT / "runs" / "classify" / "6_edit"

CLASS_NAMES = ["Healthy", "Ball", "Inner", "Outer"]


def _simsun(size: int):
    try:
        return FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=size)
    except Exception:
        return FontProperties(size=size)


def _times(size: int):
    return FontProperties(family="Times New Roman", size=size)


def plot_group_bars(
    groups: list[str],
    series: dict[str, list[float]],
    ylabel: str,
    title: str,
    out_stem: str,
    ylim=(80, 101),
):
    x = np.arange(len(groups))
    n = len(series)
    width = 0.8 / n
    colors = ["#3498db", "#e67e22", "#2ecc71"]

    for labeled in (True, False):
        fig, ax = plt.subplots(figsize=(5.6, 3.4))
        for i, (name, vals) in enumerate(series.items()):
            offset = (i - (n - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width=width, label=name, color=colors[i % len(colors)], edgecolor="#333", linewidth=0.5)
            if labeled:
                for b, v in zip(bars, vals):
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.2f}", ha="center", va="bottom", fontproperties=_times(9))

        ax.set_ylim(*ylim)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        if labeled:
            ax.set_xlabel("划分方式", fontproperties=_simsun(12))
            ax.set_ylabel(ylabel, fontproperties=_simsun(12))
            ax.set_title(title, fontproperties=_simsun(12))
            ax.set_xticks(x)
            ax.set_xticklabels(groups, fontproperties=_simsun(10))
            ax.legend(prop=_simsun(9), frameon=False)
            for t in ax.get_yticklabels():
                t.set_fontproperties(_times(10))
        else:
            ax.set_xticks(x)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title("")

        fig.tight_layout()
        suffix = "" if labeled else "_nolabel"
        out = DRAW / f"{out_stem}{suffix}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        print(f"Saved {out}")


def plot_cm(cm_norm: np.ndarray, out_stem: str, title: str):
    for labeled in (True, False):
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
        im = ax.imshow(np.nan_to_num(cm_norm), cmap="Blues", vmin=0, vmax=1)
        n = cm_norm.shape[0]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        if labeled:
            ax.set_xticklabels([f"{i}\n{CLASS_NAMES[i]}" for i in range(n)], fontproperties=_times(8))
            ax.set_yticklabels([f"{i}:{CLASS_NAMES[i]}" for i in range(n)], fontproperties=_times(8))
            ax.set_xlabel("Predicted", fontproperties=_times(10))
            ax.set_ylabel("True", fontproperties=_times(10))
            ax.set_title(title, fontproperties=_simsun(11))
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center", color="white" if cm_norm[i,j] > 0.5 else "black", fontproperties=_times(8))
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        fig.colorbar(im, ax=ax, shrink=0.85)
        fig.tight_layout()
        suffix = "" if labeled else "_nolabel"
        out = DRAW / f"{out_stem}{suffix}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        print(f"Saved {out}")


def main():
    DRAW.mkdir(parents=True, exist_ok=True)

    # 3.1.3 HUST strict vs random
    plot_group_bars(
        groups=["Strict\n(主实验)", "Random\n(seed=42)"],
        series={
            "GASF-RGB": [99.27, 99.44],
            "GADF-RGB": [98.78, 99.67],
            "双分支简单平均": [99.76, 99.89],
        },
        ylabel="测试集准确率 (%)",
        title="HUST 稳态：strict 与 random 划分对比",
        out_stem="fig_ch3_313_hust_split_compare",
        ylim=(96, 100.5),
    )

    # 3.2.3 MCC5 three splits (dual mean only bar - simpler second fig)
    plot_group_bars(
        groups=["Whole-file\n(B,主)", "Temporal\n(C)", "Random\n(D)"],
        series={"双分支简单平均": [99.33, 87.20, 98.67]},
        ylabel="测试集准确率 (%)",
        title="MCC5 变工况：不同划分方式下双分支简单平均",
        out_stem="fig_ch3_323_mcc5_dual_split_compare",
        ylim=(80, 101),
    )

    plot_group_bars(
        groups=["Whole-file\n(B)", "Temporal\n(C)", "Random\n(D)"],
        series={
            "GASF-RGB": [97.33, 86.42, 97.78],
            "GADF-RGB": [94.11, 82.68, 96.33],
            "双分支简单平均": [99.33, 87.20, 98.67],
        },
        ylabel="测试集准确率 (%)",
        title="MCC5 变工况：各划分方式诊断性能",
        out_stem="fig_ch3_323_mcc5_split_compare_detail",
        ylim=(80, 101),
    )

    # 3.2.2 MCC5 B confusion matrix
    cm = np.loadtxt(RUNS / "mcc5_cls_b" / "eval_dual_mean" / "confusion_matrix_normalized.csv", delimiter=",")
    plot_cm(cm, "fig_ch3_322_mcc5_dual_cm", "MCC5 变工况主实验双分支简单平均混淆矩阵")


if __name__ == "__main__":
    main()
