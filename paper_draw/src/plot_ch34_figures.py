"""
Chapter 3.4 extended-experiment figures (with-text + nolabel).

Outputs -> D:\\TEP_2_repro\\paper_draw\\
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

DRAW = Path(__file__).resolve().parents[1]
_REPRO = DRAW.parent
sys.path.insert(0, str(_REPRO / "YOLOv8"))
import repro_paths as pp  # noqa: E402

RUNS = pp.YOLO_ROOT / "runs" / "classify" / "6_edit"

CLASS_CWRU = ["Normal", "Inner", "Outer", "Ball"]


def _simsun(size: int):
    try:
        return FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=size)
    except Exception:
        return FontProperties(size=size)


def _times(size: int):
    return FontProperties(family="Times New Roman", size=size)


def plot_cm(cm_norm: np.ndarray, labels: list[str], out_stem: str, title: str):
    for labeled in (True, False):
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
        im = ax.imshow(np.nan_to_num(cm_norm), cmap="Blues", vmin=0, vmax=1)
        n = cm_norm.shape[0]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        if labeled:
            ax.set_xticklabels([f"{i}\n{labels[i]}" for i in range(n)], fontproperties=_times(7))
            ax.set_yticklabels([f"{i}:{labels[i]}" for i in range(n)], fontproperties=_times(7))
            ax.set_xlabel("Predicted", fontproperties=_times(10))
            ax.set_ylabel("True", fontproperties=_times(10))
            ax.set_title(title, fontproperties=_simsun(11))
            for i in range(n):
                for j in range(n):
                    ax.text(
                        j, i, f"{cm_norm[i, j]:.2f}",
                        ha="center", va="center",
                        color="white" if cm_norm[i, j] > 0.5 else "black",
                        fontproperties=_times(8),
                    )
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


def plot_bars(categories: list[str], vals_a: list[float], vals_b: list[float], ylabel: str, title: str, out_stem: str):
    x = np.arange(len(categories))
    w = 0.35
    for labeled in (True, False):
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        b1 = ax.bar(x - w / 2, vals_a, w, label="简单平均", color="#3498db", edgecolor="#333", linewidth=0.5)
        b2 = ax.bar(x + w / 2, vals_b, w, label="注意力融合", color="#e67e22", edgecolor="#333", linewidth=0.5)
        if labeled:
            ax.set_ylabel(ylabel, fontproperties=_simsun(12))
            ax.set_title(title, fontproperties=_simsun(12))
            ax.set_xticks(x)
            ax.set_xticklabels(categories, fontproperties=_simsun(10))
            ax.legend(prop=_simsun(9), frameon=False)
            for bars in (b1, b2):
                for b in bars:
                    ax.text(
                        b.get_x() + b.get_width() / 2, b.get_height() + 0.15,
                        f"{b.get_height():.2f}", ha="center", va="bottom", fontproperties=_times(9),
                    )
            for t in ax.get_yticklabels():
                t.set_fontproperties(_times(10))
        else:
            ax.set_xticks(x)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        ax.set_ylim(95, 100.5)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        fig.tight_layout()
        suffix = "" if labeled else "_nolabel"
        out = DRAW / f"{out_stem}{suffix}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        print(f"Saved {out}")


def plot_cwru_branch_bars(gasf: float, gadf: float, dual: float, out_stem: str):
    vals = [gasf, gadf, dual]
    labels = ["GASF-RGB", "GADF-RGB", "双分支\n简单平均"]
    colors = ["#3498db", "#e67e22", "#2ecc71"]
    for labeled in (True, False):
        fig, ax = plt.subplots(figsize=(4.8, 3.2))
        bars = ax.bar(range(3), vals, color=colors, edgecolor="#333", linewidth=0.5, width=0.55)
        if labeled:
            ax.set_ylabel("测试集准确率 (%)", fontproperties=_simsun(12))
            ax.set_title("CWRU v2 双分支诊断性能", fontproperties=_simsun(12))
            ax.set_xticks(range(3))
            ax.set_xticklabels(labels, fontproperties=_simsun(10))
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.2f}", ha="center", va="bottom", fontproperties=_times(10))
            for t in ax.get_yticklabels():
                t.set_fontproperties(_times(10))
        else:
            ax.set_xticks(range(3))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        ax.set_ylim(96, 100.5)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        fig.tight_layout()
        suffix = "" if labeled else "_nolabel"
        out = DRAW / f"{out_stem}{suffix}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        print(f"Saved {out}")


def main():
    DRAW.mkdir(parents=True, exist_ok=True)

    cm = np.loadtxt(RUNS / "cwru_v2_cls" / "eval_dual_mean" / "confusion_matrix_normalized.csv", delimiter=",")
    plot_cm(cm, CLASS_CWRU, "fig_ch3_341_cwru_v2_dual_cm", "CWRU v2 双分支简单平均混淆矩阵")

    plot_cwru_branch_bars(99.11, 98.37, 99.41, "fig_ch3_341_cwru_v2_branch_bars")

    plot_bars(
        ["HUST\n稳态", "MCC5\n变工况"],
        [99.40, 99.33],
        [99.50, 97.56],
        "测试集总体准确率 (%)",
        "不同场景下简单平均与注意力融合对比",
        "fig_ch3_342_fusion_mean_vs_attn",
    )


if __name__ == "__main__":
    main()
