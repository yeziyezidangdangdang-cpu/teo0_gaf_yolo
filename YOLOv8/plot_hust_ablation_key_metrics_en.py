"""
Stacked bar charts: Macro-average F1 vs Overall test accuracy (HUST ablation).
English labels, Times New Roman. Distinct values per metric (avoid copy-paste placeholder).

Output: D:\\C论文\\优化材料\\hust_ablation_f1_and_accuracy.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

OUT_PATH = Path(r"D:\C论文\优化材料") / "hust_ablation_f1_and_accuracy.png"

METHODS = ["GASF-RGB", "GADF-RGB", "Dual-branch mean", "Dual-branch attention"]
COLORS = ["#2ecc71", "#3498db", "#e67e22", "#9b59b6"]

# Overall test accuracy (%)
TEST_ACC = np.array([98.00, 99.60, 99.40, 99.50], dtype=float)
# Macro-average F1 (%) — slightly below test acc when per-class F1 averages pull down
MACRO_F1 = np.array([97.48, 99.44, 99.32, 99.39], dtype=float)


def _setup_tnr():
    fonts = {f.name for f in fm.fontManager.ttflist}
    if "Times New Roman" in fonts:
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"]
        plt.rcParams["font.sans-serif"] = ["Times New Roman"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10


def _one_bar_ax(ax, values: np.ndarray, ylabel: str, ylim: tuple[float, float]):
    x = np.arange(len(METHODS))
    bars = ax.bar(
        x,
        values,
        width=0.62,
        color=COLORS,
        edgecolor="#333333",
        linewidth=0.6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, rotation=12, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Model structure")
    ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.8)
    ax.set_axisbelow(True)
    for b, v in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.04,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def main():
    _setup_tnr()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(5.6, 5.8), dpi=300, sharex=True
    )
    _one_bar_ax(
        ax_top,
        MACRO_F1,
        "Macro-average F1 (%)",
        ylim=(97.0, 100.05),
    )
    _one_bar_ax(
        ax_bot,
        TEST_ACC,
        "Overall test accuracy (%)",
        ylim=(97.0, 100.05),
    )
    fig.suptitle(
        "Single-branch vs dual-branch: key metrics (HUST)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_PATH, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"[OK] Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
