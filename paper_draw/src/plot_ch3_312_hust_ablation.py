"""
Section 3.1.2 HUST ablation: performance table + macro-F1 / test-accuracy bar charts.

Data source: current weights (hust_gaf_rgb_cls) evaluated on
D:\\TEP_2\\gaf_yolo\\yolo_datasets\\hust_*_rgb_cls_strict, N_test=819.

Outputs -> D:\\TEP_2_repro\\paper_draw\\
  fig_ch3_312_hust_ablation_table.png (+ _nolabel)
  fig_ch3_312_hust_ablation_metrics.png (+ _nolabel)
  table_312_hust_ablation.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties

DRAW = Path(__file__).resolve().parents[1]

METHODS = ["GASF-RGB", "GADF-RGB", "双分支\n简单平均", "双分支\n注意力融合"]
METHODS_SHORT = ["GASF", "GADF", "双分支均值", "双分支注意力"]
COLORS = ["#2ecc71", "#3498db", "#e67e22", "#9b59b6"]

# Verified 2026-06-09 (strict temporal test, N=819)
VAL_ACC = np.array([99.12, 99.89, 100.00, 100.00], dtype=float)
TEST_ACC = np.array([99.27, 98.78, 99.76, 99.63], dtype=float)
MACRO_F1 = np.array([99.27, 98.78, 99.76, 99.63], dtype=float)


def _simsun(size: int):
    try:
        return FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=size)
    except Exception:
        return FontProperties(size=size)


def _times(size: int):
    return FontProperties(family="Times New Roman", size=size)


def save_table_csv():
    rows = [
        ("单分支 GASF-RGB", VAL_ACC[0], TEST_ACC[0], MACRO_F1[0], ""),
        ("单分支 GADF-RGB", VAL_ACC[1], TEST_ACC[1], MACRO_F1[1], ""),
        ("双分支简单平均", VAL_ACC[2], TEST_ACC[2], MACRO_F1[2], "验证集为两分支概率简单平均在 val 上的总体准确率"),
        ("双分支注意力融合", VAL_ACC[3], TEST_ACC[3], MACRO_F1[3], "注意力模块约 10 epoch，GASF/GADF 分支冻结"),
    ]
    df = pd.DataFrame(
        rows,
        columns=["诊断方法", "验证集最高准确率(%)", "测试集总体准确率(%)", "宏平均F1(%)", "备注"],
    )
    out = DRAW / "table_312_hust_ablation.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Saved {out}")


def plot_table(labeled: bool):
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.axis("off")

    col_labels = [
        "诊断方法",
        "验证集最高\n准确率(%)",
        "测试集总体\n准确率(%)",
    ]
    row_labels = [
        "单分支 GASF-RGB",
        "单分支 GADF-RGB",
        "双分支简单平均",
        "双分支注意力融合¹",
    ]
    cell_text = [
        [f"{VAL_ACC[i]:.2f}", f"{TEST_ACC[i]:.2f}"] for i in range(4)
    ]

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10 if labeled else 9)
    table.scale(1.0, 1.55)
    for key, cell in table.get_celld().items():
        row, col = key
        txt = cell.get_text().get_text()
        if col == -1 or (col >= 0 and row == 0):
            cell.set_text_props(fontproperties=_simsun(10 if labeled else 9))
        elif "GASF" in txt or "GADF" in txt or txt.replace(".", "").isdigit():
            cell.set_text_props(fontproperties=_times(10 if labeled else 9))
        else:
            cell.set_text_props(fontproperties=_simsun(10 if labeled else 9))

    if labeled:
        ax.set_title(
            "HUST 稳态数据集各方案性能对比",
            fontproperties=_simsun(12),
            pad=12,
        )
        note = (
            "¹ 注意力融合模块在冻结预训练 GASF-RGB / GADF-RGB 分支的前提下仅训练约 10 个 epoch；"
            "Ball 类召回率约 99.51%，总体性能略低于简单平均。"
        )
        ax.text(
            0.5, -0.08, note,
            transform=ax.transAxes,
            ha="center", va="top",
            fontproperties=_simsun(8),
            wrap=True,
        )

    suffix = "" if labeled else "_nolabel"
    out = DRAW / f"fig_ch3_312_hust_ablation_table{suffix}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Saved {out}")


def _bar_ax(ax, values: np.ndarray, ylabel: str, panel_label: str, labeled: bool):
    x = np.arange(len(METHODS_SHORT))
    bars = ax.bar(x, values, width=0.62, color=COLORS, edgecolor="#333333", linewidth=0.6)
    ax.set_ylim(97.0, 100.08)
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.8)
    ax.set_axisbelow(True)

    if labeled:
        ax.set_ylabel(ylabel, fontproperties=_simsun(11))
        ax.set_xlabel("模型结构", fontproperties=_simsun(11))
        ax.set_xticks(x)
        ax.set_xticklabels(METHODS_SHORT, fontproperties=_simsun(10))
        for t in ax.get_yticklabels():
            t.set_fontproperties(_times(10))
        for b, v in zip(bars, values):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.04,
                f"{v:.2f}",
                ha="center", va="bottom",
                fontproperties=_times(9),
            )
        ax.text(-0.12, 1.02, panel_label, transform=ax.transAxes, fontproperties=_times(11), va="bottom")
    else:
        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    return bars


def plot_metrics(labeled: bool):
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(5.6, 5.8), dpi=300, sharex=True)
    _bar_ax(ax_a, MACRO_F1, "宏平均F1(%)", "(a)", labeled)
    _bar_ax(ax_b, TEST_ACC, "测试集总体准确率(%)", "(b)", labeled)

    if labeled:
        fig.suptitle(
            "HUST 稳态：单分支与双分支关键指标对比",
            fontproperties=_simsun(12),
            y=1.01,
        )

    fig.tight_layout()
    suffix = "" if labeled else "_nolabel"
    out = DRAW / f"fig_ch3_312_hust_ablation_metrics{suffix}.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    DRAW.mkdir(parents=True, exist_ok=True)
    save_table_csv()
    for labeled in (True, False):
        plot_table(labeled)
        plot_metrics(labeled)


if __name__ == "__main__":
    main()
