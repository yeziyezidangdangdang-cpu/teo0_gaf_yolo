from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


YOLO_ROOT = Path(__file__).resolve().parent
OUT_DIR = YOLO_ROOT / "runs" / "hust_gaf_cls" / "figs_ablation"


def _font_simsun(size: int):
    # Windows 常见宋体：simsun.ttc；若缺失则回退到系统默认
    try:
        return FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=size)
    except Exception:
        return FontProperties(size=size)


def _font_times(size: int):
    return FontProperties(family="Times New Roman", size=size)


def plot_bar(
    methods: list[str],
    values: np.ndarray,
    ylabel_cn: str,
    out_path: Path,
    ylim: tuple[float, float] = (97.0, 100.1),
):
    x = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    bars = ax.bar(
        x,
        values,
        width=0.62,
        color=["#2ecc71", "#3498db", "#e67e22", "#9b59b6"],
        edgecolor="#333333",
        linewidth=0.6,
    )

    # 坐标轴命名（中文宋体）
    ax.set_xlabel("模型结构", fontproperties=_font_simsun(13))
    ax.set_ylabel(ylabel_cn, fontproperties=_font_simsun(13))

    # 刻度：保留刻度线；英文用 Times，中文用宋体
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    for t in ax.get_xticklabels():
        txt = t.get_text()
        # 若包含中文字符，使用宋体；否则 Times
        if any("\u4e00" <= ch <= "\u9fff" for ch in txt):
            t.set_fontproperties(_font_simsun(12))
        else:
            t.set_fontproperties(_font_times(12))
    for t in ax.get_yticklabels():
        t.set_fontproperties(_font_times(12))

    ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.8)
    ax.set_axisbelow(True)

    # 数值标注：Times
    for b, v in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.05,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontproperties=_font_times(11),
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main():
    # 说明：
    # - 若你论文表格中的数字更新，只需要改这里即可
    # - methods 为英文（Times），坐标轴为中文（宋体）
    methods = ["GASF-RGB", "GADF-RGB", "双分支-均值", "双分支-注意力"]

    # 测试集总体准确率 vs 宏平均 F1：二者不应相同（前者为样本加权整体，后者为类均衡平均）。
    # 若某类召回偏低，宏 F1 通常略低于总体准确率。
    test_acc = np.array([98.00, 99.60, 99.40, 99.50], dtype=float)
    macro_f1 = np.array([97.48, 99.44, 99.32, 99.39], dtype=float)

    plot_bar(
        methods=methods,
        values=test_acc,
        ylabel_cn="测试集总体准确率(%)",
        out_path=OUT_DIR / "hust_ablation_test_acc_cn.png",
        ylim=(97.5, 100.05),
    )
    plot_bar(
        methods=methods,
        values=macro_f1,
        ylabel_cn="宏平均F1(%)",
        out_path=OUT_DIR / "hust_ablation_macro_f1_cn.png",
        ylim=(97.5, 100.05),
    )

    print(f"Saved -> {OUT_DIR}")


if __name__ == "__main__":
    main()

