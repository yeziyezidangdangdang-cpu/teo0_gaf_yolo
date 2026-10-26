import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ==== 1. 把这里的数据改成你表2里的真实数值 ====
methods = ["GASF-RGB", "GADF-RGB", "Dual-Mean", "Dual-Attn"]

# 示例：验证集最高准确率(%)
val_acc = np.array([98.6, 99.7, 99.4, 99.9])

# 示例：测试集总体准确率(%)
test_acc = np.array([98.0, 99.5, 99.4, 99.5])

# 如果有宏平均 F1，可以单独画一张；没有就先画精度对比
macro_f1 = np.array([98.0, 99.5, 99.4, 99.5])  # 按需要修改，或删掉后面的代码

# ==== 2. 通用画图函数 ====
def plot_bar(y, ylabel, out_path, ylim=(97.0, 100.1)):
    x = np.arange(len(methods))
    width = 0.6

    fig, ax = plt.subplots(figsize=(5.2, 3.2))  # 期刊常用比例

    bars = ax.bar(x, y, width=width, color=["#3498db", "#2ecc71", "#e67e22", "#9b59b6"])

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # 在柱子上方标注数值
    for b, v in zip(bars, y):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.05,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print("Saved:", out_path)


def main():
    root = Path(r"D:\TEP_2_repro\YOLOv8\runs\hust_gaf_cls")  # 自己调整输出目录

    plot_bar(test_acc, "Test accuracy (%)", root / "hust_ablation_test_acc.png")
    plot_bar(val_acc, "Best val accuracy (%)", root / "hust_ablation_val_acc.png")
    plot_bar(macro_f1, "Macro F1 (%)", root / "hust_ablation_macro_f1.png")


if __name__ == "__main__":
    main()