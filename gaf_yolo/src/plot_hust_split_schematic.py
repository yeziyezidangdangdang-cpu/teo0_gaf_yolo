"""
绘制 HUST 数据集按文件/时间段严格划分示意图。
对应论文图5：滑动窗口切片 + 按文件内时间顺序 70%/15%/15% 划分，避免数据泄漏。
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 中文显示：Windows 下常见中文字体
import matplotlib.font_manager as fm
_cjk_fonts = [f.name for f in fm.fontManager.ttflist]
for _f in ("Microsoft YaHei", "SimHei", "SimSun", "KaiTi", "FangSong"):
    if _f in _cjk_fonts:
        plt.rcParams["font.sans-serif"] = [_f]
        break
else:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]  # 无中文则用默认，图中可能显示方框
plt.rcParams["axes.unicode_minus"] = False

OUT_PATH = Path(r"D:/TEP_2/fig_hust_split_schematic.png")


def main():
    plt.rc("font", family=plt.rcParams["font.sans-serif"][0])
    fig, ax = plt.subplots(1, 1, figsize=(10, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.set_aspect("equal")
    ax.axis("off")

    # 颜色：训练集 / 验证集 / 测试集
    c_train = "#2ecc71"
    c_val = "#f1c40f"
    c_test = "#3498db"

    # 假设一个文件内共有 20 个“段”示意（实际为大量片段，这里简化）
    n_seg = 20
    train_end = int(n_seg * 0.70)   # 14
    val_end = int(n_seg * 0.85)     # 17
    # 0~13 训练, 14~16 验证, 17~19 测试

    seg_w = 0.38
    seg_h = 0.7
    y_center = 2.0
    x0 = 0.5
    for i in range(n_seg):
        x = x0 + i * (seg_w + 0.02)
        if i < train_end:
            color = c_train
        elif i < val_end:
            color = c_val
        else:
            color = c_test
        rect = mpatches.Rectangle((x, y_center - seg_h / 2), seg_w, seg_h, facecolor=color, edgecolor="#333", linewidth=0.6)
        ax.add_patch(rect)

    # 文件标注
    ax.text(0.2, y_center, "某文件\n连续信号", fontsize=9, va="center", ha="right")
    ax.plot([0.32, 0.45], [y_center, y_center], "k-", lw=1)
    ax.text(5.0, y_center + seg_h / 2 + 0.35, "按时间顺序划分（同一文件内不跨集合）", fontsize=9, ha="center")

    # 比例标注
    ax.text(x0 + (train_end - 0.5) * (seg_w + 0.02) / 2, y_center - seg_h / 2 - 0.25, "70% 训练集", fontsize=8, ha="center", color=c_train)
    ax.text(x0 + (train_end + val_end - 1) / 2 * (seg_w + 0.02), y_center - seg_h / 2 - 0.25, "15% 验证集", fontsize=8, ha="center", color="#b7950b")
    ax.text(x0 + (val_end + n_seg - 1) / 2 * (seg_w + 0.02), y_center - seg_h / 2 - 0.25, "15% 测试集", fontsize=8, ha="center", color=c_test)

    # 图例
    leg_y = 0.85
    ax.add_patch(mpatches.Rectangle((0.5, leg_y - 0.12), 0.35, 0.24, facecolor=c_train, edgecolor="#333", linewidth=0.5))
    ax.text(0.92, leg_y, "训练集 (70%)", fontsize=9, va="center")
    ax.add_patch(mpatches.Rectangle((2.2, leg_y - 0.12), 0.35, 0.24, facecolor=c_val, edgecolor="#333", linewidth=0.5))
    ax.text(2.62, leg_y, "验证集 (15%)", fontsize=9, va="center")
    ax.add_patch(mpatches.Rectangle((3.4, leg_y - 0.12), 0.35, 0.24, facecolor=c_test, edgecolor="#333", linewidth=0.5))
    ax.text(3.82, leg_y, "测试集 (15%)", fontsize=9, va="center")

    # 上方说明：滑动窗口
    ax.text(5.0, 3.5, "滑动窗口切片（窗长 2048，步长 64）→ 按文件内时间顺序 70% / 15% / 15% 划分", fontsize=9, ha="center")
    ax.text(5.0, 3.15, "索引下采样（约 37.5% 片段）+ 每类 1500 张上限", fontsize=8, ha="center", color="#555")

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
