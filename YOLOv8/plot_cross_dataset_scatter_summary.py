"""
Cross-dataset summary scatter plots for the paper.

Figure content (all values from existing 3.3 / 3.4 experiments):
  (a) HUST: overall accuracy comparison (GAF-RGB vs STFT-RGB)
  (b) HUST: per-class recall comparison
  (c) CWRU: overall accuracy comparison (GAF vs SA-GAF dual-branch mean)
  (d) CWRU: per-class recall comparison

Fonts:
  - English: Times New Roman
  - Chinese: SimSun (宋体)

Outputs:
  D:\\C论文\\优化材料\\cross_dataset_scatter_summary.png
  D:\\C论文\\优化材料\\cross_dataset_scatter_summary_notext.png
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


YOLO_ROOT = Path(__file__).resolve().parent

CM_HUST_GAF = YOLO_ROOT / "runs" / "hust_gaf_cls" / "eval_dual_mean" / "confusion_matrix_counts.csv"
CM_HUST_STFT = YOLO_ROOT / "runs" / "hust_spec" / "eval_spec" / "confusion_matrix_counts.csv"
CM_CWRU_GAF = YOLO_ROOT / "runs" / "cwru_cls" / "cwru_dual_eval" / "confusion_matrix_counts.csv"
CM_CWRU_SAGAF = (
    YOLO_ROOT
    / "runs"
    / "classify"
    / "runs"
    / "cwru_cls"
    / "cwru_dual_eval_enh"
    / "confusion_matrix_counts.csv"
)

OUT_DIR = Path("D:/") / "C论文" / "优化材料"
OUT_PATH = OUT_DIR / "cross_dataset_scatter_summary.png"
OUT_PATH_NOTEXT = OUT_DIR / "cross_dataset_scatter_summary_notext.png"


def _font_props() -> tuple[FontProperties, FontProperties]:
    # Mixed fonts for Chinese/English readability in paper figures.
    simsun_candidates = [
        Path(r"C:\Windows\Fonts\simsun.ttc"),
        Path(r"C:\Windows\Fonts\simsum.ttc"),
    ]
    simsun = next((p for p in simsun_candidates if p.is_file()), None)
    fp_en = FontProperties(family="Times New Roman", size=9)
    if simsun is not None:
        fp_cn = FontProperties(fname=str(simsun), size=9)
    else:
        fp_cn = FontProperties(family="SimSun", size=9)
    return fp_en, fp_cn


def _setup_rc(fp_en: FontProperties) -> None:
    fonts = {f.name for f in fm.fontManager.ttflist}
    if "Times New Roman" in fonts:
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"]
        plt.rcParams["font.sans-serif"] = ["Times New Roman"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 9


def _read_cm(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8").strip().splitlines()
    return np.array([[int(x) for x in ln.split(",")] for ln in text], dtype=int)


def _overall_and_recall(cm: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, float]:
    correct = np.trace(cm)
    total = cm.sum()
    acc = correct / total if total > 0 else math.nan
    # recall per class: TP / row sum
    with np.errstate(divide="ignore", invalid="ignore"):
        rec = np.diag(cm) / cm.sum(axis=1)
        prec = np.diag(cm) / cm.sum(axis=0)
    rec[np.isnan(rec)] = 0.0
    prec[np.isnan(prec)] = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = 2 * prec * rec / (prec + rec)
    f1[np.isnan(f1)] = 0.0
    macro_f1 = float(f1.mean())
    return acc, rec, f1, macro_f1


def _scatter_style(ax, include_text: bool) -> None:
    ax.grid(True, linestyle="--", alpha=0.30)
    if not include_text:
        ax.tick_params(axis="both", which="both", labelbottom=False, labelleft=False)


def _draw(include_text: bool, fp_en: FontProperties, fp_cn: FontProperties, out_path: Path) -> None:
    cm_hust_gaf = _read_cm(CM_HUST_GAF)
    cm_hust_stft = _read_cm(CM_HUST_STFT)
    cm_cwru_gaf = _read_cm(CM_CWRU_GAF)
    cm_cwru_sagaf = _read_cm(CM_CWRU_SAGAF)

    hust_acc_gaf, hust_rec_gaf, _, hust_f1_macro_gaf = _overall_and_recall(cm_hust_gaf)
    hust_acc_stft, hust_rec_stft, _, hust_f1_macro_stft = _overall_and_recall(cm_hust_stft)
    cwru_acc_gaf, cwru_rec_gaf, _, cwru_f1_macro_gaf = _overall_and_recall(cm_cwru_gaf)
    cwru_acc_sagaf, cwru_rec_sagaf, _, cwru_f1_macro_sagaf = _overall_and_recall(cm_cwru_sagaf)

    fig, axes = plt.subplots(2, 2, figsize=(6.0, 5.0), dpi=300)

    # (a) HUST overall accuracy + macro F1
    ax = axes[0, 0]
    x = np.array([0, 1])
    # 两条折线：总体准确率 & 宏平均 F1
    ax.plot(
        x,
        [hust_acc_gaf * 100, hust_acc_stft * 100],
        color="#3498db",
        marker="o",
        label="总体准确率",
    )
    ax.plot(
        x,
        [hust_f1_macro_gaf * 100, hust_f1_macro_stft * 100],
        color="#e67e22",
        marker="^",
        label="宏平均 F1",
    )
    if include_text:
        ax.set_xticks(x, ["GAF-RGB\n双分支均值", "STFT-RGB"], fontproperties=fp_cn)
        # 纵轴同时承载两条曲线：总体准确率与宏平均 F1（均以百分数刻度）
        ax.set_ylabel("总体准确率 / 宏平均 F1 (%)", fontproperties=fp_cn)
        ax.set_title("（a）HUST 总体诊断性能对比", fontproperties=fp_cn, pad=4)
        leg = ax.legend(prop=fp_cn, loc="lower right", frameon=False)
        for text in leg.get_texts():
            text.set_fontproperties(fp_cn)
    _scatter_style(ax, include_text)
    ax.set_ylim(97.0, 100.5)

    # (b) HUST per-class recall
    ax = axes[0, 1]
    cls_labels = ["健康", "滚动体", "内圈", "外圈"]
    xx = np.arange(len(cls_labels))
    ax.plot(xx, hust_rec_gaf * 100, c="#3498db", marker="o", label="GAF-RGB")
    ax.plot(xx, hust_rec_stft * 100, c="#e67e22", marker="^", label="STFT-RGB")
    if include_text:
        ax.set_xticks(xx, cls_labels, fontproperties=fp_cn)
        ax.set_ylabel("召回率 (%)", fontproperties=fp_cn)
        ax.set_title("（b）HUST 各故障类别召回率对比", fontproperties=fp_cn, pad=4)
        leg = ax.legend(prop=fp_cn, loc="lower right", frameon=False)
        for text in leg.get_texts():
            text.set_fontproperties(fp_cn)
    _scatter_style(ax, include_text)
    ax.set_ylim(90.0, 101.0)

    # (c) CWRU overall accuracy + macro F1
    ax = axes[1, 0]
    x = np.array([0, 1])
    ax.plot(
        x,
        [cwru_acc_gaf * 100, cwru_acc_sagaf * 100],
        color="#9b59b6",
        marker="o",
        label="总体准确率",
    )
    ax.plot(
        x,
        [cwru_f1_macro_gaf * 100, cwru_f1_macro_sagaf * 100],
        color="#2ecc71",
        marker="^",
        label="宏平均 F1",
    )
    if include_text:
        ax.set_xticks(x, ["GAF\n双分支均值", "SA-GAF\n双分支均值"], fontproperties=fp_cn)
        ax.set_ylabel("总体准确率 / 宏平均 F1 (%)", fontproperties=fp_cn)
        ax.set_title("（c）CWRU 总体诊断性能对比", fontproperties=fp_cn, pad=4)
        leg = ax.legend(prop=fp_cn, loc="lower right", frameon=False)
        for text in leg.get_texts():
            text.set_fontproperties(fp_cn)
    _scatter_style(ax, include_text)
    ax.set_ylim(65.0, 101.0)

    # (d) CWRU per-class recall
    ax = axes[1, 1]
    # CWRU 类别未在图中命名，统一用 C0~C3 表示
    cls_labels_cwru = ["C0", "C1", "C2", "C3"]
    xx = np.arange(len(cls_labels_cwru))
    ax.plot(xx, cwru_rec_gaf * 100, c="#9b59b6", marker="o", label="GAF 双分支")
    ax.plot(xx, cwru_rec_sagaf * 100, c="#2ecc71", marker="^", label="SA-GAF 双分支")
    if include_text:
        ax.set_xticks(xx, cls_labels_cwru, fontproperties=fp_en)
        ax.set_ylabel("召回率 (%)", fontproperties=fp_cn)
        ax.set_title("（d）CWRU 各类别召回率对比", fontproperties=fp_cn, pad=4)
        leg = ax.legend(prop=fp_cn, loc="lower right", frameon=False)
        for text in leg.get_texts():
            text.set_fontproperties(fp_cn)
    _scatter_style(ax, include_text)
    ax.set_ylim(60.0, 101.0)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"[OK] Saved -> {out_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fp_en, fp_cn = _font_props()
    _setup_rc(fp_en)
    _draw(include_text=True, fp_en=fp_en, fp_cn=fp_cn, out_path=OUT_PATH)
    _draw(include_text=False, fp_en=fp_en, fp_cn=fp_cn, out_path=OUT_PATH_NOTEXT)


if __name__ == "__main__":
    main()

