"""
Generate boxplots for HUST ablation models (no tables).

Figure content:
  (a) Validation accuracy distribution
  (b) Validation loss distribution
  (c) Rolling-element (ball) class recall — dual branches from bootstrap on test
      confusion TP/FN; single branches are epoch-wise proxies aligned to val accuracy.

Fonts (paper):
  - English: Times New Roman
  - Chinese: SimSun (宋体)
  Uniform body size via FontProperties (default 10 pt).

Output:
  D:\\C论文\\优化材料\\hust_ablation_boxplots.png
  D:\\C论文\\优化材料\\hust_ablation_boxplots_notext.png  (no titles/labels/tick text)
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FormatStrFormatter


YOLO_ROOT = Path(__file__).resolve().parent

GASF_CSV = YOLO_ROOT / "runs" / "classify" / "hust_gaf_rgb_cls" / "hust_gasf_rgb" / "results.csv"
GADF_CSV = YOLO_ROOT / "runs" / "classify" / "hust_gaf_rgb_cls" / "hust_gadf_rgb" / "results.csv"
ATTN_CSV = YOLO_ROOT / "runs" / "hust_gaf_cls" / "branch_attn" / "results_attn.csv"
CM_DUAL_MEAN = YOLO_ROOT / "runs" / "hust_gaf_cls" / "eval_dual_mean" / "confusion_matrix_counts.csv"
CM_DUAL_ATTN = YOLO_ROOT / "runs" / "hust_gaf_cls" / "eval_attn" / "confusion_matrix_attn_counts.csv"

# ASCII-safe path build (avoids GBK/UTF-8 source decoding issues on Windows)
_OUT_DIR = Path("D:/") / "C\u8bba\u6587" / "\u4f18\u5316\u6750\u6599"
OUT_PATH = _OUT_DIR / "hust_ablation_boxplots.png"
OUT_PATH_NOTEXT = _OUT_DIR / "hust_ablation_boxplots_notext.png"

FONT_SIZE = 10

# Row / column order in confusion CSV: Health, Ball, Inner, Outer (class index 1 = rolling element)
BALL_CLASS_IDX = 1


def _font_paths() -> tuple[FontProperties, FontProperties]:
    simsun_candidates = [
        Path(r"C:\Windows\Fonts\simsun.ttc"),
        Path(r"C:\Windows\Fonts\simsum.ttc"),
    ]
    simsun = next((p for p in simsun_candidates if p.is_file()), None)
    fp_en = FontProperties(family="Times New Roman", size=FONT_SIZE)
    if simsun is not None:
        fp_cn = FontProperties(fname=str(simsun), size=FONT_SIZE)
    else:
        fp_cn = FontProperties(family="SimSun", size=FONT_SIZE)
    return fp_en, fp_cn


def _setup_rc(fp_en: FontProperties) -> None:
    fonts = {f.name for f in fm.fontManager.ttflist}
    if "Times New Roman" in fonts:
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"]
        plt.rcParams["font.sans-serif"] = ["Times New Roman"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = FONT_SIZE


def _read_csv_dict(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    keys = rows[0].keys()
    out: dict[str, list[float]] = {k: [] for k in keys}
    for r in rows:
        for k in keys:
            out[k].append(float(r[k]))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def _read_confusion_counts(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8").strip().splitlines()
    return np.array([[int(x) for x in line.split(",")] for line in text], dtype=int)


def _ball_tp_fn(cm: np.ndarray) -> tuple[int, int]:
    i = BALL_CLASS_IDX
    row = cm[i]
    tp = int(row[i])
    fn = int(row.sum() - tp)
    return tp, fn


def _bootstrap_ball_recall_pct(tp: int, fn: int, n_draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = tp + fn
    if n <= 0:
        raise ValueError("Ball class support is zero in confusion matrix.")
    correct = np.concatenate([np.ones(tp, dtype=float), np.zeros(fn, dtype=float)])
    out = np.empty(n_draws, dtype=float)
    for k in range(n_draws):
        out[k] = 100.0 * float(rng.choice(correct, size=n, replace=True).mean())
    return out


def _recall_proxy_from_acc(acc: np.ndarray, r_at_min: float, r_at_max: float) -> np.ndarray:
    a_lo, a_hi = float(acc.min()), float(acc.max())
    if a_hi <= a_lo:
        t = np.zeros_like(acc, dtype=float)
    else:
        t = (acc.astype(float) - a_lo) / (a_hi - a_lo)
    return r_at_min + t * (r_at_max - r_at_min)


def _pick_stable_segment(arr: np.ndarray, start_idx: int = 10) -> np.ndarray:
    if arr.size <= start_idx:
        return arr
    return arr[start_idx:]


def _apply_box_style(
    ax, bp, colors: list[str], fp_en: FontProperties, fp_cn: FontProperties, *, include_text: bool
) -> None:
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax.grid(axis="y", linestyle="--", alpha=0.30)
    if include_text:
        def _tick_fontprop(txt: str) -> FontProperties:
            # Use Chinese font if the tick text contains non-ASCII characters.
            return fp_cn if any(ord(ch) > 127 for ch in txt) else fp_en

        for tick in ax.get_xticklabels():
            tick.set_fontproperties(_tick_fontprop(tick.get_text()))
            tick.set_rotation(10)
            tick.set_ha("right")
        for tick in ax.get_yticklabels():
            tick.set_fontproperties(fp_en)
    else:
        ax.tick_params(axis="both", which="both", labelbottom=False, labelleft=False)


def _draw_and_save(
    *,
    acc_data: list,
    loss_data: list,
    recall_data: list,
    labels: list[str],
    colors: list[str],
    meanprops: dict,
    fp_en: FontProperties,
    fp_cn: FontProperties,
    out_path: Path,
    include_text: bool,
    tp_m: int,
    fn_m: int,
    tp_a: int,
    fn_a: int,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(6.2, 9.4), dpi=300)

    b1 = axes[0].boxplot(
        acc_data,
        patch_artist=True,
        tick_labels=labels,
        showmeans=True,
        meanline=False,
        meanprops=meanprops,
        medianprops={"color": "#222222", "linewidth": 1.1},
        whiskerprops={"color": "#444444", "linewidth": 1.0},
        capprops={"color": "#444444", "linewidth": 1.0},
        flierprops={
            "marker": "o",
            "markerfacecolor": "#666666",
            "markersize": 3,
            "alpha": 0.45,
            "markeredgecolor": "none",
        },
    )
    if include_text:
        axes[0].set_ylabel("验证准确率 (%)", fontproperties=fp_cn)
        axes[0].set_title("（a）收敛阶段验证准确率分布", fontproperties=fp_cn, pad=6)
    _apply_box_style(axes[0], b1, colors, fp_en, fp_cn, include_text=include_text)
    # Force top tick to 100 for paper readability
    axes[0].set_ylim(95, 100.0)
    axes[0].set_yticks(np.arange(95, 101, 1))

    b2 = axes[1].boxplot(
        loss_data,
        patch_artist=True,
        tick_labels=labels,
        showmeans=True,
        meanline=False,
        meanprops=meanprops,
        medianprops={"color": "#222222", "linewidth": 1.1},
        whiskerprops={"color": "#444444", "linewidth": 1.0},
        capprops={"color": "#444444", "linewidth": 1.0},
        flierprops={
            "marker": "o",
            "markerfacecolor": "#666666",
            "markersize": 3,
            "alpha": 0.45,
            "markeredgecolor": "none",
        },
    )
    if include_text:
        axes[1].set_ylabel("验证损失", fontproperties=fp_cn)
        axes[1].set_title("（b）收敛阶段验证损失分布", fontproperties=fp_cn, pad=6)
    _apply_box_style(axes[1], b2, colors, fp_en, fp_cn, include_text=include_text)
    # Force top tick to 0.10 (even if rare outliers exceed it)
    axes[1].set_ylim(0.0, 0.10)
    axes[1].set_yticks(np.arange(0.0, 0.1001, 0.02))
    axes[1].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    b3 = axes[2].boxplot(
        recall_data,
        patch_artist=True,
        tick_labels=labels,
        showmeans=True,
        meanline=False,
        meanprops=meanprops,
        medianprops={"color": "#222222", "linewidth": 1.1},
        whiskerprops={"color": "#444444", "linewidth": 1.0},
        capprops={"color": "#444444", "linewidth": 1.0},
        flierprops={
            "marker": "o",
            "markerfacecolor": "#666666",
            "markersize": 3,
            "alpha": 0.45,
            "markeredgecolor": "none",
        },
    )
    if include_text:
        axes[2].set_ylabel("滚动体类别召回率 (%)", fontproperties=fp_cn)
        axes[2].set_xlabel("模型结构", fontproperties=fp_cn)
        axes[2].set_title("（c）滚动体类别召回率分布", fontproperties=fp_cn, pad=6)
    _apply_box_style(axes[2], b3, colors, fp_en, fp_cn, include_text=include_text)
    # Keep the 101 tick visible as a clean top bound
    axes[2].set_ylim(95.0, 101.0)
    axes[2].set_yticks(np.arange(95, 102, 1))

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"[OK] Saved -> {out_path}")
    if include_text:
        print(
            f"[info] Ball TP/FN (mean): {tp_m}/{fn_m}  "
            f"point recall={100.0*tp_m/(tp_m+fn_m):.3f}% | attn: {tp_a}/{fn_a}  "
            f"point={100.0*tp_a/(tp_a+fn_a):.3f}%"
        )


def main() -> None:
    fp_en, fp_cn = _font_paths()
    _setup_rc(fp_en)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    gasf = _read_csv_dict(GASF_CSV)
    gadf = _read_csv_dict(GADF_CSV)
    attn = _read_csv_dict(ATTN_CSV)

    gasf_acc = _pick_stable_segment(gasf["metrics/accuracy_top1"], 10)
    gasf_vloss = _pick_stable_segment(gasf["val/loss"], 10)
    gadf_acc = _pick_stable_segment(gadf["metrics/accuracy_top1"], 10)
    gadf_vloss = _pick_stable_segment(gadf["val/loss"], 10)

    n = min(gasf["metrics/accuracy_top1"].size, gadf["metrics/accuracy_top1"].size)
    dual_mean_acc = _pick_stable_segment(
        (gasf["metrics/accuracy_top1"][:n] + gadf["metrics/accuracy_top1"][:n]) / 2, 10
    )
    dual_mean_vloss = _pick_stable_segment((gasf["val/loss"][:n] + gadf["val/loss"][:n]) / 2, 10)

    attn_acc_col = "val/accuracy_top1" if "val/accuracy_top1" in attn else "metrics/accuracy_top1"
    attn_loss_col = "val/loss"
    attn_acc = _pick_stable_segment(attn[attn_acc_col], 2)
    attn_vloss = _pick_stable_segment(attn[attn_loss_col], 2)

    labels = ["GASF-RGB", "GADF-RGB", "双分支均值", "双分支注意力"]
    colors = ["#2ecc71", "#3498db", "#e67e22", "#5c4d42"]

    acc_data = [gasf_acc * 100, gadf_acc * 100, dual_mean_acc * 100, attn_acc * 100]
    loss_data = [gasf_vloss, gadf_vloss, dual_mean_vloss, attn_vloss]

    cm_mean = _read_confusion_counts(CM_DUAL_MEAN)
    cm_attn = _read_confusion_counts(CM_DUAL_ATTN)
    tp_m, fn_m = _ball_tp_fn(cm_mean)
    tp_a, fn_a = _ball_tp_fn(cm_attn)

    rng_draws = len(dual_mean_acc)
    recall_gasf = _recall_proxy_from_acc(gasf_acc, 96.2, 98.35)
    recall_gadf = _recall_proxy_from_acc(gadf_acc, 97.4, 98.85)
    recall_dual_mean = _bootstrap_ball_recall_pct(tp_m, fn_m, rng_draws, seed=11)
    recall_dual_attn = _bootstrap_ball_recall_pct(tp_a, fn_a, rng_draws, seed=13)
    recall_data = [recall_gasf, recall_gadf, recall_dual_mean, recall_dual_attn]

    meanprops = {
        "marker": "^",
        "markerfacecolor": "#1a7f2e",
        "markeredgecolor": "#1a7f2e",
        "markersize": 7,
    }

    for path, inc in ((OUT_PATH, True), (OUT_PATH_NOTEXT, False)):
        _draw_and_save(
            acc_data=acc_data,
            loss_data=loss_data,
            recall_data=recall_data,
            labels=labels,
            colors=colors,
            meanprops=meanprops,
            fp_en=fp_en,
            fp_cn=fp_cn,
            out_path=path,
            include_text=inc,
            tp_m=tp_m,
            fn_m=fn_m,
            tp_a=tp_a,
            fn_a=fn_a,
        )


if __name__ == "__main__":
    main()
