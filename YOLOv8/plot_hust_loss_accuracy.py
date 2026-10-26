"""
Plot loss/accuracy curves of four models on the HUST dataset (paper figure).

Data sources:
- Single-branch GASF-RGB / GADF-RGB: runs/classify/hust_gaf_rgb_cls/hust_*_rgb/results.csv
- Dual-branch simple average: mean of the two single-branch curves
- Dual-branch attention: runs/hust_gaf_cls/branch_attn/results_attn.csv

Output:
- fig_hust_loss_accuracy_en.png  (English, repo root)
- fig_hust_loss_accuracy_cn.png  (Chinese, D:\\C论文\\优化材料)
- fig_hust_loss_accuracy_cn_notext_4k.png  (no text, 3840 px wide, D:\\C论文\\优化材料)

Inset improvements:
- Highlighted zoom rectangle on the main loss axes (dashed border + light fill).
- Inset placed upper-right; inset y-axis ticks on the RIGHT to avoid clashing with main y-axis labels.
- Legend moved to upper-left so it does not overlap the inset.
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
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

YOLO_ROOT = Path(__file__).resolve().parent
GASF_CSV = YOLO_ROOT / "runs" / "classify" / "hust_gaf_rgb_cls" / "hust_gasf_rgb" / "results.csv"
GADF_CSV = YOLO_ROOT / "runs" / "classify" / "hust_gaf_rgb_cls" / "hust_gadf_rgb" / "results.csv"
ATTN_CSV = YOLO_ROOT / "runs" / "hust_gaf_cls" / "branch_attn" / "results_attn.csv"

OUT_EN = YOLO_ROOT / "fig_hust_loss_accuracy_en.png"
_OUT_DIR = Path("D:/") / "C\u8bba\u6587" / "\u4f18\u5316\u6750\u6599"
OUT_CN = _OUT_DIR / "fig_hust_loss_accuracy_cn.png"
OUT_CN_NOTEXT_4K = _OUT_DIR / "fig_hust_loss_accuracy_cn_notext_4k.png"

CM = 1 / 2.54
FIG_W, FIG_H = 8.6 * CM, 9.2 * CM  # slightly taller for inset + legend

# 4K export: target width in pixels (height follows figure aspect ratio)
TARGET_PX_WIDTH_4K = 3840

_fonts = {f.name for f in fm.fontManager.ttflist}


def _fp_tnr(size: float) -> FontProperties | None:
    if "Times New Roman" not in _fonts:
        return None
    return FontProperties(family="Times New Roman", size=size)


def _fp_mixed_cn(size: float) -> tuple[FontProperties, FontProperties]:
    """Chinese Songti + English TNR for axis text."""
    fp_en = _fp_tnr(size) or FontProperties(size=size)
    simsun = Path(r"C:\Windows\Fonts\simsun.ttc")
    if simsun.is_file():
        fp_cn = FontProperties(fname=str(simsun), size=size)
    else:
        fp_cn = FontProperties(family="SimSun", size=size)
    return fp_en, fp_cn


def _apply_ticks_font(ax, fp) -> None:
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontproperties(fp)


def read_csv_as_dict(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    cols: dict[str, list] = {k: [] for k in rows[0].keys()}
    for r in rows:
        for k, v in r.items():
            try:
                cols[k].append(float(v))
            except Exception:
                cols[k].append(v)
    out: dict[str, np.ndarray] = {}
    for k, vs in cols.items():
        if vs and isinstance(vs[0], (float, int)):
            out[k] = np.asarray(vs, dtype=float)
        else:
            out[k] = np.asarray(vs)
    return out


def _plot_figure(
    *,
    chinese: bool,
    out_path: Path,
    notext: bool = False,
    dpi: int = 300,
    pixel_width: int | None = None,
) -> None:
    fig_w_in, fig_h_in = FIG_W, FIG_H
    if pixel_width is not None:
        fig_w_in = float(pixel_width) / float(dpi)
        fig_h_in = fig_w_in * (FIG_H / FIG_W)
    line_scale = (fig_w_in / FIG_W) ** 0.35
    lw_main = max(1.2, 1.2 * line_scale)
    lw_inset = max(1.05, 1.05 * line_scale)

    if chinese:
        fp_en, fp_cn = _fp_mixed_cn(8.0)
        lbl_train = "训练损失"
        lbl_val = "验证准确率"
        lbl_epoch = "迭代轮次"
        lbl_zoom = "局部放大"
        leg = {
            "GASF-RGB": "GASF-RGB",
            "GADF-RGB": "GADF-RGB",
            "avg": "双分支简单平均",
            "attn": "双分支注意力",
        }
        tick_fp = fp_cn
    else:
        fp_en = _fp_tnr(8.0)
        fp_cn = fp_en
        lbl_train = "Training loss"
        lbl_val = "Validation accuracy"
        lbl_epoch = "Epoch"
        lbl_zoom = "Zoom"
        leg = {
            "GASF-RGB": "GASF-RGB",
            "GADF-RGB": "GADF-RGB",
            "avg": "Dual-branch (avg.)",
            "attn": "Dual-branch (attn.)",
        }
        tick_fp = fp_en or FontProperties(size=8.0)

    if notext:
        lbl_train = lbl_val = lbl_epoch = lbl_zoom = ""
        leg = {k: "" for k in ("GASF-RGB", "GADF-RGB", "avg", "attn")}

    tick_fs = max(6, int(round(6 * (fig_w_in / FIG_W) ** 0.25)))

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = float(max(8.0, 8.0 * (fig_w_in / FIG_W) ** 0.2))
    if fp_en:
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"]
        plt.rcParams["font.sans-serif"] = ["Times New Roman"]

    gasf = read_csv_as_dict(GASF_CSV)
    gadf = read_csv_as_dict(GADF_CSV)
    dual_mean = {
        "epoch": gasf["epoch"],
        "train/loss": (gasf["train/loss"] + gadf["train/loss"]) / 2,
        "metrics/accuracy_top1": (gasf["metrics/accuracy_top1"] + gadf["metrics/accuracy_top1"]) / 2,
        "val/loss": (gasf["val/loss"] + gadf["val/loss"]) / 2,
    }
    attn = read_csv_as_dict(ATTN_CSV) if ATTN_CSV.is_file() else None

    colors = {
        "GASF-RGB": "#2ecc71",
        "GADF-RGB": "#3498db",
        "avg": "#e74c3c",
        "attn": "#9b59b6",
    }
    linestyles = {"GASF-RGB": "-", "GADF-RGB": "-", "avg": "--", "attn": ":"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_w_in, fig_h_in), sharex=True)
    fig.subplots_adjust(hspace=0.12)

    ax1.plot(
        gasf["epoch"],
        gasf["train/loss"],
        color=colors["GASF-RGB"],
        ls=linestyles["GASF-RGB"],
        lw=lw_main,
        label=leg["GASF-RGB"],
        zorder=3,
    )
    ax1.plot(
        gadf["epoch"],
        gadf["train/loss"],
        color=colors["GADF-RGB"],
        ls=linestyles["GADF-RGB"],
        lw=lw_main,
        label=leg["GADF-RGB"],
        zorder=3,
    )
    ax1.plot(
        dual_mean["epoch"],
        dual_mean["train/loss"],
        color=colors["avg"],
        ls=linestyles["avg"],
        lw=lw_main,
        label=leg["avg"],
        zorder=3,
    )
    if attn is not None:
        ax1.plot(
            attn["epoch"],
            attn["train/loss"],
            color=colors["attn"],
            ls=linestyles["attn"],
            lw=lw_main,
            label=leg["attn"],
            zorder=3,
        )

    if not notext:
        ax1.set_ylabel(lbl_train, fontproperties=fp_cn if chinese else (fp_en or tick_fp))
        ax1.legend(
            loc="upper left",
            fontsize=max(5, tick_fs - 1),
            frameon=True,
            fancybox=False,
            edgecolor="#cccccc",
            prop=tick_fp,
        )
    ax1.set_ylim(bottom=0)
    ax1.tick_params(labelsize=tick_fs)
    if notext:
        ax1.tick_params(axis="both", which="both", labelbottom=False, labelleft=False)
    else:
        _apply_ticks_font(ax1, tick_fp)
    ax1.grid(True, linestyle="--", alpha=0.25, linewidth=0.6)

    # Zoom window (match late-epoch convergence band)
    zx0, zx1 = 22.0, 40.0
    zy0, zy1 = 0.0, 0.11
    zoom_rect = Rectangle(
        (zx0, zy0),
        zx1 - zx0,
        zy1 - zy0,
        linewidth=max(1.4, 1.4 * line_scale),
        edgecolor="#c0392b",
        facecolor="#fadbd8",
        alpha=0.28,
        linestyle="--",
        zorder=4,
    )
    ax1.add_patch(zoom_rect)

    # Inset upper-right: y ticks on the right → no overlap with main left y-axis labels
    axins = inset_axes(ax1, width="44%", height="46%", loc="upper right", borderpad=0.45)
    for key, ep, loss in (
        ("GASF-RGB", gasf["epoch"], gasf["train/loss"]),
        ("GADF-RGB", gadf["epoch"], gadf["train/loss"]),
        ("avg", dual_mean["epoch"], dual_mean["train/loss"]),
    ):
        axins.plot(
            ep,
            loss,
            color=colors[key],
            ls=linestyles[key],
            lw=lw_inset,
        )
    if attn is not None:
        axins.plot(attn["epoch"], attn["train/loss"], color=colors["attn"], ls=linestyles["attn"], lw=lw_inset)

    axins.set_xlim(zx0, zx1)
    axins.set_ylim(zy0, zy1)
    axins.grid(True, linestyle="--", alpha=0.35, linewidth=max(0.55, 0.55 * line_scale))
    if not notext:
        axins.set_title(lbl_zoom, fontproperties=tick_fp, fontsize=max(6, tick_fs + 1), pad=3)
    axins.yaxis.tick_right()
    axins.yaxis.set_ticks_position("right")
    axins.yaxis.set_label_position("right")
    axins.tick_params(axis="both", labelsize=tick_fs, pad=2)
    if notext:
        axins.tick_params(axis="both", which="both", labelbottom=False, labelleft=False, labelright=False)
    else:
        _apply_ticks_font(axins, tick_fp)

    mark_inset(
        ax1,
        axins,
        loc1=2,
        loc2=4,
        fc="none",
        ec="#c0392b",
        lw=max(1.1, 1.1 * line_scale),
        linestyle="--",
    )

    ax2.plot(
        gasf["epoch"],
        gasf["metrics/accuracy_top1"],
        color=colors["GASF-RGB"],
        ls=linestyles["GASF-RGB"],
        lw=lw_main,
        label=leg["GASF-RGB"],
    )
    ax2.plot(
        gadf["epoch"],
        gadf["metrics/accuracy_top1"],
        color=colors["GADF-RGB"],
        ls=linestyles["GADF-RGB"],
        lw=lw_main,
        label=leg["GADF-RGB"],
    )
    ax2.plot(
        dual_mean["epoch"],
        dual_mean["metrics/accuracy_top1"],
        color=colors["avg"],
        ls=linestyles["avg"],
        lw=lw_main,
        label=leg["avg"],
    )
    if attn is not None:
        acc_col = next((c for c in ("val/accuracy_top1", "metrics/accuracy_top1") if c in attn), list(attn.keys())[2])
        ax2.plot(attn["epoch"], attn[acc_col], color=colors["attn"], ls=linestyles["attn"], lw=lw_main, label=leg["attn"])

    if not notext:
        ax2.set_xlabel(lbl_epoch, fontproperties=fp_cn if chinese else (fp_en or tick_fp))
        ax2.set_ylabel(lbl_val, fontproperties=fp_cn if chinese else (fp_en or tick_fp))
    ax2.set_ylim(0.8, 1.02)
    ax2.tick_params(labelsize=tick_fs)
    if notext:
        ax2.tick_params(axis="both", which="both", labelbottom=False, labelleft=False)
    else:
        _apply_ticks_font(ax2, tick_fp)
    ax2.grid(True, linestyle="--", alpha=0.25, linewidth=max(0.6, 0.6 * line_scale))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if pixel_width is not None:
        # 固定整图尺寸，避免 tight 裁切导致达不到目标像素宽度
        w_in, h_in = fig.get_size_inches()
        fig.savefig(
            out_path,
            dpi=dpi,
            bbox_inches=Bbox.from_bounds(0, 0, w_in, h_in),
            pad_inches=0,
        )
    else:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[OK] Saved -> {out_path}")


def main() -> None:
    if not GASF_CSV.is_file():
        raise FileNotFoundError(f"GASF results not found: {GASF_CSV}")
    if not GADF_CSV.is_file():
        raise FileNotFoundError(f"GADF results not found: {GADF_CSV}")
    _plot_figure(chinese=False, out_path=OUT_EN)
    _plot_figure(chinese=True, out_path=OUT_CN)
    _plot_figure(
        chinese=True,
        out_path=OUT_CN_NOTEXT_4K,
        notext=True,
        dpi=300,
        pixel_width=TARGET_PX_WIDTH_4K,
    )


if __name__ == "__main__":
    main()
