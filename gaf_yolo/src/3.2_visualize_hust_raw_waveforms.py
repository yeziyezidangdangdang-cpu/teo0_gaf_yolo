"""
可视化 HUST-bearing 原始切片波形（用于解释 GAF 区分度不明显的原因）。

读取：
- data/processed/final_dataset_hust.csv
  每行结构: x_0..x_{L-1}, y_0.., z_0.., label（无表头）

输出：
- data/diagnostics/hust_waveforms_raw/label_<k>/wave_<k>_<idx>.png

说明：
- 这里的“原始”指 0.0_prepare_hust_dataset.py 切片后的时域波形（未做 1.1 的去均值/标准化）。
- 如果这些波形在不同标签间就很相似，那么 GAF 也往往只呈现弱差异（尤其在 per-series 归一化后）。
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import utils.project_paths as pp  # noqa: E402

from __future__ import annotations

import os
from pathlib import Path
import random

import numpy as np
import matplotlib.pyplot as plt


RAW_CSV = str(pp.FINAL_DATASET_HUST)
OUT_DIR = str(pp.DIAGNOSTICS / 'hust_waveforms_raw')

SEGMENT_LENGTH = 2048
MAX_PLOTS_PER_LABEL = 8
RANDOM_SEED = 42

# Optional: export a single figure to a custom path (for paper).
# Set to None to keep the original behavior (reservoir sampling plots).
EXPORT_SINGLE_ROW_IDX = 792
EXPORT_SINGLE_LABEL = 3
EXPORT_SINGLE_OUT_PNG = str(pp.PAPER_DRAW_ROOT / 'output' / 'wave_3_00792_xyz.png')
EXPORT_SINGLE_NO_TEXT_PNG = str(pp.PAPER_DRAW_ROOT / 'output' / 'wave_3_00792_xyz_notext.png')
EXPORT_SINGLE_PANEL_PNG = str(pp.PAPER_DRAW_ROOT / 'output' / 'wave_3_00792_single.png')


def iter_rows(csv_path: str):
    """逐行读取 CSV，返回 (row_index, values[3*L+1])，values 为 float32。"""
    expected = SEGMENT_LENGTH * 3 + 1
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            arr = np.fromstring(line, sep=",", dtype=np.float32)
            if arr.size != expected:
                raise ValueError(f"第 {i} 行列数异常：期望 {expected}，实际 {arr.size}")
            yield i, arr


def reservoir_sample_per_label(csv_path: str, k: int, seed: int):
    """
    流式按 label 做 reservoir sampling，避免一次性读入整个 CSV。
    返回 dict[label] -> list[(row_index, x, y, z)]，每类最多 k 条。
    """
    rng = random.Random(seed)
    # label -> (seen_count, samples[list])
    state: dict[int, tuple[int, list]] = {}

    L = SEGMENT_LENGTH
    for row_idx, arr in iter_rows(csv_path):
        lab = int(round(float(arr[-1])))
        seen, samples = state.get(lab, (0, []))
        seen += 1

        if len(samples) < k:
            x = arr[0:L].copy()
            y = arr[L : 2 * L].copy()
            z = arr[2 * L : 3 * L].copy()
            samples.append((row_idx, x, y, z))
        else:
            j = rng.randrange(seen)
            if j < k:
                x = arr[0:L].copy()
                y = arr[L : 2 * L].copy()
                z = arr[2 * L : 3 * L].copy()
                samples[j] = (row_idx, x, y, z)

        state[lab] = (seen, samples)

    out = {lab: samples for lab, (seen, samples) in state.items()}
    return out


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    return float(np.sqrt(np.mean(x * x)))


def plot_one(out_path: Path, x: np.ndarray, y: np.ndarray, z: np.ndarray, label: int, idx: int) -> None:
    # Paper-friendly layout:
    # - keep X/Y/Z only
    # - remove all rms text
    # - use gentle colors (low contrast) + distinguish by line style/width
    #   (avoid overly saturated colors in the reference figure)
    colors = {
        "x": "#4C78A8",  # muted blue
        "y": "#72B7B2",  # muted teal
        "z": "#F2CF5B",  # muted yellow
    }
    alpha = 0.90
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(x, lw=2.0, ls="-", color=colors["x"], alpha=alpha)
    axes[0].set_title(f"Label {label} | sample idx {idx} | X")

    axes[1].plot(y, lw=0.9, ls="-", color=colors["y"], alpha=alpha)
    axes[1].set_title("Y")

    axes[2].plot(z, lw=1.4, ls="--", color=colors["z"], alpha=alpha)
    axes[2].set_title("Z")
    axes[2].set_xlabel("sample")

    for ax in axes:
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_one_no_text(out_path: Path, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    """原版三子图布局，仅保留曲线与边框，去坐标、去所有文字。"""
    colors = {"x": "#4C78A8", "y": "#72B7B2", "z": "#F2CF5B"}
    alpha = 0.90
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(x, lw=2.0, ls="-", color=colors["x"], alpha=alpha)
    axes[1].plot(y, lw=0.9, ls="-", color=colors["y"], alpha=alpha)
    axes[2].plot(z, lw=1.4, ls="--", color=colors["z"], alpha=alpha)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_one_single_panel(
    out_path: Path, x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> None:
    """
    三轴波形绘于同一图内，无横纵坐标与刻度，仅保留曲线与边框，预留图例空间。
    参考“时序1/2/3”单图风格。
    """
    colors = {"x": "#4C78A8", "y": "#72B7B2", "z": "#F2CF5B"}
    alpha = 0.90
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    n = len(x)
    ax.plot(x, lw=2.0, ls="-", color=colors["x"], alpha=alpha, label="X")
    ax.plot(y, lw=1.0, ls="-", color=colors["y"], alpha=alpha, label="Y")
    ax.plot(z, lw=1.4, ls="--", color=colors["z"], alpha=alpha, label="Z")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.axis("off")
    # 保留外框，预留图例空间（图例框不显示文字时可作占位）
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    # Export one exact sample if configured.
    if EXPORT_SINGLE_OUT_PNG and EXPORT_SINGLE_ROW_IDX is not None and EXPORT_SINGLE_LABEL is not None:
        L = SEGMENT_LENGTH
        for row_idx, arr in iter_rows(RAW_CSV):
            if row_idx != EXPORT_SINGLE_ROW_IDX:
                continue
            # Note:
            # The "label" used in the diagnostic image file name/title may come from
            # an earlier label mapping version. For paper figures we keep the label
            # text consistent with the reference image and use the exact row index
            # waveform from RAW_CSV.
            x = arr[0:L].copy()
            y = arr[L : 2 * L].copy()
            z = arr[2 * L : 3 * L].copy()
            out_path = Path(EXPORT_SINGLE_OUT_PNG)
            plot_one(out_path, x, y, z, int(EXPORT_SINGLE_LABEL), row_idx)
            print(f"[HUST-WAVE] exported single plot -> {out_path}")
            if EXPORT_SINGLE_NO_TEXT_PNG:
                notext_path = Path(EXPORT_SINGLE_NO_TEXT_PNG)
                plot_one_no_text(notext_path, x, y, z)
                print(f"[HUST-WAVE] exported (no coords, no text) -> {notext_path}")
            if EXPORT_SINGLE_PANEL_PNG:
                panel_path = Path(EXPORT_SINGLE_PANEL_PNG)
                plot_one_single_panel(panel_path, x, y, z)
                print(f"[HUST-WAVE] exported single-panel (no axes) -> {panel_path}")
            return
        raise ValueError(f"Row index {EXPORT_SINGLE_ROW_IDX} not found in {RAW_CSV}")

    # Default behavior: reservoir sampling plots per label (diagnostic).
    os.makedirs(OUT_DIR, exist_ok=True)
    per_label_samples = reservoir_sample_per_label(RAW_CSV, MAX_PLOTS_PER_LABEL, RANDOM_SEED)
    labels = sorted(per_label_samples.keys())
    print(f"[HUST-WAVE] labels found: {labels} | max_plots_per_label={MAX_PLOTS_PER_LABEL}")

    for lab in labels:
        samples = per_label_samples.get(lab, [])
        print(f"[HUST-WAVE] label {lab}: plotting {len(samples)}")
        for row_idx, x, y, z in samples:
            out_path = Path(OUT_DIR) / f"label_{lab}" / f"wave_{lab}_{row_idx:05d}.png"
            plot_one(out_path, x, y, z, lab, row_idx)

    print(f"[HUST-WAVE] done. outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()

