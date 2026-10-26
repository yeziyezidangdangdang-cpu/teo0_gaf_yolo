"""Evaluate dual-branch SA-GAF (enhanced GASF + enhanced GADF) classifiers on the CWRU test set.

This script:
- Loads the best weights for enhanced GASF and GADF classifiers (trained by train_cwru_dual_enh.py)
- Iterates over the test split created by 2.0_prepare_yolov8_datasets_cwru_gaf_strict.py (enhanced branch)
- For each sample, runs both models, fuses probabilities by averaging, and updates a 4x4 confusion matrix
- Prints overall accuracy and per-class accuracy
- Saves confusion matrix (raw counts and normalized) to CSV and PNG

Usage:
    cd D:\TEP_2_repro\YOLOv8
    python eval_cwru_dual_enh.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

from cwru_paths import (
    CWRU_V2_IMGSZ,
    CWRU_V2_PROJECT,
    EVAL_DUAL_MEAN_ENH_DIR,
    GADF_DATA_ROOT_ENH,
    GASF_DATA_ROOT_ENH,
    RUNS_DIR,
    branch_weight_path,
)

DEFAULT_RUN = 1

LABELS = [0, 1, 2, 3]
CLASS_NAMES = {
    0: "Normal",
    1: "Inner race fault",
    2: "Outer race fault",
    3: "Rolling element fault",
}

OUT_DIR = EVAL_DUAL_MEAN_ENH_DIR


def get_weights_paths(run: int):
    return branch_weight_path("gasf_enh", run), branch_weight_path("gadf_enh", run)


def load_models(run: int = DEFAULT_RUN):
    gasf_pt, gadf_pt = get_weights_paths(run)
    if not gasf_pt.is_file():
        raise FileNotFoundError(f"GASF_ENH weights not found: {gasf_pt}")
    if not gadf_pt.is_file():
        raise FileNotFoundError(f"GADF_ENH weights not found: {gadf_pt}")
    gasf_model = YOLO(str(gasf_pt))
    gadf_model = YOLO(str(gadf_pt))
    return gasf_model, gadf_model


def check_data_integrity():
    """统计 test 集每类 GASF_ENH/GADF_ENH 数量及可配对数量，便于检查数据完整性。"""
    gasf_test_root = GASF_DATA_ROOT_ENH / "test"
    gadf_test_root = GADF_DATA_ROOT_ENH / "test"
    print("Data integrity (SA-GAF test set):")
    print("  class  gasf_enh_count  gadf_enh_count  paired  (paired = same idx in both)")
    for lab in LABELS:
        gasf_dir = gasf_test_root / str(lab)
        gadf_dir = gadf_test_root / str(lab)
        gasf_list = list(gasf_dir.glob("*.png")) if gasf_dir.is_dir() else []
        gadf_map = {p.name: p for p in gadf_dir.glob("*.png")} if gadf_dir.is_dir() else {}
        paired = sum(1 for p in gasf_list if gadf_map.get(p.name.replace("gasf_", "gadf_", 1)) is not None)
        print(f"  {lab}     {len(gasf_list):6d}         {len(gadf_map):6d}        {paired:6d}")
    print()


def evaluate_dual(run: int = DEFAULT_RUN):
    gasf_model, gadf_model = load_models(run)

    gasf_test_root = GASF_DATA_ROOT_ENH / "test"
    gadf_test_root = GADF_DATA_ROOT_ENH / "test"

    cm = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    total = 0

    for lab in LABELS:
        lab_str = str(lab)
        gasf_dir = gasf_test_root / lab_str
        gadf_dir = gadf_test_root / lab_str

        if not gasf_dir.is_dir() or not gadf_dir.is_dir():
            raise FileNotFoundError(f"Missing SA-GAF test folder for label {lab}: {gasf_dir} or {gadf_dir}")

        gasf_imgs = sorted([p for p in gasf_dir.glob("*.png") if p.is_file()])
        # GADF_ENH 文件名为 gadf_<label>_<idx>.png，GASF_ENH 为 gasf_<label>_<idx>.png，需按对应名配对
        gadf_map = {p.name: p for p in gadf_dir.glob("*.png") if p.is_file()}

        for gasf_img in gasf_imgs:
            # gasf_0_12345.png -> gadf_0_12345.png
            gadf_name = gasf_img.name.replace("gasf_", "gadf_", 1)
            gadf_img = gadf_map.get(gadf_name)
            if gadf_img is None:
                continue

            # Forward on both branches
            r1 = gasf_model(str(gasf_img), imgsz=CWRU_V2_IMGSZ, verbose=False)[0]
            r2 = gadf_model(str(gadf_img), imgsz=CWRU_V2_IMGSZ, verbose=False)[0]

            probs1 = r1.probs.data.cpu().numpy()
            probs2 = r2.probs.data.cpu().numpy()
            probs = (probs1 + probs2) / 2.0

            pred = int(np.argmax(probs))
            true = lab

            cm[true, pred] += 1
            total += 1

    return cm, total


def save_confusion_matrix(cm: np.ndarray, total: int):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Raw counts CSV
    counts_csv = OUT_DIR / "confusion_matrix_counts.csv"
    np.savetxt(counts_csv, cm, fmt="%d", delimiter=",")

    # Normalized by true class (avoid divide by zero: 某类无样本时该行和为 0)
    true_row_sums = cm.sum(axis=1, keepdims=True)
    safe_sums = np.where(true_row_sums == 0, 1, true_row_sums)
    cm_norm = cm.astype(float) / safe_sums
    norm_csv = OUT_DIR / "confusion_matrix_normalized.csv"
    np.savetxt(norm_csv, cm_norm, fmt="%.4f", delimiter=",")

    # Plot heatmap (mask invalid for display if needed)
    plot_mat = np.nan_to_num(cm_norm, nan=0.0, posinf=0.0, neginf=0.0)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(plot_mat, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels([f"{i}\n{CLASS_NAMES[i]}" for i in LABELS])
    ax.set_yticklabels([f"{i}:{CLASS_NAMES[i]}" for i in LABELS])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Dual-branch SA-GAF (GASF_enh+GADF_enh) confusion matrix (normalized)")

    # Annotate cells (use true_row_sums to show "-" when class has no samples)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            val = plot_mat[i, j]
            txt = f"{val:.2f}" if true_row_sums[i, 0] > 0 else "-"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color="white" if val > 0.5 else "black",
            )

    fig.tight_layout()
    fig_path = OUT_DIR / "confusion_matrix.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate dual-branch SA-GAF (GASF_enh+GADF_enh) on test set")
    parser.add_argument(
        "--run",
        type=int,
        default=DEFAULT_RUN,
        help=f"Run index (1=cwru_gasf_enh_enh/cwru_gadf_enh_enh, 2=cwru_gasf_enh_enh2/..., default {DEFAULT_RUN})",
    )
    args = parser.parse_args()

    check_data_integrity()
    gasf_pt, gadf_pt = get_weights_paths(args.run)
    print(f"Using GASF_ENH: {gasf_pt.name}, GADF_ENH: {gadf_pt.name}")
    print("Evaluating dual-branch SA-GAF (GASF_enh+GADF_enh) classifier on test set...")
    cm, total = evaluate_dual(args.run)

    overall_acc = cm.trace() / total if total > 0 else 0.0
    print(f"Total samples: {total}")
    print(f"Overall accuracy: {overall_acc:.4f}")

    # Per-class accuracy
    for lab in LABELS:
        true_total = cm[lab, :].sum()
        acc = cm[lab, lab] / true_total if true_total > 0 else 0.0
        print(f"Class {lab} ({CLASS_NAMES[lab]}): accuracy = {acc:.4f} ({cm[lab, lab]}/{true_total})")

    save_confusion_matrix(cm, total)
    print(f"Confusion matrix saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()

