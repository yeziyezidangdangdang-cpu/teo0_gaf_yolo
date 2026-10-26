"""
Evaluate CWRU v2 dual-branch (GASF + GADF) with 4-class simple mean fusion.

p_fused = (p_gasf + p_gadf) / 2  → argmax → 预测类

输出目录：runs/classify/6_edit/cwru_v2_cls/eval_dual_mean/
并生成 table3 风格 CSV/MD（含单分支 test + 双分支简单平均 test）。

Usage:
    cd D:\\TEP_2_repro\\YOLOv8
    python eval_cwru_dual.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

from cwru_paths import (
    CWRU_V2_IMGSZ,
    EVAL_DUAL_MEAN_DIR,
    GADF_DATA_ROOT,
    GASF_DATA_ROOT,
    TABLE3_CSV,
    TABLE3_MD,
    branch_weight_path,
    branch_results_csv,
)

DEFAULT_RUN = 1

LABELS = [0, 1, 2, 3]
CLASS_NAMES = {
    0: "Normal",
    1: "Inner race fault",
    2: "Outer race fault",
    3: "Rolling element fault",
}

OUT_DIR = EVAL_DUAL_MEAN_DIR


def get_weights_paths(run: int = DEFAULT_RUN):
    return branch_weight_path("gasf", run), branch_weight_path("gadf", run)


def load_models(run: int = DEFAULT_RUN):
    gasf_pt, gadf_pt = get_weights_paths(run)
    if not gasf_pt.is_file():
        raise FileNotFoundError(f"GASF weights not found: {gasf_pt}")
    if not gadf_pt.is_file():
        raise FileNotFoundError(f"GADF weights not found: {gadf_pt}")
    return YOLO(str(gasf_pt)), YOLO(str(gadf_pt))


def read_best_val_accuracy(branch: str) -> float:
    csv_path = branch_results_csv(branch)
    if not csv_path.is_file():
        return float("nan")
    df = pd.read_csv(csv_path)
    col = "metrics/accuracy_top1"
    if col not in df.columns:
        return float("nan")
    return float(df[col].max()) * 100.0


def check_data_integrity():
    gasf_test_root = GASF_DATA_ROOT / "test"
    gadf_test_root = GADF_DATA_ROOT / "test"
    print("Data integrity (test set):")
    print("  class  gasf_count  gadf_count  paired")
    for lab in LABELS:
        gasf_dir = gasf_test_root / str(lab)
        gadf_dir = gadf_test_root / str(lab)
        gasf_list = list(gasf_dir.glob("*.png")) if gasf_dir.is_dir() else []
        gadf_map = {p.name: p for p in gadf_dir.glob("*.png")} if gadf_dir.is_dir() else {}
        paired = sum(1 for p in gasf_list if gadf_map.get(p.name.replace("gasf_", "gadf_", 1)))
        print(f"  {lab}     {len(gasf_list):6d}     {len(gadf_map):6d}     {paired:6d}")
    print()


def _evaluate_single_branch(model: YOLO, data_root: Path) -> tuple[np.ndarray, int]:
    cm = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    total = 0
    test_root = data_root / "test"
    for lab in LABELS:
        img_dir = test_root / str(lab)
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.glob("*.png")):
            r = model(str(img), imgsz=CWRU_V2_IMGSZ, verbose=False)[0]
            pred = int(np.argmax(r.probs.data.cpu().numpy()))
            cm[lab, pred] += 1
            total += 1
    return cm, total


def evaluate_dual(run: int = DEFAULT_RUN):
    gasf_model, gadf_model = load_models(run)
    gasf_test_root = GASF_DATA_ROOT / "test"
    gadf_test_root = GADF_DATA_ROOT / "test"

    cm = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    total = 0

    for lab in LABELS:
        gasf_dir = gasf_test_root / str(lab)
        gadf_dir = gadf_test_root / str(lab)
        if not gasf_dir.is_dir() or not gadf_dir.is_dir():
            raise FileNotFoundError(f"Missing test folder for label {lab}")

        gasf_imgs = sorted(gasf_dir.glob("*.png"))
        gadf_map = {p.name: p for p in gadf_dir.glob("*.png")}

        for gasf_img in gasf_imgs:
            gadf_name = gasf_img.name.replace("gasf_", "gadf_", 1)
            gadf_img = gadf_map.get(gadf_name)
            if gadf_img is None:
                continue

            r1 = gasf_model(str(gasf_img), imgsz=CWRU_V2_IMGSZ, verbose=False)[0]
            r2 = gadf_model(str(gadf_img), imgsz=CWRU_V2_IMGSZ, verbose=False)[0]
            probs = (r1.probs.data.cpu().numpy() + r2.probs.data.cpu().numpy()) / 2.0
            pred = int(np.argmax(probs))
            cm[lab, pred] += 1
            total += 1

    return cm, total


def metrics_from_cm(cm: np.ndarray, total: int) -> dict:
    overall = cm.trace() / total if total > 0 else 0.0
    recalls = []
    for lab in LABELS:
        row = cm[lab, :].sum()
        recalls.append(cm[lab, lab] / row if row > 0 else 0.0)
    return {"overall": overall, "recalls": recalls}


def save_confusion_matrix(cm: np.ndarray, out_dir: Path, title: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / "confusion_matrix_counts.csv", cm, fmt="%d", delimiter=",")
    row_sums = cm.sum(axis=1, keepdims=True)
    safe = np.where(row_sums == 0, 1, row_sums)
    cm_norm = cm.astype(float) / safe
    np.savetxt(out_dir / "confusion_matrix_normalized.csv", cm_norm, fmt="%.4f", delimiter=",")

    plot_mat = np.nan_to_num(cm_norm, nan=0.0)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(plot_mat, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels([f"{i}\n{CLASS_NAMES[i]}" for i in LABELS])
    ax.set_yticklabels([f"{i}:{CLASS_NAMES[i]}" for i in LABELS])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            val = plot_mat[i, j]
            txt = f"{val:.2f}" if row_sums[i, 0] > 0 else "-"
            ax.text(j, i, txt, ha="center", va="center", color="white" if val > 0.5 else "black")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)


def export_table3(
    gasf_val: float,
    gadf_val: float,
    gasf_test: dict,
    gadf_test: dict,
    dual_test: dict,
) -> None:
    rows = [
        ["方法", "总体准确率(%)", "类0召回(%)", "类1召回(%)", "类2召回(%)", "类3召回(%)", "数据来源"],
        [
            "v2 GAF单分支GASF",
            f"{gasf_test['overall'] * 100:.2f}",
            *[f"{r * 100:.2f}" for r in gasf_test["recalls"]],
            "测试集",
        ],
        [
            "v2 GAF单分支GADF",
            f"{gadf_test['overall'] * 100:.2f}",
            *[f"{r * 100:.2f}" for r in gadf_test["recalls"]],
            "测试集",
        ],
        [
            "v2 GAF双分支简单平均",
            f"{dual_test['overall'] * 100:.2f}",
            *[f"{r * 100:.2f}" for r in dual_test["recalls"]],
            "测试集",
        ],
        ["v2 GASF验证集最佳", f"{gasf_val:.2f}", "—", "—", "—", "—", "验证集"],
        ["v2 GADF验证集最佳", f"{gadf_val:.2f}", "—", "—", "—", "—", "验证集"],
    ]
    TABLE3_CSV.parent.mkdir(parents=True, exist_ok=True)
    with TABLE3_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(rows)

    md = [
        "# CWRU v2 双分支性能（A 点修正管线）",
        "",
        f"- GASF 验证集最佳准确率：{gasf_val:.2f}%",
        f"- GADF 验证集最佳准确率：{gadf_val:.2f}%",
        f"- 测试集双分支简单平均：{dual_test['overall'] * 100:.2f}%",
        "",
        "详见 `table3_cwru_v2_dual_branch_performance.csv`",
    ]
    TABLE3_MD.write_text("\n".join(md), encoding="utf-8")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CWRU v2 dual-branch simple mean eval")
    parser.add_argument("--run", type=int, default=DEFAULT_RUN)
    args = parser.parse_args()

    check_data_integrity()
    gasf_pt, gadf_pt = get_weights_paths(args.run)
    print(f"Using GASF: {gasf_pt}")
    print(f"Using GADF: {gadf_pt}")

    gasf_model, gadf_model = load_models(args.run)

    print("\n=== Single-branch test (GASF) ===")
    cm_g, n_g = _evaluate_single_branch(gasf_model, GASF_DATA_ROOT)
    m_g = metrics_from_cm(cm_g, n_g)
    print(f"Overall: {m_g['overall']:.4f} (N={n_g})")

    print("\n=== Single-branch test (GADF) ===")
    cm_d, n_d = _evaluate_single_branch(gadf_model, GADF_DATA_ROOT)
    m_d = metrics_from_cm(cm_d, n_d)
    print(f"Overall: {m_d['overall']:.4f} (N={n_d})")

    print("\n=== Dual-branch simple mean (test) ===")
    cm_dual, n_dual = evaluate_dual(args.run)
    m_dual = metrics_from_cm(cm_dual, n_dual)
    print(f"Overall: {m_dual['overall']:.4f} (N={n_dual})")
    for lab in LABELS:
        row = cm_dual[lab, :].sum()
        acc = cm_dual[lab, lab] / row if row > 0 else 0.0
        print(f"  Class {lab} ({CLASS_NAMES[lab]}): {acc:.4f} ({cm_dual[lab, lab]}/{row})")

    save_confusion_matrix(cm_dual, OUT_DIR, "CWRU v2 dual mean (test)")
    save_confusion_matrix(cm_g, OUT_DIR / "gasf_only", "CWRU v2 GASF only (test)")
    save_confusion_matrix(cm_d, OUT_DIR / "gadf_only", "CWRU v2 GADF only (test)")

    gasf_val = read_best_val_accuracy("gasf")
    gadf_val = read_best_val_accuracy("gadf")
    export_table3(gasf_val, gadf_val, m_g, m_d, m_dual)

    print(f"\nResults saved to: {OUT_DIR}")
    print(f"Table3: {TABLE3_CSV}")


if __name__ == "__main__":
    main()
