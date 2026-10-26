"""
Evaluate MCC5-THU dual-branch GASF-RGB + GADF-RGB with simple mean fusion.

  b — whole_file（论文 MCC5 主结果）
  c — temporal（对照）
  d — random shuffle（劣化对照）

Usage:
    cd D:\\TEP_2_repro\\YOLOv8
    python eval_mcc5_dual.py --exp-point d
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

from mcc5_paths import get_exp_point, set_exp_point, set_split_mode

LABELS = [0, 1, 2, 3]
CLASS_NAMES = {
    0: "Healthy",
    1: "Ball",
    2: "Inner race",
    3: "Outer race",
}

EXP_POINT_LABELS = {
    "b": "whole_file (B点)",
    "c": "temporal (C点)",
    "d": "random shuffle (D点)",
}


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(text, encoding="utf-8")
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}_latest{path.suffix}")
        alt.write_text(text, encoding="utf-8")
        print(
            f"[WARN] Cannot overwrite locked file:\n  {path}\n"
            f"  Close Excel or other programs using it. Wrote instead:\n  {alt}"
        )
        return alt


def _write_csv(path: Path, rows: list[list[str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerows(rows)
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}_latest{path.suffix}")
        with alt.open("w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerows(rows)
        print(
            f"[WARN] Cannot overwrite locked file:\n  {path}\n"
            f"  Close Excel or other programs using it. Wrote instead:\n  {alt}"
        )
        return alt


def load_models() -> tuple[YOLO, YOLO]:
    import mcc5_paths as mp

    gasf_pt = mp.branch_weight_path("gasf")
    gadf_pt = mp.branch_weight_path("gadf")
    if not gasf_pt.is_file():
        raise FileNotFoundError(
            f"GASF weights not found: {gasf_pt}\n"
            f"Run: python train_mcc5_dual.py --exp-point {mp.get_exp_point()}"
        )
    if not gadf_pt.is_file():
        raise FileNotFoundError(f"GADF weights not found: {gadf_pt}")
    return YOLO(str(gasf_pt)), YOLO(str(gadf_pt))


def read_best_val_accuracy(branch: str) -> float:
    import mcc5_paths as mp

    csv_path = mp.branch_results_csv(branch)
    if not csv_path.is_file():
        return float("nan")
    df = pd.read_csv(csv_path)
    col = "metrics/accuracy_top1"
    if col not in df.columns:
        return float("nan")
    return float(df[col].max()) * 100.0


def _evaluate_single_branch(model: YOLO, data_root: Path, imgsz: int) -> tuple[np.ndarray, int]:
    cm = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    total = 0
    test_root = data_root / "test"
    for lab in LABELS:
        img_dir = test_root / str(lab)
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.glob("*.png")):
            r = model(str(img), imgsz=imgsz, verbose=False)[0]
            pred = int(np.argmax(r.probs.data.cpu().numpy()))
            cm[lab, pred] += 1
            total += 1
    return cm, total


def evaluate_dual() -> tuple[np.ndarray, int]:
    import mcc5_paths as mp

    gasf_model, gadf_model = load_models()
    gasf_test_root = mp.GASF_DATA_ROOT / "test"
    gadf_test_root = mp.GADF_DATA_ROOT / "test"

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
            r1 = gasf_model(str(gasf_img), imgsz=mp.MCC5_IMGSZ, verbose=False)[0]
            r2 = gadf_model(str(gadf_img), imgsz=mp.MCC5_IMGSZ, verbose=False)[0]
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


def save_confusion_matrix(cm: np.ndarray, out_dir: Path, title: str) -> None:
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


def export_table3(gasf_val, gadf_val, gasf_test, gadf_test, dual_test, split_label: str) -> tuple[Path, Path]:
    import mcc5_paths as mp

    rows = [
        ["方法", "总体准确率(%)", "Healthy", "Ball", "Inner", "Outer", "数据来源"],
        [
            "MCC5 GASF-RGB 单分支",
            f"{gasf_test['overall']*100:.2f}",
            *[f"{r*100:.2f}" for r in gasf_test["recalls"]],
            "测试集",
        ],
        [
            "MCC5 GADF-RGB 单分支",
            f"{gadf_test['overall']*100:.2f}",
            *[f"{r*100:.2f}" for r in gadf_test["recalls"]],
            "测试集",
        ],
        [
            "MCC5 双分支简单平均",
            f"{dual_test['overall']*100:.2f}",
            *[f"{r*100:.2f}" for r in dual_test["recalls"]],
            "测试集",
        ],
        ["MCC5 GASF 验证集最佳", f"{gasf_val:.2f}", "—", "—", "—", "—", "验证集"],
        ["MCC5 GADF 验证集最佳", f"{gadf_val:.2f}", "—", "—", "—", "—", "验证集"],
    ]
    mp.TABLE3_CSV.parent.mkdir(parents=True, exist_ok=True)
    csv_path = _write_csv(mp.TABLE3_CSV, rows)
    md_path = _write_text(
        mp.TABLE3_MD,
        "\n".join([
            f"# MCC5-THU 双分支 GAF-RGB 性能（{split_label}）",
            "",
            f"- 划分：{split_label}",
            f"- 测试集双分支简单平均：{dual_test['overall']*100:.2f}%",
            f"- GASF 验证集最佳：{gasf_val:.2f}%",
            f"- GADF 验证集最佳：{gadf_val:.2f}%",
        ]),
    )
    return csv_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-point", choices=("c", "b", "d"), default="d", help="B=whole_file; C=temporal; D=random（默认 D）")
    parser.add_argument("--split-mode", choices=("temporal", "whole_file", "random"), default=None)
    args = parser.parse_args()

    if args.split_mode:
        set_split_mode(args.split_mode)
    else:
        set_exp_point(args.exp_point)

    import mcc5_paths as mp

    point = get_exp_point()
    split_label = EXP_POINT_LABELS.get(point, point)
    print(f"[eval_mcc5_dual] exp_point={point.upper()} split={split_label}")
    print(f"  weights dir: {mp.RUNS_DIR}")
    print(f"  GASF data:   {mp.GASF_DATA_ROOT}")
    print(f"  GADF data:   {mp.GADF_DATA_ROOT}")
    print(f"  eval out:    {mp.EVAL_DUAL_MEAN_DIR}")

    gasf_model, gadf_model = load_models()
    print(f"GASF: {mp.branch_weight_path('gasf')}")
    print(f"GADF: {mp.branch_weight_path('gadf')}")

    cm_g, n_g = _evaluate_single_branch(gasf_model, mp.GASF_DATA_ROOT, mp.MCC5_IMGSZ)
    m_g = metrics_from_cm(cm_g, n_g)
    print(f"GASF test overall: {m_g['overall']:.4f} (N={n_g})")

    cm_d, n_d = _evaluate_single_branch(gadf_model, mp.GADF_DATA_ROOT, mp.MCC5_IMGSZ)
    m_d = metrics_from_cm(cm_d, n_d)
    print(f"GADF test overall: {m_d['overall']:.4f} (N={n_d})")

    cm_dual, n_dual = evaluate_dual()
    m_dual = metrics_from_cm(cm_dual, n_dual)
    print(f"Dual mean test overall: {m_dual['overall']:.4f} (N={n_dual})")

    save_confusion_matrix(cm_dual, mp.EVAL_DUAL_MEAN_DIR, f"MCC5 dual mean ({split_label})")
    save_confusion_matrix(cm_g, mp.EVAL_DUAL_MEAN_DIR / "gasf_only", "MCC5 GASF only (test)")
    save_confusion_matrix(cm_d, mp.EVAL_DUAL_MEAN_DIR / "gadf_only", "MCC5 GADF only (test)")

    csv_path, md_path = export_table3(
        read_best_val_accuracy("gasf"),
        read_best_val_accuracy("gadf"),
        m_g,
        m_d,
        m_dual,
        split_label,
    )
    print(f"\nSaved: {mp.EVAL_DUAL_MEAN_DIR}")
    print(f"Table3: {csv_path}")
    print(f"Table3 md: {md_path}")


if __name__ == "__main__":
    main()
