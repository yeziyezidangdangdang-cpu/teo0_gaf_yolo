"""
Evaluate HUST GAF dual-branch (GASF-RGB + GADF-RGB) with simple mean fusion.

  main — strict temporal test
  d    — random split test（劣化对照）

Usage:
    cd D:\\TEP_2_repro\\YOLOv8
    python eval_hust_gaf_dual.py --exp-point d
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from ultralytics import YOLO

from hust_paths import (
    EVAL_DUAL_MEAN_DIR,
    GADF_DATA_ROOT,
    GASF_DATA_ROOT,
    HUST_IMGSZ,
    TABLE3_CSV,
    TABLE3_MD,
    branch_results_csv,
    branch_weight_path,
    get_exp_point,
    set_exp_point,
)

_fonts = [f.name for f in fm.fontManager.ttflist]
FP_EN = fm.FontProperties(family="Times New Roman", size=8) if "Times New Roman" in _fonts else None
plt.rcParams["font.size"] = 8.0
plt.rcParams["axes.unicode_minus"] = False

LABELS = [0, 1, 2, 3]
CLASS_NAMES = {0: "Healthy", 1: "Ball", 2: "Inner race", 3: "Outer race"}


def load_models() -> tuple[YOLO, YOLO]:
    import hust_paths as hp

    gasf_pt, gadf_pt = hp.branch_weight_path("gasf_rgb"), hp.branch_weight_path("gadf_rgb")
    if not gasf_pt.is_file():
        raise FileNotFoundError(f"GASF weights not found: {gasf_pt}\nRun train_hust_gaf_rgb_dual.py first")
    if not gadf_pt.is_file():
        raise FileNotFoundError(f"GADF weights not found: {gadf_pt}")
    return YOLO(str(gasf_pt)), YOLO(str(gadf_pt))


def read_best_val_accuracy(branch: str) -> float:
    import hust_paths as hp

    csv_path = hp.branch_results_csv(branch)
    if not csv_path.is_file():
        return float("nan")
    df = pd.read_csv(csv_path)
    col = "metrics/accuracy_top1"
    if col not in df.columns:
        return float("nan")
    return float(df[col].max()) * 100.0


def metrics_from_cm(cm: np.ndarray, total: int) -> dict:
    overall = cm.trace() / total if total > 0 else 0.0
    recalls = []
    for i in range(len(LABELS)):
        row = cm[i, :].sum()
        recalls.append(cm[i, i] / row if row > 0 else 0.0)
    return {"overall": overall, "recalls": recalls}


def evaluate_dual() -> tuple[np.ndarray, int]:
    import hust_paths as hp

    gasf_model, gadf_model = load_models()
    gasf_test = hp.GASF_DATA_ROOT / "test"
    gadf_test = hp.GADF_DATA_ROOT / "test"
    cm = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    total = 0

    for lab in LABELS:
        gd = gasf_test / str(lab)
        dd = gadf_test / str(lab)
        if not gd.is_dir() or not dd.is_dir():
            continue
        gadf_map = {p.name: p for p in dd.glob("*.png")}
        for gasf_img in sorted(gd.glob("*.png")):
            gadf_name = gasf_img.name.replace("gasf_rgb_", "gadf_rgb_", 1)
            gadf_img = gadf_map.get(gadf_name)
            if gadf_img is None:
                continue
            r1 = gasf_model(str(gasf_img), imgsz=hp.HUST_IMGSZ, verbose=False)[0]
            r2 = gadf_model(str(gadf_img), imgsz=hp.HUST_IMGSZ, verbose=False)[0]
            probs = (r1.probs.data.cpu().numpy() + r2.probs.data.cpu().numpy()) / 2.0
            cm[lab, int(np.argmax(probs))] += 1
            total += 1
    return cm, total


def save_confusion_matrix(cm: np.ndarray, out_dir: Path, title: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / "confusion_matrix_counts.csv", cm, fmt="%d", delimiter=",")
    row_sums = cm.sum(axis=1, keepdims=True)
    safe = np.where(row_sums == 0, 1, row_sums)
    cm_norm = cm.astype(float) / safe
    np.savetxt(out_dir / "confusion_matrix_normalized.csv", cm_norm, fmt="%.4f", delimiter=",")

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(np.nan_to_num(cm_norm), cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels([f"{i}\n{CLASS_NAMES[i]}" for i in LABELS])
    ax.set_yticklabels([f"{i}:{CLASS_NAMES[i]}" for i in LABELS])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)


def export_table3(gasf_val, gadf_val, gasf_test, gadf_test, dual_test, split_label: str) -> None:
    import hust_paths as hp

    rows = [
        ["方法", "总体准确率(%)", "Healthy", "Ball", "Inner", "Outer", "数据来源"],
        [
            "HUST GASF-RGB 单分支",
            f"{gasf_test['overall']*100:.2f}",
            *[f"{r*100:.2f}" for r in gasf_test["recalls"]],
            "测试集",
        ],
        [
            "HUST GADF-RGB 单分支",
            f"{gadf_test['overall']*100:.2f}",
            *[f"{r*100:.2f}" for r in gadf_test["recalls"]],
            "测试集",
        ],
        [
            "HUST 双分支简单平均",
            f"{dual_test['overall']*100:.2f}",
            *[f"{r*100:.2f}" for r in dual_test["recalls"]],
            "测试集",
        ],
        ["HUST GASF 验证集最佳", f"{gasf_val:.2f}", "—", "—", "—", "—", "验证集"],
        ["HUST GADF 验证集最佳", f"{gadf_val:.2f}", "—", "—", "—", "—", "验证集"],
    ]
    hp.TABLE3_CSV.parent.mkdir(parents=True, exist_ok=True)
    with hp.TABLE3_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(rows)
    hp.TABLE3_MD.write_text(
        "\n".join([
            f"# HUST 双分支 GAF-RGB 性能（{split_label}）",
            "",
            f"- 划分：{split_label}",
            f"- 测试集双分支简单平均：{dual_test['overall']*100:.2f}%",
            f"- GASF 验证集最佳：{gasf_val:.2f}%",
            f"- GADF 验证集最佳：{gadf_val:.2f}%",
        ]),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-point", choices=("main", "d"), default="main")
    args = parser.parse_args()
    set_exp_point(args.exp_point)
    import hust_paths as hp

    split_label = "strict temporal" if get_exp_point() == "main" else "random shuffle (D点)"
    gasf_model, gadf_model = load_models()
    print(f"[eval_hust_gaf_dual] exp_point={get_exp_point().upper()} split={split_label}")

    def eval_branch(model, root):
        cm = np.zeros((len(LABELS), len(LABELS)), dtype=int)
        n = 0
        for lab in LABELS:
            d = root / "test" / str(lab)
            if not d.is_dir():
                continue
            for img in sorted(d.glob("*.png")):
                pred = int(np.argmax(model(str(img), imgsz=hp.HUST_IMGSZ, verbose=False)[0].probs.data.cpu().numpy()))
                cm[lab, pred] += 1
                n += 1
        return cm, n

    cm_g, n_g = eval_branch(gasf_model, hp.GASF_DATA_ROOT)
    cm_d, n_d = eval_branch(gadf_model, hp.GADF_DATA_ROOT)
    cm_dual, n_dual = evaluate_dual()

    m_g = metrics_from_cm(cm_g, n_g)
    m_d = metrics_from_cm(cm_d, n_d)
    m_dual = metrics_from_cm(cm_dual, n_dual)

    print(f"GASF test: {m_g['overall']:.4f} (N={n_g})")
    print(f"GADF test: {m_d['overall']:.4f} (N={n_d})")
    print(f"Dual mean test: {m_dual['overall']:.4f} (N={n_dual})")

    save_confusion_matrix(cm_dual, hp.EVAL_DUAL_MEAN_DIR, f"HUST dual mean ({split_label})")
    export_table3(
        read_best_val_accuracy("gasf_rgb"),
        read_best_val_accuracy("gadf_rgb"),
        m_g,
        m_d,
        m_dual,
        split_label,
    )
    print(f"Saved: {hp.EVAL_DUAL_MEAN_DIR}")
    print(f"Table3: {hp.TABLE3_CSV}")


if __name__ == "__main__":
    main()
