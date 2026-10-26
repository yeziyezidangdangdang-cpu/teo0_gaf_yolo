"""Compile chapter 3.1.3 / 3.2 split-comparison tables."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT.parent
sys.path.insert(0, str(REPRO / "YOLOv8"))
import repro_paths as pp  # noqa: E402

RUNS = pp.YOLO_ROOT / "runs" / "classify" / "6_edit"


def cm_overall(cm_path: Path) -> tuple[float, int]:
    cm = np.loadtxt(cm_path, delimiter=",").astype(int)
    total = int(cm.sum())
    return float(np.trace(cm) / total * 100.0) if total else 0.0, total


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main() -> None:
    hust_d_dual, hust_d_n = cm_overall(
        RUNS / "hust_gaf_rgb_cls_d" / "eval_dual_mean" / "confusion_matrix_counts.csv"
    )
    mcc5_b, n_b = cm_overall(RUNS / "mcc5_cls_b" / "eval_dual_mean" / "confusion_matrix_counts.csv")
    mcc5_c, n_c = cm_overall(RUNS / "mcc5_cls_c" / "eval_dual_mean" / "confusion_matrix_counts.csv")
    mcc5_d, n_d = cm_overall(RUNS / "mcc5_cls_random" / "eval_dual_mean" / "confusion_matrix_counts.csv")

    write_csv(
        ROOT / "table_313_hust_split_compare.csv",
        ["划分方式", "GASF(%)", "GADF(%)", "双分支简单平均(%)", "N_test", "备注"],
        [
            ["strict temporal (main)", "98.00", "99.60", "99.40", "—", "3.1.2 ablation Table 2"],
            ["random shuffle (seed=42)", "99.44", "99.67", f"{hust_d_dual:.2f}", str(hust_d_n), "hust_gaf_rgb_cls_d"],
        ],
    )
    write_csv(
        ROOT / "table_32_mcc5_split_compare.csv",
        ["划分方式", "GASF(%)", "GADF(%)", "双分支简单平均(%)", "N_test", "目录"],
        [
            ["whole_file (main, B)", "99.78", "98.78", f"{mcc5_b:.2f}", str(n_b), "mcc5_cls_b"],
            ["temporal (C)", "86.42", "82.68", f"{mcc5_c:.2f}", str(n_c), "mcc5_cls_c"],
            ["random shuffle (D)", "97.78", "96.33", f"{mcc5_d:.2f}", str(n_d), "mcc5_cls_random"],
        ],
    )
    print("Wrote split tables under", ROOT)


if __name__ == "__main__":
    main()
