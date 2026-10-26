"""
Compile chapter 3.4 extended-experiment tables from existing eval outputs.

Outputs -> D:\\TEP_2_repro\\paper_supplementary\\
"""
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

HUST_DUAL = 99.76
HUST_ATTN = 99.63


def cm_overall(cm_path: Path) -> tuple[float, list[float], int]:
    cm = np.loadtxt(cm_path, delimiter=",").astype(int)
    total = int(cm.sum())
    overall = float(np.trace(cm) / total * 100.0) if total else 0.0
    recalls = []
    for i in range(cm.shape[0]):
        row = int(cm[i, :].sum())
        recalls.append(float(cm[i, i] / row * 100.0) if row else 0.0)
    return overall, recalls, total


def read_table3(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            out[row["方法"]] = float(row["总体准确率(%)"])
    return out


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main() -> None:
    cwru_t3 = read_table3(RUNS / "cwru_v2_cls" / "table3_cwru_v2_dual_branch_performance.csv")
    cwru_dual, cwru_recalls, cwru_n = cm_overall(
        RUNS / "cwru_v2_cls" / "eval_dual_mean" / "confusion_matrix_counts.csv"
    )
    cwru_rows = [
        ["方法", "总体准确率(%)", "类0", "类1", "类2", "类3", "N_test", "备注"],
        [
            "v2 GASF-RGB 单分支",
            f"{cwru_t3['v2 GAF单分支GASF']:.2f}",
            "—", "—", "—", "—",
            str(cwru_n),
            "测试集",
        ],
        [
            "v2 GADF-RGB 单分支",
            f"{cwru_t3['v2 GAF单分支GADF']:.2f}",
            "—", "—", "—", "—",
            str(cwru_n),
            "测试集",
        ],
        [
            "v2 双分支简单平均",
            f"{cwru_dual:.2f}",
            *[f"{r:.2f}" for r in cwru_recalls],
            str(cwru_n),
            "6_edit/cwru_v2_cls",
        ],
    ]
    write_csv(ROOT / "table_341_cwru_v2_generalization.csv", cwru_rows[0], cwru_rows[1:])

    mcc5_dual, _, _ = cm_overall(RUNS / "mcc5_cls_b" / "eval_dual_mean" / "confusion_matrix_counts.csv")
    mcc5_attn, _, _ = cm_overall(RUNS / "mcc5_cls_b" / "eval_attn" / "confusion_matrix_counts_attn.csv")

    fusion_rows = [
        ["数据集/场景", "划分", "简单平均(%)", "注意力融合(%)", "差值(pp)", "说明"],
        ["HUST 稳态", "strict", f"{HUST_DUAL:.2f}", f"{HUST_ATTN:.2f}", f"{HUST_ATTN - HUST_DUAL:+.2f}", "3.1.2 表2"],
        ["MCC5 变工况", "whole_file", f"{mcc5_dual:.2f}", f"{mcc5_attn:.2f}", f"{mcc5_attn - mcc5_dual:+.2f}", "mcc5_cls_b"],
        ["CWRU", "strict v2", f"{cwru_dual:.2f}", "—", "—", "未训注意力分支"],
    ]
    write_csv(ROOT / "table_342_fusion_cross_scenario.csv", fusion_rows[0], fusion_rows[1:])

    md = ROOT / "ch3_4_extended_summary.md"
    md.write_text(
        "\n".join([
            "# 3.4 Extended experiments summary",
            "",
            f"## CWRU v2 dual mean: **{cwru_dual:.2f}%** (N={cwru_n})",
            f"- GASF: {cwru_t3['v2 GAF单分支GASF']:.2f}%",
            f"- GADF: {cwru_t3['v2 GAF单分支GADF']:.2f}%",
            "",
            "## Fusion: mean vs attention",
            f"- HUST: {HUST_DUAL:.2f}% vs {HUST_ATTN:.2f}%",
            f"- MCC5: {mcc5_dual:.2f}% vs {mcc5_attn:.2f}%",
        ]),
        encoding="utf-8",
    )
    print("Wrote", ROOT / "table_341_cwru_v2_generalization.csv")
    print("Wrote", ROOT / "table_342_fusion_cross_scenario.csv")
    print("Wrote", md)


if __name__ == "__main__":
    main()
