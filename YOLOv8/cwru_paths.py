"""CWRU v2 路径：数据集在 D:\\TEP_2，YOLO 输出在 repro runs/classify/6_edit。"""
from __future__ import annotations

from pathlib import Path

import repro_paths as pp

YOLO_ROOT = pp.YOLO_ROOT
EDIT_ROOT = YOLO_ROOT / "runs" / "classify" / "6_edit"
RUNS_DIR = EDIT_ROOT / "cwru_v2_cls"

GASF_DATA_ROOT = pp.YOLO_GASF_STRICT_V2
GADF_DATA_ROOT = pp.YOLO_GADF_STRICT_V2
GASF_DATA_ROOT_ENH = pp.YOLO_GASF_STRICT_ENH_V2
GADF_DATA_ROOT_ENH = pp.YOLO_GADF_STRICT_ENH_V2

CWRU_V2_IMGSZ = pp.CWRU_V2_IMGSZ
CWRU_V2_PROJECT = str(RUNS_DIR)

EVAL_DUAL_MEAN_DIR = RUNS_DIR / "eval_dual_mean"
EVAL_DUAL_MEAN_ENH_DIR = RUNS_DIR / "eval_dual_mean_enh"
TABLE3_CSV = RUNS_DIR / "table3_cwru_v2_dual_branch_performance.csv"
TABLE3_MD = RUNS_DIR / "table3_cwru_v2_dual_branch_performance.md"


def branch_weight_path(branch: str, run: int = 1) -> Path:
    suffix = "" if run == 1 else str(run)
    candidates = [
        RUNS_DIR / f"cwru_v2_{branch}{suffix}" / "weights" / "best.pt",
        YOLO_ROOT / "runs" / "classify" / "runs" / "classify" / "runs" / "cwru_v2_cls"
        / f"cwru_v2_{branch}{suffix}" / "weights" / "best.pt",
        YOLO_ROOT / "runs" / "classify" / "runs" / "cwru_v2_cls"
        / f"cwru_v2_{branch}{suffix}" / "weights" / "best.pt",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def branch_results_csv(branch: str) -> Path:
    p = RUNS_DIR / f"cwru_v2_{branch}" / "results.csv"
    return p
