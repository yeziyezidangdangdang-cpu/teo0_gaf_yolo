"""
MCC5 实验点路径：B / C / D。

YOLO 数据集在 D:\\TEP_2\\gaf_yolo\\yolo_datasets_mcc5；权重/结果在 repro。
"""
from __future__ import annotations

from pathlib import Path

import repro_paths as pp

YOLO_ROOT = pp.YOLO_ROOT
YOLO_DS = pp.YOLO_DS_MCC5
EDIT_ROOT = YOLO_ROOT / "runs" / "classify" / "6_edit"

MCC5_IMGSZ = pp.MCC5_YOLO_IMGSZ

EXP_POINTS: dict[str, tuple[str, str, str]] = {
    "c": ("temporal", "mcc5_cls_c", ""),
    "b": ("whole_file", "mcc5_cls_b", "_by_file"),
    "d": ("random", "mcc5_cls_random", "_random"),
}

_CURRENT_POINT = "b"


def _yolo_roots(suffix: str) -> tuple[Path, Path]:
    return (
        YOLO_DS / f"mcc5_gasf_rgb_cls_strict{suffix}",
        YOLO_DS / f"mcc5_gadf_rgb_cls_strict{suffix}",
    )


def set_exp_point(point: str) -> str:
    global _CURRENT_POINT
    global RUNS_DIR, GASF_DATA_ROOT, GADF_DATA_ROOT, MCC5_PROJECT
    global EVAL_DUAL_MEAN_DIR, EVAL_DUAL_ATTN_DIR, ATTN_DIR, ATTN_CKPT
    global TABLE3_CSV, TABLE3_MD

    key = point.lower().strip()
    if key not in EXP_POINTS:
        raise ValueError(f"exp_point 应为 b、c 或 d，收到: {point}")

    split_mode, runs_name, suffix = EXP_POINTS[key]
    _CURRENT_POINT = key
    RUNS_DIR = EDIT_ROOT / runs_name
    GASF_DATA_ROOT, GADF_DATA_ROOT = _yolo_roots(suffix)
    MCC5_PROJECT = str(RUNS_DIR)
    EVAL_DUAL_MEAN_DIR = RUNS_DIR / "eval_dual_mean"
    EVAL_DUAL_ATTN_DIR = RUNS_DIR / "eval_attn"
    ATTN_DIR = RUNS_DIR / "branch_attn"
    ATTN_CKPT = ATTN_DIR / "branch_attn.pt"
    TABLE3_CSV = RUNS_DIR / "table3_mcc5_dual_branch_performance.csv"
    TABLE3_MD = RUNS_DIR / "table3_mcc5_dual_branch_performance.md"
    return split_mode


def get_exp_point() -> str:
    return _CURRENT_POINT


RUNS_DIR = EDIT_ROOT / "mcc5_cls_b"
GASF_DATA_ROOT, GADF_DATA_ROOT = _yolo_roots("_by_file")
MCC5_PROJECT = str(RUNS_DIR)
EVAL_DUAL_MEAN_DIR = RUNS_DIR / "eval_dual_mean"
EVAL_DUAL_ATTN_DIR = RUNS_DIR / "eval_attn"
ATTN_DIR = RUNS_DIR / "branch_attn"
ATTN_CKPT = ATTN_DIR / "branch_attn.pt"
TABLE3_CSV = RUNS_DIR / "table3_mcc5_dual_branch_performance.csv"
TABLE3_MD = RUNS_DIR / "table3_mcc5_dual_branch_performance.md"


def branch_weight_path(branch: str) -> Path:
    return RUNS_DIR / f"mcc5_{branch}" / "weights" / "best.pt"


def branch_results_csv(branch: str) -> Path:
    return RUNS_DIR / f"mcc5_{branch}" / "results.csv"


def set_split_mode(mode: str) -> None:
    for key, (sm, _, _) in EXP_POINTS.items():
        if sm == mode:
            set_exp_point(key)
            return
    raise ValueError(f"unknown split mode: {mode} (use temporal|whole_file|random)")
