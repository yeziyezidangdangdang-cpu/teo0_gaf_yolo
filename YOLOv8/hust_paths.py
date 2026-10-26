"""
HUST GAF-RGB 实验点路径。

  main — strict temporal（2.2，论文主线）
  d    — random shuffle（2.1，劣化对照）

YOLO 数据集在 D:\\TEP_2\\gaf_yolo\\yolo_datasets；权重/结果在 repro。
"""
from __future__ import annotations

from pathlib import Path

import repro_paths as pp

YOLO_ROOT = pp.YOLO_ROOT
YOLO_DS = pp.YOLO_DS
EDIT_ROOT = YOLO_ROOT / "runs" / "classify" / "6_edit"
LEGACY_RUNS = YOLO_ROOT / "runs" / "classify" / "hust_gaf_rgb_cls"

HUST_IMGSZ = pp.HUST_YOLO_IMGSZ

EXP_POINTS: dict[str, tuple[str, str, str]] = {
    "main": ("temporal_strict", "hust_gaf_rgb_cls", "strict"),
    "d": ("random", "hust_gaf_rgb_cls_d", "random"),
}

_CURRENT_POINT = "main"


def _yolo_roots(tag: str) -> tuple[Path, Path]:
    if tag == "random":
        return pp.YOLO_GASF_RGB_HUST_RANDOM, pp.YOLO_GADF_RGB_HUST_RANDOM
    return pp.YOLO_GASF_RGB_HUST_STRICT, pp.YOLO_GADF_RGB_HUST_STRICT


def set_exp_point(point: str) -> str:
    global _CURRENT_POINT
    global RUNS_DIR, GASF_DATA_ROOT, GADF_DATA_ROOT, HUST_PROJECT
    global EVAL_DUAL_MEAN_DIR, TABLE3_CSV, TABLE3_MD

    key = point.lower().strip()
    if key not in EXP_POINTS:
        raise ValueError(f"exp_point 应为 main 或 d，收到: {point}")

    split_mode, runs_name, tag = EXP_POINTS[key]
    _CURRENT_POINT = key
    RUNS_DIR = EDIT_ROOT / runs_name if key == "d" else LEGACY_RUNS
    GASF_DATA_ROOT, GADF_DATA_ROOT = _yolo_roots(tag)
    HUST_PROJECT = str(RUNS_DIR.resolve())
    EVAL_DUAL_MEAN_DIR = (EDIT_ROOT / runs_name / "eval_dual_mean") if key == "d" else (
        YOLO_ROOT / "runs" / "hust_gaf_cls" / "eval_dual_mean"
    )
    TABLE3_CSV = (EDIT_ROOT / runs_name / "table3_hust_dual_branch_performance.csv") if key == "d" else (
        EDIT_ROOT / "hust_gaf_rgb_cls" / "table3_hust_dual_branch_performance.csv"
    )
    TABLE3_MD = TABLE3_CSV.with_suffix(".md")
    return split_mode


def get_exp_point() -> str:
    return _CURRENT_POINT


RUNS_DIR = LEGACY_RUNS
GASF_DATA_ROOT, GADF_DATA_ROOT = _yolo_roots("strict")
HUST_PROJECT = str(LEGACY_RUNS.resolve())
EVAL_DUAL_MEAN_DIR = YOLO_ROOT / "runs" / "hust_gaf_cls" / "eval_dual_mean"
TABLE3_CSV = EDIT_ROOT / "hust_gaf_rgb_cls" / "table3_hust_dual_branch_performance.csv"
TABLE3_MD = TABLE3_CSV.with_suffix(".md")


def branch_weight_path(branch: str) -> Path:
    if branch in ("gasf", "gasf_rgb"):
        return RUNS_DIR / "hust_gasf_rgb" / "weights" / "best.pt"
    return RUNS_DIR / "hust_gadf_rgb" / "weights" / "best.pt"


def branch_results_csv(branch: str) -> Path:
    if branch in ("gasf", "gasf_rgb"):
        return RUNS_DIR / "hust_gasf_rgb" / "results.csv"
    return RUNS_DIR / "hust_gadf_rgb" / "results.csv"
