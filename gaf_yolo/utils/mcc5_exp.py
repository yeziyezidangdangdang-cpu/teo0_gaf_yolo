"""MCC5 B/C 实验点配置（gaf_yolo 与 YOLOv8 共用）。"""
from __future__ import annotations

from pathlib import Path

from utils.project_paths import GAF_ROOT, YOLO_DS_MCC5

# B：whole_file；C：temporal；D：random 劣化对照
EXP_POINTS: dict[str, dict[str, str]] = {
    "c": {"split_mode": "temporal", "runs_name": "mcc5_cls_c", "yolo_suffix": ""},
    "b": {"split_mode": "whole_file", "runs_name": "mcc5_cls_b", "yolo_suffix": "_by_file"},
    "d": {"split_mode": "random", "runs_name": "mcc5_cls_random", "yolo_suffix": "_random"},
}

YOLO_EDIT_ROOT = GAF_ROOT.parent / "YOLOv8" / "runs" / "classify" / "6_edit"


def split_mode_for(point: str) -> str:
    key = point.lower().strip()
    if key not in EXP_POINTS:
        raise ValueError(f"exp_point 应为 b、c 或 d，收到: {point}")
    return EXP_POINTS[key]["split_mode"]


def yolo_dataset_dirs(point: str) -> tuple[Path, Path]:
    key = point.lower().strip()
    suffix = EXP_POINTS[key]["yolo_suffix"]
    gasf = YOLO_DS_MCC5 / f"mcc5_gasf_rgb_cls_strict{suffix}"
    gadf = YOLO_DS_MCC5 / f"mcc5_gadf_rgb_cls_strict{suffix}"
    return gasf, gadf


def runs_dir(point: str) -> Path:
    key = point.lower().strip()
    return YOLO_EDIT_ROOT / EXP_POINTS[key]["runs_name"]
