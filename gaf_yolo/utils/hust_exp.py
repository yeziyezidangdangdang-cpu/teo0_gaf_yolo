"""HUST 实验点：main=strict temporal；d=random 劣化对照。"""
from __future__ import annotations

from pathlib import Path

from utils.project_paths import (
    GAF_ROOT,
    YOLO_GADF_RGB_HUST_RANDOM,
    YOLO_GADF_RGB_HUST_STRICT,
    YOLO_GASF_RGB_HUST_RANDOM,
    YOLO_GASF_RGB_HUST_STRICT,
)

YOLO_EDIT_ROOT = GAF_ROOT.parent / "YOLOv8" / "runs" / "classify" / "6_edit"

EXP_POINTS: dict[str, dict[str, str]] = {
    "main": {
        "split_mode": "temporal_strict",
        "runs_name": "hust_gaf_rgb_cls",
        "dataset_tag": "strict",
    },
    "d": {
        "split_mode": "random",
        "runs_name": "hust_gaf_rgb_cls_d",
        "dataset_tag": "random",
    },
}


def yolo_dataset_dirs(point: str) -> tuple[Path, Path]:
    key = point.lower().strip()
    if key == "d":
        return YOLO_GASF_RGB_HUST_RANDOM, YOLO_GADF_RGB_HUST_RANDOM
    return YOLO_GASF_RGB_HUST_STRICT, YOLO_GADF_RGB_HUST_STRICT


def runs_dir(point: str) -> Path:
    key = point.lower().strip()
    if key not in EXP_POINTS:
        raise ValueError(f"exp_point 应为 main 或 d，收到: {point}")
    return YOLO_EDIT_ROOT / EXP_POINTS[key]["runs_name"]
