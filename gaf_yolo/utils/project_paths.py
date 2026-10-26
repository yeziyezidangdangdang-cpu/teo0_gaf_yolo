"""
TEP_2 混合存储路径（论文复现）。

  D:\\TEP_2_repro  — 代码、原始数据 raw、训练权重/评估结果、论文作图
  D:\\TEP_2        — 大体积中间数据（processed NPY、GAF 图、YOLO 数据集、STFT 图等）

所有 gaf_yolo / YOLOv8 脚本应 import 本模块，勿硬编码绝对路径。
"""
from __future__ import annotations

from pathlib import Path

REPRO_ROOT = Path(r"D:\TEP_2_repro")
ARCHIVE_ROOT = Path(r"D:\TEP_2")

REPRO_GAF = REPRO_ROOT / "gaf_yolo"
ARCHIVE_GAF = ARCHIVE_ROOT / "gaf_yolo"

YOLO_ROOT = REPRO_ROOT / "YOLOv8"
PAPER_DRAW_ROOT = REPRO_ROOT / "paper_draw"
PAPER_SUPP_ROOT = REPRO_ROOT / "paper_supplementary"

# ── 原始数据：仅 repro ─────────────────────────────────────────
DATA_RAW = REPRO_GAF / "data" / "raw"
RAW = DATA_RAW

HUST_RAW_DIR = RAW / "HUST-bearing-dataset" / "Raw data (原始数据)"
MCC5_RAW_DIR = RAW / "MCC5-THU Motor"
CWRU_CSV_DIR = RAW / "CWRU" / "csv"
CWRU_MAT_DIR = RAW / "CWRU" / "cwrn"

# ── 大中间数据：仅 archive（TEP_2）────────────────────────────
DATA = ARCHIVE_GAF / "data"
PROCESSED = DATA / "processed"

FINAL_DATASET_HUST = PROCESSED / "final_dataset_hust.csv"
FILE_RANGES_HUST = PROCESSED / "file_ranges_hust.csv"
GAF_INPUT_HUST_AXES = PROCESSED / "gaf_input_hust_axes.npy"
GAF_INPUT_HUST_AXES_ENV = PROCESSED / "gaf_input_hust_axes_env.npy"
STFT_INPUT_HUST = PROCESSED / "stft_input_hust.npy"
GAF_PREPROCESSED_HUST = PROCESSED / "gaf_preprocessed_hust.npy"

GAF_RGB_HUST = DATA / "gaf_rgb"
GASF_RGB_HUST = GAF_RGB_HUST / "gasf"
GADF_RGB_HUST = GAF_RGB_HUST / "gadf"
GAF_RGB_ENV_HUST = DATA / "gaf_rgb_env"

IMAGES = DATA / "images"
IMAGES_ENHANCED = DATA / "images_enhanced"
SPECTROGRAMS = DATA / "spectrograms"
SPECTROGRAMS_CWRU = DATA / "spectrograms_cwru"
SPECTROGRAMS_MAG_RESID = DATA / "spectrograms_mag_resid"
SPEC_X = DATA / "spec_x"
SPEC_Y = DATA / "spec_y"
SPEC_Z = DATA / "spec_z"
DIAGNOSTICS = DATA / "diagnostics"

YOLO_DS = ARCHIVE_GAF / "yolo_datasets"
YOLO_GASF_RGB_HUST_STRICT = YOLO_DS / "hust_gasf_rgb_cls_strict"
YOLO_GADF_RGB_HUST_STRICT = YOLO_DS / "hust_gadf_rgb_cls_strict"
YOLO_GASF_RGB_HUST_RANDOM = YOLO_DS / "hust_gasf_rgb_cls_random"
YOLO_GADF_RGB_HUST_RANDOM = YOLO_DS / "hust_gadf_rgb_cls_random"
YOLO_SPEC_STRICT = YOLO_DS / "spec_cls_strict"

# CWRU 旧管线（archive）
MERGED_SEGMENTS = PROCESSED / "merged_segments.csv"
FINAL_DATASET_CWRU = PROCESSED / "final_dataset.csv"
GAF_PREPROCESSED_CWRU = PROCESSED / "gaf_preprocessed_cwru.npy"

# CWRU v2 修正管线（2048/512 + 去均值分位数，中间 CSV/NPY 也在 archive）
PROCESSED_V2 = DATA / "processed_v2"
IMAGES_V2 = DATA / "images_v2"
IMAGES_ENH_V2 = DATA / "images_enhanced_v2"
YOLO_DS_V2 = ARCHIVE_GAF / "yolo_datasets_v2"
YOLO_DS_V2_ENH = ARCHIVE_GAF / "yolo_datasets_v2_enhanced"

MERGED_SEGMENTS_V2 = PROCESSED_V2 / "merged_segments.csv"
FILE_RANGES_CWRU_V2 = PROCESSED_V2 / "file_ranges_cwru.csv"
FINAL_DATASET_CWRU_V2 = PROCESSED_V2 / "final_dataset.csv"
GAF_PREPROCESSED_CWRU_V2 = PROCESSED_V2 / "gaf_preprocessed_cwru.npy"

GASF_IMAGES_V2 = IMAGES_V2 / "gasf"
GADF_IMAGES_V2 = IMAGES_V2 / "gadf"
GASF_IMAGES_ENH_V2 = IMAGES_ENH_V2 / "gasf"
GADF_IMAGES_ENH_V2 = IMAGES_ENH_V2 / "gadf"

YOLO_GASF_STRICT_V2 = YOLO_DS_V2 / "gasf_cls_strict"
YOLO_GADF_STRICT_V2 = YOLO_DS_V2 / "gadf_cls_strict"
YOLO_GASF_STRICT_ENH_V2 = YOLO_DS_V2_ENH / "gasf_cls_strict"
YOLO_GADF_STRICT_ENH_V2 = YOLO_DS_V2_ENH / "gadf_cls_strict"

# MCC5-THU（中间数据在 archive；raw 在 repro）
PROCESSED_MCC5 = DATA / "processed_mcc5"
GAF_RGB_MCC5 = DATA / "gaf_rgb_mcc5"
YOLO_DS_MCC5 = ARCHIVE_GAF / "yolo_datasets_mcc5"

FINAL_DATASET_MCC5 = PROCESSED_MCC5 / "final_dataset_mcc5.csv"
FILE_RANGES_MCC5 = PROCESSED_MCC5 / "file_ranges_mcc5.csv"
GAF_INPUT_MCC5_AXES = PROCESSED_MCC5 / "gaf_input_mcc5_axes.npy"
MCC5_META_JSON = PROCESSED_MCC5 / "mcc5_prepare_meta.json"

GASF_RGB_MCC5 = GAF_RGB_MCC5 / "gasf"
GADF_RGB_MCC5 = GAF_RGB_MCC5 / "gadf"
YOLO_GASF_RGB_MCC5_STRICT = YOLO_DS_MCC5 / "mcc5_gasf_rgb_cls_strict"
YOLO_GADF_RGB_MCC5_STRICT = YOLO_DS_MCC5 / "mcc5_gadf_rgb_cls_strict"
YOLO_GASF_RGB_MCC5_RANDOM = YOLO_DS_MCC5 / "mcc5_gasf_rgb_cls_strict_random"
YOLO_GADF_RGB_MCC5_RANDOM = YOLO_DS_MCC5 / "mcc5_gadf_rgb_cls_strict_random"

# ── 信号 / 图像参数 ───────────────────────────────────────────
HUST_GAF_SIZE = 256
HUST_YOLO_IMGSZ = 128

MCC5_FS = 12800
MCC5_SEGMENT_LENGTH = 2048
MCC5_STRIDE = 512
MCC5_GAF_SIZE = 256
MCC5_YOLO_IMGSZ = 128

CWRU_FS = 12000
CWRU_SEGMENT_LENGTH = 2048
CWRU_STRIDE = 512
CWRU_GAF_SIZE = 256
ROBUST_PLOW = 1.0
ROBUST_PHIGH = 99.0
MCC5_ROBUST_PLOW = ROBUST_PLOW
MCC5_ROBUST_PHIGH = ROBUST_PHIGH

CWRU_V2_IMGSZ = 256
CWRU_V2_RUNS_PROJECT = "runs/classify/6_edit/cwru_v2_cls"


def archive_data_path(*parts: str) -> Path:
    """archive 下 gaf_yolo/data/…"""
    return DATA.joinpath(*parts)


def repro_data_path(*parts: str) -> Path:
    """repro 下 gaf_yolo/data/…（仅 raw 等小文件）"""
    return REPRO_GAF.joinpath("data", *parts)


def ensure_archive_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def ensure_v2_dirs() -> None:
    ensure_archive_dirs(
        PROCESSED_V2,
        IMAGES_V2,
        IMAGES_ENH_V2,
        GASF_IMAGES_V2,
        GADF_IMAGES_V2,
        YOLO_DS_V2,
        YOLO_DS_V2_ENH,
    )


def ensure_hust_dirs() -> None:
    ensure_archive_dirs(
        PROCESSED,
        GAF_RGB_HUST,
        GASF_RGB_HUST,
        GADF_RGB_HUST,
        YOLO_DS,
    )


def ensure_mcc5_dirs() -> None:
    ensure_archive_dirs(
        PROCESSED_MCC5,
        GAF_RGB_MCC5,
        GASF_RGB_MCC5,
        GADF_RGB_MCC5,
        YOLO_DS_MCC5,
    )
