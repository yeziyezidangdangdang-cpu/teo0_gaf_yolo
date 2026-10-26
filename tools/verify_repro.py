"""验证 TEP_2_repro 能否在混合存储下复现论文关键结果。"""
from __future__ import annotations

import sys
from pathlib import Path

REPRO = Path(r"D:\TEP_2_repro")
sys.path.insert(0, str(REPRO / "YOLOv8"))
sys.path.insert(0, str(REPRO / "gaf_yolo"))

import repro_paths as pp  # noqa: E402
import hust_paths as hp  # noqa: E402
import mcc5_paths as mp  # noqa: E402
import cwru_paths as cp  # noqa: E402


def check(label: str, path: Path, min_files: int = 1) -> bool:
    ok = path.exists()
    if path.is_dir():
        n = sum(1 for _ in path.rglob("*") if _.is_file())
        ok = ok and n >= min_files
        print(f"{'OK' if ok else 'FAIL'} {label}: {path} ({n} files)")
    else:
        print(f"{'OK' if ok else 'FAIL'} {label}: {path}")
    return ok


def main() -> int:
    ok = True
    ok &= check("raw HUST", pp.HUST_RAW_DIR, 10)
    ok &= check("raw MCC5", pp.MCC5_RAW_DIR, 10)
    ok &= check("raw CWRU csv", pp.CWRU_CSV_DIR, 5)
    ok &= check("archive processed hust csv", pp.FINAL_DATASET_HUST)
    ok &= check("archive hust axes npy", pp.GAF_INPUT_HUST_AXES)
    ok &= check("archive gaf_rgb", pp.GAF_RGB_HUST, 1000)
    ok &= check("archive yolo hust strict gasf", pp.YOLO_GASF_RGB_HUST_STRICT, 100)
    ok &= check("archive yolo mcc5 by_file gasf", mp.GASF_DATA_ROOT, 100)
    ok &= check("archive yolo cwru v2 gasf", cp.GASF_DATA_ROOT, 100)
    ok &= check("repro weight hust gasf", hp.branch_weight_path("gasf_rgb"))
    ok &= check("repro weight mcc5 gasf", mp.branch_weight_path("gasf"))
    ok &= check("repro weight cwru v2 gasf", cp.branch_weight_path("gasf"))
    ok &= check("repro eval mcc5 cm", mp.EVAL_DUAL_MEAN_DIR / "confusion_matrix_counts.csv")
    ok &= check("repro eval cwru v2 cm", cp.EVAL_DUAL_MEAN_DIR / "confusion_matrix_counts.csv")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
