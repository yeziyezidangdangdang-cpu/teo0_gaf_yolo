"""
MCC5 GAF-RGB 严格划分 YOLO 数据集（按 file_ranges 时间轴 70/15/15）。
"""
from __future__ import annotations

import csv
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.project_paths import (  # noqa: E402
    FILE_RANGES_MCC5,
    GADF_RGB_MCC5,
    GASF_RGB_MCC5,
    YOLO_GADF_RGB_MCC5_STRICT,
    YOLO_GASF_RGB_MCC5_STRICT,
    ensure_mcc5_dirs,
)

SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}
GASF_NAME_RE = re.compile(r"^gasf_rgb_(\d+)_(\d+)\.png$", re.IGNORECASE)
GADF_NAME_RE = re.compile(r"^gadf_rgb_(\d+)_(\d+)\.png$", re.IGNORECASE)


def load_file_ranges(csv_path: Path) -> List[Tuple[int, int, int]]:
    ranges: List[Tuple[int, int, int]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ranges.append((int(row["file_id"]), int(row["start_idx"]), int(row["end_idx"])))
    ranges.sort(key=lambda x: x[1])
    return ranges


def build_idx_to_split(ranges: List[Tuple[int, int, int]]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for _fid, s, e in ranges:
        length = e - s
        if length <= 0:
            continue
        if length < 3:
            for idx in range(s, e):
                mapping[idx] = "train"
            continue
        n_train = max(1, int(length * SPLIT_RATIOS["train"]))
        n_val = max(1, int(length * SPLIT_RATIOS["val"]))
        if n_train + n_val >= length:
            n_val = max(1, length - n_train - 1)
        n_test = length - n_train - n_val
        if n_test <= 0:
            n_test = 1
            if n_train > n_val:
                n_train -= 1
            else:
                n_val -= 1
        train_end = s + n_train
        val_end = train_end + n_val
        for idx in range(s, train_end):
            mapping[idx] = "train"
        for idx in range(train_end, val_end):
            mapping[idx] = "val"
        for idx in range(val_end, e):
            mapping[idx] = "test"
    return mapping


def extract_idx(name: str, is_gasf: bool) -> int:
    m = (GASF_NAME_RE if is_gasf else GADF_NAME_RE).match(name)
    if not m:
        raise ValueError(f"文件名不符合预期: {name}")
    return int(m.group(2))


def prepare_branch_strict(
    images_root: Path,
    out_root: Path,
    idx_to_split: Dict[int, str],
    is_gasf: bool,
) -> None:
    if not images_root.is_dir():
        raise FileNotFoundError(f"源图像目录不存在: {images_root}")
    if out_root.exists():
        shutil.rmtree(out_root)

    label_dirs = sorted([p for p in images_root.iterdir() if p.is_dir()])
    labels = [p.name for p in label_dirs]
    for split in SPLIT_RATIOS:
        for lab in labels:
            (out_root / split / lab).mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Dict[str, int]] = {}
    for lab_dir in label_dirs:
        lab = lab_dir.name
        for img in sorted(lab_dir.glob("*.png")):
            idx = extract_idx(img.name, is_gasf=is_gasf)
            split = idx_to_split.get(idx, "train")
            shutil.copy2(img, out_root / split / lab / img.name)
            summary.setdefault(lab, {"train": 0, "val": 0, "test": 0})[split] += 1

    print(f"\n[MCC5-STRICT] {images_root.name} -> {out_root}")
    for lab in labels:
        s = summary.get(lab, {"train": 0, "val": 0, "test": 0})
        print(f"  类别 {lab}: train={s['train']}, val={s['val']}, test={s['test']}")


def main() -> None:
    ensure_mcc5_dirs()
    ranges = load_file_ranges(FILE_RANGES_MCC5)
    idx_to_split = build_idx_to_split(ranges)
    print(f"[MCC5-STRICT] file_ranges: {FILE_RANGES_MCC5} ({len(ranges)} files)")

    prepare_branch_strict(GASF_RGB_MCC5, YOLO_GASF_RGB_MCC5_STRICT, idx_to_split, is_gasf=True)
    prepare_branch_strict(GADF_RGB_MCC5, YOLO_GADF_RGB_MCC5_STRICT, idx_to_split, is_gasf=False)

    print("\n[MCC5-STRICT] 完成")
    print(f"  GASF: {YOLO_GASF_RGB_MCC5_STRICT}")
    print(f"  GADF: {YOLO_GADF_RGB_MCC5_STRICT}")


if __name__ == "__main__":
    main()
