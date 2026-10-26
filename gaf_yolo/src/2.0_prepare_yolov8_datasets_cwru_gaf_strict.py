"""
基于 CWRU GAF 图像（GASF/GADF）构建「按文件严格划分」的 YOLOv8 分类数据集，与 HUST GAF 策略一致。

目的：
- 避免同一原始 CSV 文件切出来的高度重叠片段同时出现在 train/val/test 中，降低数据泄漏。
- 使用 file_ranges_cwru.csv 在每个文件内按 70%/15%/15% 划为 train/val/test。
- 每类最多 1500 张（1050 train / 225 val / 225 test）。

输入：
- GASF：project_root / data / images / gasf / <label> / gasf_<label>_<idx>.png
- GADF：project_root / data / images / gadf / <label> / gadf_<label>_<idx>.png
- 划分索引：project_root / data / processed / file_ranges_cwru.csv

输出：
- yolo_datasets/gasf_cls_strict, yolo_datasets/gadf_cls_strict
- 若存在 SA-GAF 图像：yolo_datasets_enhanced/gasf_cls_strict, gadf_cls_strict
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
import re
from collections import defaultdict
from typing import Dict, List, Tuple


SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}
MAX_PER_LABEL = 1500
MAX_TRAIN_PER_LABEL = int(MAX_PER_LABEL * SPLIT_RATIOS["train"])
MAX_VAL_PER_LABEL = int(MAX_PER_LABEL * SPLIT_RATIOS["val"])
MAX_TEST_PER_LABEL = MAX_PER_LABEL - MAX_TRAIN_PER_LABEL - MAX_VAL_PER_LABEL

GASF_NAME_RE = re.compile(r"^gasf_(\d+)_(\d+)\.png$", re.IGNORECASE)
GADF_NAME_RE = re.compile(r"^gadf_(\d+)_(\d+)\.png$", re.IGNORECASE)


def load_file_ranges(csv_path: Path) -> List[Tuple[int, int, int]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"file_ranges_cwru.csv 不存在: {csv_path}")
    ranges: List[Tuple[int, int, int]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = int(row["file_id"])
            s = int(row["start_idx"])
            e = int(row["end_idx"])
            ranges.append((fid, s, e))
    ranges.sort(key=lambda x: x[1])
    return ranges


def build_idx_to_split(ranges: List[Tuple[int, int, int]]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for fid, s, e in ranges:
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
        raise ValueError(f"文件名不符合 gasf_<label>_<idx>.png 格式: {name}")
    return int(m.group(2))


def build_idx_to_file(ranges: List[Tuple[int, int, int]]) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for _fid, start, end in ranges:
        for idx in range(start, end):
            mapping[idx] = _fid
    return mapping


def prepare_branch_strict(
    images_root: Path,
    out_root: Path,
    idx_to_split: Dict[int, str],
    idx_to_file: Dict[int, int],
    is_gasf: bool,
) -> None:
    if not images_root.exists():
        raise FileNotFoundError(f"源图像目录不存在: {images_root}")
    if out_root.exists():
        shutil.rmtree(out_root)

    label_dirs = sorted([p for p in images_root.iterdir() if p.is_dir()])
    labels = [p.name for p in label_dirs]
    for split in SPLIT_RATIOS.keys():
        for lab in labels:
            (out_root / split / lab).mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Dict[str, int]] = {}
    cap_counts: Dict[str, Dict[str, int]] = {lab: {"train": 0, "val": 0, "test": 0} for lab in labels}
    per_file_cap: Dict[str, Dict[str, Dict[int, int]]] = {
        lab: {"train": {}, "val": {}, "test": {}} for lab in labels
    }
    per_file_counts: Dict[str, Dict[str, Dict[int, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )

    for lab_dir in label_dirs:
        lab = lab_dir.name
        imgs = sorted([p for p in lab_dir.glob("*.png") if p.is_file()])
        if not imgs:
            print(f"[CWRU-STRICT] 标签 {lab} 在 {lab_dir} 下未找到 PNG 图像")
            continue
        summary.setdefault(lab, {"train": 0, "val": 0, "test": 0})
        file_ids = {
            idx_to_file.get(extract_idx(img.name, is_gasf=is_gasf))
            for img in imgs
        }
        file_ids.discard(None)
        n_files = max(len(file_ids), 1)
        per_file_cap[lab] = {
            "train": MAX_TRAIN_PER_LABEL // n_files,
            "val": MAX_VAL_PER_LABEL // n_files,
            "test": MAX_TEST_PER_LABEL // n_files,
        }
        for img in imgs:
            idx = extract_idx(img.name, is_gasf=is_gasf)
            split = idx_to_split.get(idx, "train")
            fid = idx_to_file.get(idx)
            if fid is None:
                continue
            if cap_counts[lab][split] >= {
                "train": MAX_TRAIN_PER_LABEL,
                "val": MAX_VAL_PER_LABEL,
                "test": MAX_TEST_PER_LABEL,
            }[split]:
                continue
            if per_file_counts[lab][split][fid] >= per_file_cap[lab][split]:
                continue
            dst = out_root / split / lab / img.name
            shutil.copy2(img, dst)
            summary[lab][split] += 1
            cap_counts[lab][split] += 1
            per_file_counts[lab][split][fid] += 1

    print(f"\n[CWRU-STRICT] 已整理: {images_root} -> {out_root}")
    for lab in labels:
        if lab in summary:
            s = summary[lab]
            print(f"  类别 {lab}: train={s['train']}, val={s['val']}, test={s['test']}")
        else:
            print(f"  类别 {lab}: 无图像")
    total_train = sum(s["train"] for s in summary.values())
    total_val = sum(s["val"] for s in summary.values())
    total_test = sum(s["test"] for s in summary.values())
    print(f"[CWRU-STRICT] 全局: train={total_train}, val={total_val}, test={total_test}")
    if total_val == 0 or total_test == 0:
        print("[CWRU-STRICT][WARN] val 或 test 为空，请检查 file_ranges_cwru.csv 或 1.2/1.3 生成数量。")


def main() -> None:
    import sys

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    from utils.project_paths import (  # noqa: E402
        FILE_RANGES_CWRU_V2,
        GADF_IMAGES_ENH_V2,
        GADF_IMAGES_V2,
        GASF_IMAGES_ENH_V2,
        GASF_IMAGES_V2,
        YOLO_DS_V2,
        YOLO_DS_V2_ENH,
        ensure_v2_dirs,
    )

    ensure_v2_dirs()
    ranges_csv = FILE_RANGES_CWRU_V2
    print("=== CWRU GAF v2 严格划分 + 每类 1500 张 ===")
    print(f"  读取: {ranges_csv}")
    ranges = load_file_ranges(ranges_csv)
    idx_to_split = build_idx_to_split(ranges)
    idx_to_file = build_idx_to_file(ranges)

    gasf_root = GASF_IMAGES_V2
    gadf_root = GADF_IMAGES_V2
    out_root = YOLO_DS_V2
    print("\n=== GASF 严格划分 ===")
    prepare_branch_strict(
        gasf_root, out_root / "gasf_cls_strict", idx_to_split, idx_to_file, is_gasf=True
    )
    print("\n=== GADF 严格划分 ===")
    prepare_branch_strict(
        gadf_root, out_root / "gadf_cls_strict", idx_to_split, idx_to_file, is_gasf=False
    )

    gasf_enh = GASF_IMAGES_ENH_V2
    gadf_enh = GADF_IMAGES_ENH_V2
    if gasf_enh.exists() and gadf_enh.exists():
        out_enh = YOLO_DS_V2_ENH
        print("\n=== SA-GAF GASF 严格划分 ===")
        prepare_branch_strict(
            gasf_enh, out_enh / "gasf_cls_strict", idx_to_split, idx_to_file, is_gasf=True
        )
        print("\n=== SA-GAF GADF 严格划分 ===")
        prepare_branch_strict(
            gadf_enh, out_enh / "gadf_cls_strict", idx_to_split, idx_to_file, is_gasf=False
        )
        print(f"[CWRU-STRICT] 增强数据集: {out_enh}")

    print("\n[CWRU-STRICT] 完成.")
    print(f"   gasf_cls_strict: {out_root / 'gasf_cls_strict'}")
    print(f"   gadf_cls_strict: {out_root / 'gadf_cls_strict'}")


if __name__ == "__main__":
    main()
