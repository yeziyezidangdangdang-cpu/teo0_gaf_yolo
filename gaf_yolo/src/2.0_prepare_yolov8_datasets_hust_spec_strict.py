"""
基于已生成的 HUST STFT 频谱图，按「按文件严格划分」的 YOLOv8 分类数据集（与 GAF 2.2 策略一致）。

目的：
- 避免同一原始 .xls 文件切出来的高度重叠片段同时出现在 train/val/test 中，降低数据泄漏。
- 使用 file_ranges_hust.csv 将每个文件内片段按 70%/15%/15% 划入 train/val/test。
- 每类最多保留 1500 张图（1050 train / 225 val / 225 test），与 GAF 侧「每类 1500 张」一致。

输入：
- 频谱图：project_root / data / spectrograms / <label> / spec_<label>_<idx>.png
- 文件划分：project_root / data / processed / file_ranges_hust.csv

输出：
- project_root / yolo_datasets / spec_cls_strict
  - train|val|test / <label>/*.png
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
import re
from typing import Dict, List, Tuple


SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}
# 每类最多 1500 张，按 70/15/15 分配
MAX_PER_LABEL = 1500
MAX_TRAIN_PER_LABEL = int(MAX_PER_LABEL * SPLIT_RATIOS["train"])   # 1050
MAX_VAL_PER_LABEL = int(MAX_PER_LABEL * SPLIT_RATIOS["val"])       # 225
MAX_TEST_PER_LABEL = MAX_PER_LABEL - MAX_TRAIN_PER_LABEL - MAX_VAL_PER_LABEL  # 225

FNAME_RE = re.compile(r"^spec_(\d+)_(\d+)\.png$", re.IGNORECASE)


def load_file_ranges(csv_path: Path) -> List[Tuple[int, int, int]]:
    """读取 file_ranges_hust.csv，返回 [(file_id, start_idx, end_idx), ...]。"""
    if not csv_path.is_file():
        raise FileNotFoundError(f"file_ranges_hust.csv 不存在: {csv_path}")
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
    """按每个文件的 [start_idx, end_idx) 在时间轴上 70%/15%/15% 划分，构建 idx -> split。"""
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


def extract_index(path: Path) -> int:
    """从 spec_<label>_<idx>.png 解析 idx。"""
    m = FNAME_RE.match(path.name)
    if not m:
        raise ValueError(f"文件名不符合 spec_<label>_<idx>.png 格式: {path}")
    return int(m.group(2))


def prepare_strict(
    images_root: Path,
    out_root: Path,
    idx_to_split: Dict[int, str],
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
    # 每类每 split 已复制数量，用于 1500 上限
    cap_counts: Dict[str, Dict[str, int]] = {lab: {"train": 0, "val": 0, "test": 0} for lab in labels}

    for lab_dir in label_dirs:
        lab = lab_dir.name
        imgs = sorted([p for p in lab_dir.glob("*.png") if p.is_file()], key=extract_index)
        if not imgs:
            print(f"[STRICT-SPEC] 标签 {lab} 在 {lab_dir} 下未找到 PNG 图像")
            continue

        summary.setdefault(lab, {"train": 0, "val": 0, "test": 0})
        for img in imgs:
            idx = extract_index(img)
            split = idx_to_split.get(idx, "train")

            # 每类每 split 上限
            if split == "train" and cap_counts[lab]["train"] >= MAX_TRAIN_PER_LABEL:
                continue
            if split == "val" and cap_counts[lab]["val"] >= MAX_VAL_PER_LABEL:
                continue
            if split == "test" and cap_counts[lab]["test"] >= MAX_TEST_PER_LABEL:
                continue

            dst = out_root / split / lab / img.name
            shutil.copy2(img, dst)
            summary[lab][split] += 1
            cap_counts[lab][split] += 1

    print(f"\n[STRICT-SPEC] 已整理严格划分数据集: {images_root} -> {out_root}")
    print("划分统计（每类 train/val/test 张数，每类上限 1500）：")
    for lab in labels:
        if lab in summary:
            s = summary[lab]
            print(f"  类别 {lab}: train={s['train']}, val={s['val']}, test={s['test']}")
        else:
            print(f"  类别 {lab}: 无图像")
    total_train = sum(s["train"] for s in summary.values())
    total_val = sum(s["val"] for s in summary.values())
    total_test = sum(s["test"] for s in summary.values())
    print(f"[STRICT-SPEC] 全局: train={total_train}, val={total_val}, test={total_test}")
    if total_val == 0 or total_test == 0:
        print("[STRICT-SPEC][WARN] val 或 test 为空，请检查 file_ranges 或 1.2 生成图像数量。")


def main():
    project_root = Path(__file__).resolve().parents[1]
    images_root = project_root / "data" / "spectrograms"
    out_root = project_root / "yolo_datasets" / "spec_cls_strict"
    ranges_csv = project_root / "data" / "processed" / "file_ranges_hust.csv"

    print("=== 准备 HUST STFT 频谱图严格划分分类数据集（按文件 + 每类 1500 张）===")
    print(f"  读取文件划分: {ranges_csv}")
    ranges = load_file_ranges(ranges_csv)
    idx_to_split = build_idx_to_split(ranges)
    prepare_strict(images_root, out_root, idx_to_split)
    print("\n[STRICT-SPEC] 完成。")
    print(f"   SPEC_STRICT root: {out_root}")


if __name__ == "__main__":
    main()
