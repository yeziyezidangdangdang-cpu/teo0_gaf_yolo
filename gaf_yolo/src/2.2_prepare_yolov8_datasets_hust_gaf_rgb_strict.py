"""
为 HUST GAF-RGB / GADF-RGB 图像构建“按文件严格划分”的 YOLOv8 分类数据集。

目的：
- 避免同一原始 .xls 文件切出来的高度重叠片段同时出现在 train / val / test 中，
  降低数据泄漏导致的“接近 100% 准确率”情况，使评估结果更接近真实泛化能力。

思路：
- 使用 `data/processed/file_ranges_hust.csv` 中记录的 (file_id, start_idx, end_idx)，
  将每个文件内部的片段索引区间 [start_idx, end_idx) 按 70%/15%/15% 在“时间轴”上切成
  train/val/test 三段（文件内前 70% → train，中间 15% → val，最后 15% → test）；
- 对于 GASF-RGB 和 GADF-RGB 两个分支，按照相同的 idx→split 规则复制图片。

输入目录：
- GASF-RGB 源图像：project_root / data / gaf_rgb / gasf / <label> / gasf_rgb_<label>_<idx>.png
- GADF-RGB 源图像：project_root / data / gaf_rgb / gadf / <label> / gadf_rgb_<label>_<idx>.png

输出目录：
- GASF-RGB 数据集：project_root / yolo_datasets / hust_gasf_rgb_cls_strict / train|val|test / <label>/*.png
- GADF-RGB 数据集：project_root / yolo_datasets / hust_gadf_rgb_cls_strict / train|val|test / <label>/*.png
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
import re
from typing import Dict, List, Tuple


SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}

GASF_NAME_RE = re.compile(r"^gasf_rgb_(\d+)_(\d+)\.png$", re.IGNORECASE)
GADF_NAME_RE = re.compile(r"^gadf_rgb_(\d+)_(\d+)\.png$", re.IGNORECASE)


def load_file_ranges(csv_path: Path) -> List[Tuple[int, int, int]]:
    """
    读取 file_ranges_hust.csv，返回列表 [(file_id, start_idx, end_idx), ...]，
    其中 start_idx/end_idx 为 [start, end) 半开区间。
    """
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
    """
    基于每个文件的 [start_idx, end_idx) 区间，在“时间轴”上做 70%/15%/15% 划分，
    构建 idx -> split("train"/"val"/"test") 的查表。
    """
    mapping: Dict[int, str] = {}
    for fid, s, e in ranges:
        length = e - s
        if length <= 0:
            continue
        # 样本极少时，全部放入 train，避免 val/test 为空
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
        raise ValueError(f"文件名不符合预期格式: {name}")
    return int(m.group(2))


def prepare_branch_strict(
    images_root: Path,
    out_root: Path,
    idx_to_split: Dict[int, str],
    is_gasf: bool,
) -> None:
    if not images_root.exists():
        raise FileNotFoundError(f"源图像目录不存在: {images_root}")

    if out_root.exists():
        shutil.rmtree(out_root)

    # 发现所有标签
    label_dirs = sorted([p for p in images_root.iterdir() if p.is_dir()])
    labels = [p.name for p in label_dirs]

    # 创建输出目录
    for split in SPLIT_RATIOS.keys():
        for lab in labels:
            (out_root / split / lab).mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Dict[str, int]] = {}

    for lab_dir in label_dirs:
        lab = lab_dir.name
        imgs = sorted([p for p in lab_dir.glob("*.png") if p.is_file()])
        if not imgs:
            print(f"[STRICT-GAF] 标签 {lab} 在 {lab_dir} 下未找到 PNG 图像")
            continue

        for img in imgs:
            idx = extract_idx(img.name, is_gasf=is_gasf)
            split = idx_to_split.get(idx, "train")
            dst = out_root / split / lab / img.name
            shutil.copy2(img, dst)
            summary.setdefault(lab, {"train": 0, "val": 0, "test": 0})[split] += 1

    print(f"\n[STRICT-GAF] 已整理数据集: {images_root} -> {out_root}")
    print("划分统计（每类 train/val/test 张数）：")
    for lab in labels:
        if lab in summary:
            s = summary[lab]
            print(f"  类别 {lab}: train={s['train']}, val={s['val']}, test={s['test']}")
        else:
            print(f"  类别 {lab}: 无图像")

    # 额外警告：若某个 split 全局为空，明确提示，方便用户检查
    total_train = sum(s["train"] for s in summary.values())
    total_val = sum(s["val"] for s in summary.values())
    total_test = sum(s["test"] for s in summary.values())
    print(
        f"[STRICT-GAF] 全局统计: train={total_train}, val={total_val}, test={total_test}"
    )
    if total_val == 0 or total_test == 0:
        print(
            "[STRICT-GAF][WARN] 存在 val 或 test 为空的情况，请检查 file_ranges_hust.csv "
            "或考虑增加文件数量/调整划分策略。"
        )


def main():
    project_root = Path(__file__).resolve().parents[1]

    gasf_root = project_root / "data" / "gaf_rgb" / "gasf"
    gadf_root = project_root / "data" / "gaf_rgb" / "gadf"

    out_root = project_root / "yolo_datasets"
    gasf_out = out_root / "hust_gasf_rgb_cls_strict"
    gadf_out = out_root / "hust_gadf_rgb_cls_strict"

    ranges_csv = project_root / "data" / "processed" / "file_ranges_hust.csv"
    print(f"[STRICT-GAF] 读取文件划分信息: {ranges_csv}")
    ranges = load_file_ranges(ranges_csv)
    idx_to_split = build_idx_to_split(ranges)

    print("\n=== 准备 GASF-RGB 严格划分数据集 ===")
    prepare_branch_strict(gasf_root, gasf_out, idx_to_split, is_gasf=True)

    print("\n=== 准备 GADF-RGB 严格划分数据集 ===")
    prepare_branch_strict(gadf_root, gadf_out, idx_to_split, is_gasf=False)

    print("\n[STRICT-GAF] YOLOv8 严格划分 GAF 数据集已准备完成。")
    print(f"   GASF_STRICT root: {gasf_out}")
    print(f"   GADF_STRICT root: {gadf_out}")


if __name__ == "__main__":
    main()

