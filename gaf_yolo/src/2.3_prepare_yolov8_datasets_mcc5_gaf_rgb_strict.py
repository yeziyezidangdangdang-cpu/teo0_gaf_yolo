"""
MCC5 GAF-RGB YOLO 数据集（支持论文划分 / 整文件划分）。

划分模式（--split-mode）：
  temporal   — 论文图5：每个 CSV 文件内片段按时间 70/15/15（与 HUST/CWRU strict 一致）
  whole_file — 整文件划入 train/val/test（MCC5 变工况补充实验）

用法：
  python src/2.3_prepare_yolov8_datasets_mcc5_gaf_rgb_strict.py --exp-point c
  python src/2.3_prepare_yolov8_datasets_mcc5_gaf_rgb_strict.py --exp-point b
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset_split import (  # noqa: E402
    build_idx_to_split_temporal_within_file,
    build_idx_to_split_whole_file,
)
from utils.project_paths import (  # noqa: E402
    FILE_RANGES_MCC5,
    GADF_RGB_MCC5,
    GASF_RGB_MCC5,
    MCC5_META_JSON,
    YOLO_DS_MCC5,
    ensure_mcc5_dirs,
)

SPLIT_RATIOS_LOCAL = {"train": 0.7, "val": 0.15, "test": 0.15}
MAX_PER_LABEL = 1500
MAX_TRAIN = int(MAX_PER_LABEL * SPLIT_RATIOS_LOCAL["train"])
MAX_VAL = int(MAX_PER_LABEL * SPLIT_RATIOS_LOCAL["val"])
MAX_TEST = MAX_PER_LABEL - MAX_TRAIN - MAX_VAL
GASF_NAME_RE = re.compile(r"^gasf_rgb_(\d+)_(\d+)\.png$", re.IGNORECASE)
GADF_NAME_RE = re.compile(r"^gadf_rgb_(\d+)_(\d+)\.png$", re.IGNORECASE)
SPLIT_MODES = ("temporal", "whole_file")
EXP_POINT_TO_MODE = {"c": "temporal", "b": "whole_file"}


def load_file_ranges(csv_path: Path) -> List[Tuple[int, int, int]]:
    ranges: List[Tuple[int, int, int]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ranges.append((int(row["file_id"]), int(row["start_idx"]), int(row["end_idx"])))
    ranges.sort(key=lambda x: x[1])
    return ranges


def load_file_labels(meta_path: Path) -> Dict[int, int]:
    if not meta_path.is_file():
        raise FileNotFoundError(f"缺少元数据: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    labels: Dict[int, int] = {}
    for item in meta.get("files", []):
        labels[int(item["file_id"])] = int(item["label"])
    return labels


def output_roots(split_mode: str) -> tuple[Path, Path]:
    suffix = "" if split_mode == "temporal" else "_by_file"
    gasf_out = YOLO_DS_MCC5 / f"mcc5_gasf_rgb_cls_strict{suffix}"
    gadf_out = YOLO_DS_MCC5 / f"mcc5_gadf_rgb_cls_strict{suffix}"
    return gasf_out, gadf_out


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
    for split in ("train", "val", "test"):
        for lab in labels:
            (out_root / split / lab).mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Dict[str, int]] = {}
    caps = {"train": MAX_TRAIN, "val": MAX_VAL, "test": MAX_TEST}
    split_counts: Dict[str, Dict[str, int]] = {}

    for lab_dir in label_dirs:
        lab = lab_dir.name
        for img in sorted(lab_dir.glob("*.png")):
            idx = extract_idx(img.name, is_gasf=is_gasf)
            split = idx_to_split.get(idx, "train")
            split_counts.setdefault(split, {}).setdefault(lab, 0)
            if split_counts[split][lab] >= caps[split]:
                continue
            shutil.copy2(img, out_root / split / lab / img.name)
            split_counts[split][lab] += 1
            summary.setdefault(lab, {"train": 0, "val": 0, "test": 0})[split] += 1

    print(f"\n[MCC5-STRICT] {images_root.name} -> {out_root}")
    for lab in labels:
        s = summary.get(lab, {"train": 0, "val": 0, "test": 0})
        print(f"  类别 {lab}: train={s['train']}, val={s['val']}, test={s['test']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MCC5 GAF-RGB strict YOLO datasets")
    parser.add_argument(
        "--exp-point",
        choices=("c", "b"),
        default="b",
        help="B=whole_file（论文 MCC5 统一口径）；C=temporal 备用",
    )
    parser.add_argument(
        "--split-mode",
        choices=SPLIT_MODES,
        default=None,
        help="可选，直接指定划分模式（覆盖 --exp-point）",
    )
    args = parser.parse_args()

    split_mode = args.split_mode or EXP_POINT_TO_MODE[args.exp_point]
    print(f"[MCC5-STRICT] exp_point={args.exp_point.upper()} split_mode={split_mode}")
    ensure_mcc5_dirs()
    ranges = load_file_ranges(FILE_RANGES_MCC5)
    file_labels = load_file_labels(MCC5_META_JSON)

    if split_mode == "temporal":
        idx_to_split = build_idx_to_split_temporal_within_file(ranges)
        print("[MCC5-STRICT] 划分模式: temporal（论文：每文件内 70/15/15）")
    else:
        idx_to_split = build_idx_to_split_whole_file(ranges, file_labels)
        from utils.dataset_split import build_file_id_to_split_whole_file

        fts = build_file_id_to_split_whole_file(file_labels)
        print("[MCC5-STRICT] 划分模式: whole_file（整文件 70/15/15）")
        for fid in sorted(fts.keys()):
            print(f"  file_id={fid:02d} label={file_labels[fid]} -> {fts[fid]}")

    gasf_out, gadf_out = output_roots(split_mode)
    prepare_branch_strict(GASF_RGB_MCC5, gasf_out, idx_to_split, is_gasf=True)
    prepare_branch_strict(GADF_RGB_MCC5, gadf_out, idx_to_split, is_gasf=False)

    sidecar = YOLO_DS_MCC5 / f"split_mode_{split_mode}.json"
    sidecar.write_text(
        json.dumps(
            {
                "exp_point": args.exp_point,
                "split_mode": split_mode,
                "gasf_out": str(gasf_out),
                "gadf_out": str(gadf_out),
                "num_files": len(file_labels),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n[MCC5-STRICT] 完成 | exp_point={args.exp_point.upper()} mode={split_mode}")
    print(f"  GASF: {gasf_out}")
    print(f"  GADF: {gadf_out}")


if __name__ == "__main__":
    main()
