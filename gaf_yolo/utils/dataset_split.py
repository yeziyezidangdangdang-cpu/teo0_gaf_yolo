"""
数据集划分工具（与论文 2.2 / 图5 一致）。

论文方法（temporal_within_file）：
  对每个原始文件内的片段索引按时间顺序 70% / 15% / 15% → train / val / test，
  同一文件内片段不跨集合。

扩展方法（whole_file）：
  按 file_id 将整文件划入 train / val / test（MCC5 变工况泛化可选用）。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


def build_idx_to_split_temporal_within_file(
    ranges: List[Tuple[int, int, int]],
    ratios: Dict[str, float] = SPLIT_RATIOS,
) -> Dict[int, str]:
    """论文默认：每个文件内部按片段索引时间顺序划分。"""
    mapping: Dict[int, str] = {}
    for _fid, s, e in ranges:
        length = e - s
        if length <= 0:
            continue
        if length < 3:
            for idx in range(s, e):
                mapping[idx] = "train"
            continue

        n_train = max(1, int(length * ratios["train"]))
        n_val = max(1, int(length * ratios["val"]))
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


def build_file_id_to_split_whole_file(
    file_labels: Dict[int, int],
    ratios: Dict[str, float] = SPLIT_RATIOS,
) -> Dict[int, str]:
    """整文件划分：同类内按 file_id 排序后按文件数比例分配。"""
    by_label: Dict[int, List[int]] = {}
    for fid, lab in file_labels.items():
        by_label.setdefault(int(lab), []).append(int(fid))

    file_to_split: Dict[int, str] = {}
    for lab in sorted(by_label.keys()):
        fids = sorted(set(by_label[lab]))
        n = len(fids)
        if n == 0:
            continue
        if n < 3:
            for f in fids:
                file_to_split[f] = "train"
            continue

        n_train = max(1, int(n * ratios["train"]))
        n_val = max(1, int(n * ratios["val"]))
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        n_test = n - n_train - n_val
        if n_test <= 0:
            n_test = 1
            if n_train > n_val:
                n_train -= 1
            else:
                n_val -= 1

        for i, fid in enumerate(fids):
            if i < n_train:
                file_to_split[fid] = "train"
            elif i < n_train + n_val:
                file_to_split[fid] = "val"
            else:
                file_to_split[fid] = "test"
    return file_to_split


def build_idx_to_split_whole_file(
    ranges: List[Tuple[int, int, int]],
    file_labels: Dict[int, int],
    ratios: Dict[str, float] = SPLIT_RATIOS,
) -> Dict[int, str]:
    file_to_split = build_file_id_to_split_whole_file(file_labels, ratios)
    mapping: Dict[int, str] = {}
    for fid, s, e in ranges:
        split = file_to_split.get(fid, "train")
        for idx in range(s, e):
            mapping[idx] = split
    return mapping
