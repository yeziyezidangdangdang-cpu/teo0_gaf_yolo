"""
MCC5-THU 纯轴承四分类文件筛选（B 点泛化验证）。

仅保留 health / bearing_ball_H / bearing_inner_H / bearing_outer_H，
排除所有复合故障（文件名含 _and_）。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from utils.project_paths import MCC5_RAW_DIR

# 与 HUST 主实验一致：0=Healthy, 1=Ball, 2=Inner, 3=Outer
FAULT_PREFIX_TO_LABEL: dict[str, int] = {
    "health": 0,
    "bearing_ball_H": 1,
    "bearing_inner_H": 2,
    "bearing_outer_H": 3,
}

# 文件名在工况后缀前截断，例如 bearing_ball_H_speed_circulation_...
_FAULT_STEM_RE = re.compile(
    r"^(health|bearing_ball_H|bearing_inner_H|bearing_outer_H)_"
    r"(?:speed_circulation|torque_circulation)_"
)


def parse_fault_prefix(filename: str) -> str | None:
    """从 CSV 文件名解析故障前缀；非纯轴承类返回 None。"""
    stem = Path(filename).stem
    if "_and_" in stem:
        return None
    m = _FAULT_STEM_RE.match(stem)
    if not m:
        return None
    return m.group(1)


def label_from_path(path: Path) -> int | None:
    prefix = parse_fault_prefix(path.name)
    if prefix is None:
        return None
    return FAULT_PREFIX_TO_LABEL.get(prefix)


def iter_selected_csv_files(raw_dir: Path | None = None) -> Iterator[tuple[Path, int]]:
    """按 (path, label) 遍历已筛选的 MCC5 CSV。"""
    root = raw_dir or MCC5_RAW_DIR
    if not root.is_dir():
        raise FileNotFoundError(f"MCC5 原始目录不存在: {root}")

    for path in sorted(root.rglob("*.csv")):
        lab = label_from_path(path)
        if lab is not None:
            yield path, lab


def count_selected_files(raw_dir: Path | None = None) -> dict[int, int]:
    counts: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
    for _, lab in iter_selected_csv_files(raw_dir):
        counts[lab] += 1
    return counts
