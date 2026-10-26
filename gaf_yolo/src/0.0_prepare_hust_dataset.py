from __future__ import annotations

import os
import re
from pathlib import Path

import sys

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import utils.project_paths as pp  # noqa: E402

# === 配置区 ===
# HUST 原始数据目录（repro）；切片 CSV 写入 archive（TEP_2）
RAW_DIR = pp.HUST_RAW_DIR

OUTPUT_CSV = pp.FINAL_DATASET_HUST
FILE_RANGES_CSV = pp.FILE_RANGES_HUST

# 只选哪些转速（Hz）
SELECTED_SPEEDS = {30}

# 只选哪些故障类型（根据文件名前缀）
FAULT_MAP = {
    "H": 0,  # Healthy
    "B": 1,  # Ball fault
    "I": 2,  # Inner race
    "O": 3,  # Outer race
}

# 滑动窗口设置
# 目标：每标签约 1500 张图。片段数 = (n - SEGMENT_LENGTH) // STRIDE + 1；
# STRIDE=64 时约 10 万点/文件可得 ~1530 段/类；若原始文件较短可改为 32
SEGMENT_LENGTH = 2048
STRIDE = 64


def is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def load_hust_file(path: Path) -> np.ndarray:
    """
    解析 HUST 原始 .xls 文本，返回形状 (N, 5) 的数组：
    [time, speed, X, Y, Z]
    """
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            if all(is_float(p) for p in parts):
                rows.append([float(p) for p in parts])
    if not rows:
        raise ValueError(f"未能在文件中找到数值数据: {path}")
    return np.asarray(rows, dtype=float)


def iter_selected_files():
    """
    遍历 RAW_DIR 下的 .xls 文件，筛选:
    - 文件名形如 'H_30Hz.xls', 'B_30Hz.xls', 'I_30Hz.xls', 'O_30Hz.xls'
    - 只保留 SELECTED_SPEEDS 中的转速
    """
    pattern = re.compile(r"^([HBI O])_(\d+)Hz\.xls$", re.IGNORECASE)
    for path in sorted(RAW_DIR.glob("*.xls")):
        m = pattern.match(path.name.replace(" ", ""))
        if not m:
            continue
        prefix, hz_str = m.groups()
        speed = int(hz_str)
        if speed not in SELECTED_SPEEDS:
            continue
        key = prefix.upper()
        if key not in FAULT_MAP:
            continue
        label = FAULT_MAP[key]
        yield path, label


def build_segments() -> tuple[pd.DataFrame, list[tuple[int, int, int]]]:
    """
    返回 (DataFrame, file_ranges)。
    file_ranges: [(file_id, start_idx, end_idx), ...]，用于 2.2 按时间段严格划分。
    """
    all_rows = []
    file_ranges: list[tuple[int, int, int]] = []

    for file_id, (path, label) in enumerate(iter_selected_files()):
        data = load_hust_file(path)
        # 列: time, speed, X, Y, Z
        x = data[:, 2]
        y = data[:, 3]
        z = data[:, 4]

        n = x.size
        if n < SEGMENT_LENGTH:
            print(f"⚠ 跳过 {path.name}：长度 {n} < {SEGMENT_LENGTH}")
            continue

        start_idx = len(all_rows)
        start = 0
        seg_idx = 0
        while start + SEGMENT_LENGTH <= n:
            seg_x = x[start : start + SEGMENT_LENGTH]
            seg_y = y[start : start + SEGMENT_LENGTH]
            seg_z = z[start : start + SEGMENT_LENGTH]
            row = np.concatenate([seg_x, seg_y, seg_z, np.array([label], dtype=float)])
            all_rows.append(row.tolist())
            start += STRIDE
            seg_idx += 1

        end_idx = len(all_rows)
        file_ranges.append((file_id, start_idx, end_idx))
        rel = path.relative_to(RAW_DIR)
        print(f"✅ {rel} → 生成 {seg_idx} 个片段（label={label}, file_id={file_id}, 全局索引 [{start_idx},{end_idx})）")

    if not all_rows:
        raise RuntimeError("未生成任何 HUST 片段，请检查路径和筛选条件。")

    L = SEGMENT_LENGTH
    cols_x = [f"x_{i}" for i in range(L)]
    cols_y = [f"y_{i}" for i in range(L)]
    cols_z = [f"z_{i}" for i in range(L)]
    columns = cols_x + cols_y + cols_z + ["label"]
    return pd.DataFrame(all_rows, columns=columns), file_ranges


def main():
    os.makedirs(OUTPUT_CSV.parent, exist_ok=True)
    df, file_ranges = build_segments()
    df.to_csv(OUTPUT_CSV, index=False, header=False)

    # 写出严格划分用：每文件的 (file_id, start_idx, end_idx)
    ranges_df = pd.DataFrame(file_ranges, columns=["file_id", "start_idx", "end_idx"])
    ranges_df.to_csv(FILE_RANGES_CSV, index=False)
    print(f"   严格划分索引: {FILE_RANGES_CSV}")

    print("🎉 HUST 数据集切片完成")
    print(f"   输出文件: {OUTPUT_CSV}")
    print(f"   样本数: {len(df)}，每样本长度: {SEGMENT_LENGTH} × 3 轴")
    counts = df.iloc[:, -1].value_counts().sort_index()
    print("   每个标签的样本数：")
    for lab, cnt in counts.items():
        print(f"     label {int(lab)}: {int(cnt)}")


if __name__ == "__main__":
    main()

