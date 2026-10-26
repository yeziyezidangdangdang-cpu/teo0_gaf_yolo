"""
0.0_prepare_cwru_stft.py

将 CWRU CSV 数据集切片并打标签，供 STFT 频谱图 + YOLO（方案 A）使用。

数据源：
- D:/TEP_2_repro/gaf_yolo/data/raw/CWRU/csv/*.csv
  - 每个文件为单通道振动信号（无表头）

输出：
- D:/TEP_2_repro/gaf_yolo/data/processed/final_dataset_cwru.csv
  - 每行：[t_0, t_1, ..., t_{L-1}, label]
  - label 取值：
    - 0: Normal（文件名前缀以 'N' 开头）
    - 1: Inner race fault（前缀 'IR'）
    - 2: Outer race fault（前缀 'OR'）
    - 3: Ball fault（前缀 'B'）
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import utils.project_paths as pp  # noqa: E402

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


RAW_DIR = Path(str(pp.CWRU_CSV_DIR))
OUTPUT_CSV = Path(str(pp.FINAL_DATASET_CWRU))

# 与 HUST 思路一致：加长片段以提升 STFT 时频分辨率（原 128 太短导致频谱图"压扁"）
# 1024 点 @ 12kHz ≈ 85ms，可覆盖更多轴承故障周期
SEGMENT_LENGTH = 1024
STRIDE = 256


def map_fault_to_label(filename: str) -> int:
    """
    将文件名映射到 0/1/2/3 四类标签（按“正常/内圈/外圈/滚动体”的顺序）：
    - 以 'N' 开头   → 0（Normal）
    - 以 'IR' 开头  → 1（Inner race）
    - 以 'OR' 开头  → 2（Outer race）
    - 以 'B' 开头   → 3（Ball fault）
    """
    name = filename
    if "." in name:
        name = name.split(".", 1)[0]
    token = name.split("_", 1)[0]

    if token.startswith("N"):
        return 0
    if token.startswith("IR"):
        return 1
    if token.startswith("OR"):
        return 2
    if token.startswith("B"):
        return 3

    raise ValueError(f"无法从文件名映射标签: {filename}")


def build_segments() -> pd.DataFrame:
    all_rows: list[list[float]] = []

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"未找到 CWRU CSV 目录: {RAW_DIR}")

    for filename in sorted(os.listdir(RAW_DIR)):
        if not filename.lower().endswith(".csv"):
            continue
        path = RAW_DIR / filename
        try:
            signal = pd.read_csv(path, header=None).iloc[:, 0].to_numpy(dtype=np.float32)
        except Exception as e:
            print(f"[WARN] Read fail {filename}: {e}")
            continue

        n = signal.size
        if n < SEGMENT_LENGTH:
            print(f"[WARN] Skip {filename}: len {n} < {SEGMENT_LENGTH}")
            continue

        label = map_fault_to_label(filename)
        start = 0
        seg_idx = 0
        while start + SEGMENT_LENGTH <= n:
            seg = signal[start : start + SEGMENT_LENGTH]
            row = np.concatenate([seg, np.array([label], dtype=np.float32)])
            all_rows.append(row.tolist())
            start += STRIDE
            seg_idx += 1

        print(f"[OK] {filename} -> {seg_idx} segments (label={label})")

    if not all_rows:
        raise RuntimeError("未生成任何 CWRU 片段，请检查 RAW_DIR。")

    cols_signal = [f"t_{i}" for i in range(SEGMENT_LENGTH)]
    columns = cols_signal + ["label"]
    return pd.DataFrame(all_rows, columns=columns)


def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = build_segments()
    df.to_csv(OUTPUT_CSV, index=False, header=False)

    print("[CWRU] STFT dataset prepared")
    print(f"   输出文件: {OUTPUT_CSV}")
    print(f"   样本数: {len(df)}，每样本长度: {SEGMENT_LENGTH}")
    counts = df.iloc[:, -1].value_counts().sort_index()
    print("   每个标签的样本数：")
    for lab, cnt in counts.items():
        print(f"     label {int(lab)}: {int(cnt)}")


if __name__ == "__main__":
    main()

