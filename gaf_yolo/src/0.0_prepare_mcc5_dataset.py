"""
MCC5-THU：原始 CSV → 三轴滑动窗口片段 + file_ranges（strict 划分用）。

筛选：health / bearing_ball_H / bearing_inner_H / bearing_outer_H（高严重度），
排除复合故障。振动列取 motor_vibration_X/Y/Z（9 列格式时 index 3:6）。

输出：
- data/processed_mcc5/final_dataset_mcc5.csv
- data/processed_mcc5/file_ranges_mcc5.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.mcc5_files import FAULT_PREFIX_TO_LABEL, iter_selected_csv_files  # noqa: E402
from utils.project_paths import (  # noqa: E402
    FILE_RANGES_MCC5,
    FINAL_DATASET_MCC5,
    MCC5_FS,
    MCC5_META_JSON,
    MCC5_RAW_DIR,
    MCC5_SEGMENT_LENGTH,
    MCC5_STRIDE,
    ensure_mcc5_dirs,
)


def load_mcc5_vibration_xyz(path: Path) -> np.ndarray:
    """读取单文件三轴振动，返回 (N, 3) float32。"""
    df = pd.read_csv(path, header=None)
    ncol = df.shape[1]
    if ncol == 9:
        xyz = df.iloc[:, 3:6].to_numpy(dtype=np.float32)
    elif ncol == 8:
        xyz = df.iloc[:, 2:5].to_numpy(dtype=np.float32)
    else:
        raise ValueError(f"{path.name}: 期望 8 或 9 列，实际 {ncol}")
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"{path.name}: 三轴形状异常 {xyz.shape}")
    return xyz


def build_segments() -> tuple[pd.DataFrame, list[tuple[int, int, int]], dict]:
    all_rows: list[list[float]] = []
    file_ranges: list[tuple[int, int, int]] = []
    meta_files: list[dict] = []

    for file_id, (path, label) in enumerate(iter_selected_csv_files()):
        xyz = load_mcc5_vibration_xyz(path)
        n = xyz.shape[0]
        if n < MCC5_SEGMENT_LENGTH:
            print(f"⚠ 跳过 {path.name}：长度 {n} < {MCC5_SEGMENT_LENGTH}")
            continue

        start_idx = len(all_rows)
        seg_count = 0
        start = 0
        while start + MCC5_SEGMENT_LENGTH <= n:
            seg = xyz[start : start + MCC5_SEGMENT_LENGTH]  # (L, 3)
            row = np.concatenate(
                [seg[:, 0], seg[:, 1], seg[:, 2], np.array([label], dtype=float)]
            )
            all_rows.append(row.tolist())
            start += MCC5_STRIDE
            seg_count += 1

        end_idx = len(all_rows)
        file_ranges.append((file_id, start_idx, end_idx))
        meta_files.append(
            {
                "file_id": file_id,
                "path": str(path.relative_to(MCC5_RAW_DIR)),
                "label": label,
                "segments": seg_count,
                "start_idx": start_idx,
                "end_idx": end_idx,
            }
        )
        print(
            f"✅ [{file_id:02d}] {path.name} → {seg_count} 段 "
            f"(label={label}, idx [{start_idx},{end_idx}))"
        )

    if not all_rows:
        raise RuntimeError("未生成任何 MCC5 片段，请检查 raw 目录与筛选规则。")

    L = MCC5_SEGMENT_LENGTH
    cols = [f"x_{i}" for i in range(L)] + [f"y_{i}" for i in range(L)] + [f"z_{i}" for i in range(L)] + ["label"]
    df = pd.DataFrame(all_rows, columns=cols)

    meta = {
        "fs_hz": MCC5_FS,
        "segment_length": MCC5_SEGMENT_LENGTH,
        "stride": MCC5_STRIDE,
        "fault_map": FAULT_PREFIX_TO_LABEL,
        "num_files": len(file_ranges),
        "num_segments": len(df),
        "files": meta_files,
    }
    return df, file_ranges, meta


def main() -> None:
    ensure_mcc5_dirs()
    df, file_ranges, meta = build_segments()

    df.to_csv(FINAL_DATASET_MCC5, index=False, header=False)
    pd.DataFrame(file_ranges, columns=["file_id", "start_idx", "end_idx"]).to_csv(
        FILE_RANGES_MCC5, index=False
    )
    MCC5_META_JSON.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n🎉 MCC5 切片完成")
    print(f"   CSV: {FINAL_DATASET_MCC5}")
    print(f"   file_ranges: {FILE_RANGES_MCC5}")
    print(f"   样本数: {len(df)}，每样本 {MCC5_SEGMENT_LENGTH}×3 轴")
    counts = df.iloc[:, -1].value_counts().sort_index()
    print("   各类片段数：")
    for lab, cnt in counts.items():
        print(f"     label {int(lab)}: {int(cnt)}")


if __name__ == "__main__":
    main()
