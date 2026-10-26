import os
from pathlib import Path

import numpy as np
import pandas as pd


# === 配置区 ===
# 吸尘器振动监测数据集根目录（仅使用振动通道）
VIBRATION_DIR = Path(
    r"D:\TEP_2_repro\gaf_yolo\data\raw\Vacuum Cleaner Vibration Monitoring\Vibration"
)

# 输出：形状为 [t_0, ..., t_127, label] 的表格，供 1.1 使用
OUTPUT_CSV = Path(r"D:\TEP_2_repro\gaf_yolo\data\processed\final_dataset_vacuum.csv")

# 滑动窗口参数（与 CWRU 流水线保持一致）
SEGMENT_LENGTH = 128
STRIDE = 64

# 选择哪一轴作为 1D 信号，这里用 Z 轴
AXIS_COLUMN = "acc_z"


def iter_label_files():
    """遍历 vibration 目录下的 label_*.csv 文件，返回 (path, label_id)。"""
    for csv_path in sorted(VIBRATION_DIR.glob("label_*.csv")):
        name = csv_path.stem  # e.g. 'label_9'
        try:
            label_id = int(name.split("_", 1)[1])
        except (IndexError, ValueError):
            # 非标准命名直接跳过
            continue
        yield csv_path, label_id


def build_segments() -> pd.DataFrame:
    """读取各标签文件，按滑动窗口切成 (segment, label) 样本集合。"""
    all_rows = []

    for csv_path, label_id in iter_label_files():
        df = pd.read_csv(csv_path)
        if AXIS_COLUMN not in df.columns:
            raise ValueError(f"{csv_path} 中缺少列 {AXIS_COLUMN}")

        signal = df[AXIS_COLUMN].to_numpy(dtype=float)
        if signal.size < SEGMENT_LENGTH:
            print(f"⚠ 跳过 {csv_path.name}：长度 {signal.size} < {SEGMENT_LENGTH}")
            continue

        start = 0
        seg_idx = 0
        while start + SEGMENT_LENGTH <= signal.size:
            segment = signal[start : start + SEGMENT_LENGTH]
            all_rows.append(segment.tolist() + [label_id])
            start += STRIDE
            seg_idx += 1

        print(f"✅ {csv_path.name} → 生成 {seg_idx} 个片段（label={label_id}）")

    if not all_rows:
        raise RuntimeError("未生成任何样本，请检查数据目录和滑动窗口配置。")

    columns = [f"t_{i}" for i in range(SEGMENT_LENGTH)] + ["label"]
    return pd.DataFrame(all_rows, columns=columns)


def main():
    os.makedirs(OUTPUT_CSV.parent, exist_ok=True)
    df_segments = build_segments()

    df_segments.to_csv(OUTPUT_CSV, index=False, header=False)

    print("🎉 Vacuum 数据集切片完成")
    print(f"   输出文件: {OUTPUT_CSV}")
    print(f"   样本数: {len(df_segments)}，每样本长度: {SEGMENT_LENGTH}")
    label_counts = df_segments["label"].value_counts().sort_index()
    print("   每个标签的样本数：")
    for lab, cnt in label_counts.items():
        print(f"     label {lab}: {cnt}")


if __name__ == "__main__":
    main()

