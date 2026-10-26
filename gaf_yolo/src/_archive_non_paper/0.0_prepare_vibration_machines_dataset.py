import os
from pathlib import Path

import numpy as np
import pandas as pd


# === 配置区 ===
# 原始数据集根目录（不强制移动到项目下，如有需要可自行修改）
RAW_ROOT = Path(r"D:\TEP_2_repro\gaf_yolo\data\raw\Vibration based Fault Diagnosis of Machines\Data")

# 使用的工况类别（4 类：健康 + 3 种故障）
CONDITION_DIRS = [
    "Ideal",          # 健康
    "Wear",           # 磨损
    "Cracking",       # 裂纹
    "Offset_Pulley",  # 皮带轮偏移
]

# 只用 Z 轴，目录形式如：Ideal\Ideal_Z\M(1).csv
AXIS_SUFFIX = "Z"

# 输出：每行 [t_0, ..., t_127, label]
OUTPUT_CSV = Path(r"D:\TEP_2_repro\gaf_yolo\data\processed\final_dataset_vibration_machines.csv")

SEGMENT_LENGTH = 128
STRIDE = 64


def read_single_file(csv_path: Path) -> np.ndarray:
    """
    读取单个 M(k).csv，返回一维时间序列（Channel 1）。
    文件前面是一些元数据，从第 12 行（索引 11）开始为：
    index, value, ...
    """
    # 跳过前 11 行头信息
    # 注意：该数据集中常见单位字符如 `m/s²`，源文件往往是 Windows-1252(cp1252) 编码，
    # 直接按 utf-8 读会触发 UnicodeDecodeError。
    try:
        df = pd.read_csv(csv_path, header=None, skiprows=11, encoding="utf-8")
    except UnicodeDecodeError:
        # cp1252 与 latin1 都能覆盖 0xB2(²) 这类字节；优先 cp1252 更贴近实际来源
        df = pd.read_csv(csv_path, header=None, skiprows=11, encoding="cp1252")
    # 第 0 列是样本索引，第 1 列是振动值
    # 有些尾部可能为空行，这里自动丢弃 NaN
    values = pd.to_numeric(df.iloc[:, 1], errors="coerce").dropna().to_numpy(dtype=float)
    return values


def iter_condition_signals():
    """
    遍历各工况目录，依次产生 (signal, label_id)。
    label_id 根据 CONDITION_DIRS 的顺序映射为 0..C-1。
    """
    for label_id, cond in enumerate(CONDITION_DIRS):
        cond_dir = RAW_ROOT / cond
        axis_dir = cond_dir / f"{cond}_{AXIS_SUFFIX}"
        if not axis_dir.exists():
            # 某个工况下可能缺少该轴，直接跳过
            continue

        for csv_path in sorted(axis_dir.glob("M(*).csv")):
            yield csv_path, label_id


def build_segments() -> pd.DataFrame:
    all_rows = []
    per_label_files = {}

    for csv_path, label_id in iter_condition_signals():
        signal = read_single_file(csv_path)
        if signal.size < SEGMENT_LENGTH:
            print(f"⚠ 跳过 {csv_path}: 长度 {signal.size} < {SEGMENT_LENGTH}")
            continue

        start = 0
        seg_idx = 0
        while start + SEGMENT_LENGTH <= signal.size:
            segment = signal[start : start + SEGMENT_LENGTH]
            all_rows.append(segment.tolist() + [label_id])
            start += STRIDE
            seg_idx += 1

        per_label_files[label_id] = per_label_files.get(label_id, 0) + 1
        print(f"✅ {csv_path} → 生成 {seg_idx} 个片段（label={label_id}）")

    if not all_rows:
        raise RuntimeError("未生成任何样本，请检查 RAW_ROOT / CONDITION_DIRS 配置。")

    columns = [f"t_{i}" for i in range(SEGMENT_LENGTH)] + ["label"]
    df = pd.DataFrame(all_rows, columns=columns)

    print("=== 每个标签使用的源文件数 ===")
    for lid, cnt in sorted(per_label_files.items()):
        print(f"  label {lid}: {cnt} 个 M(k).csv 文件")

    return df


def main():
    os.makedirs(OUTPUT_CSV.parent, exist_ok=True)
    df_segments = build_segments()

    df_segments.to_csv(OUTPUT_CSV, index=False, header=False)

    print("🎉 Vibration Machines 数据集切片完成")
    print(f"   输出文件: {OUTPUT_CSV}")
    print(f"   样本数: {len(df_segments)}，每样本长度: {SEGMENT_LENGTH}")
    label_counts = df_segments["label"].value_counts().sort_index()
    print("   每个标签的样本数：")
    for lab, cnt in label_counts.items():
        print(f"     label {lab}: {cnt}")


if __name__ == "__main__":
    main()

