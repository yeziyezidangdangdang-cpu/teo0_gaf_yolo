import os
from pathlib import Path

import numpy as np
import pandas as pd


# === 配置区 ===
# SUBF v1.0 原始 CSV 根目录
SUBF_ROOT = Path(r"D:\TEP_2_repro\gaf_yolo\data\raw\SUBF v1.0\CSV Files")

# 输出：每行 [t_0, ..., t_1023, label]，无表头
OUTPUT_CSV = Path(r"D:\TEP_2_repro\gaf_yolo\data\processed\final_dataset_subf.csv")

# 滑动窗口设置（参考 SUBF_GAF_ANALYSIS）
SEGMENT_LENGTH = 1024
STRIDE = 512


def iter_class_files():
    """
    遍历 SUBF 的类别子文件夹，返回 (csv_path, label_id)。
    子文件夹名作为类别名，例如：
    - 'Healthy'
    - 'Inner Race Fault'
    - 'Outer Race Fault'
    """
    subdirs = [d for d in SUBF_ROOT.iterdir() if d.is_dir()]
    # 按名称排序后映射到 0/1/2
    subdirs_sorted = sorted(subdirs, key=lambda p: p.name)
    class_to_label = {p.name: idx for idx, p in enumerate(subdirs_sorted)}

    for cls_dir in subdirs_sorted:
        label = class_to_label[cls_dir.name]
        for csv_path in sorted(cls_dir.glob("XYZ_*.csv")):
            yield csv_path, label


def build_segments() -> pd.DataFrame:
    """读取 SUBF 原始 CSV，按模值合成 + 滑动窗口切成片段。"""
    all_rows = []

    for csv_path, label in iter_class_files():
        # 原始 SUBF XYZ_*.csv：无表头，三列 X/Y/Z
        df = pd.read_csv(csv_path, header=None)
        if df.shape[1] < 3:
            raise ValueError(f"{csv_path} 列数不足 3，无法解析 X/Y/Z")

        x = df.iloc[:, 0].to_numpy(dtype=float)
        y = df.iloc[:, 1].to_numpy(dtype=float)
        z = df.iloc[:, 2].to_numpy(dtype=float)

        # 三轴合成为模值信号
        mag = np.sqrt(x**2 + y**2 + z**2)

        if mag.size < SEGMENT_LENGTH:
            print(f"⚠ 跳过 {csv_path}：长度 {mag.size} < {SEGMENT_LENGTH}")
            continue

        start = 0
        seg_idx = 0
        while start + SEGMENT_LENGTH <= mag.size:
            segment = mag[start : start + SEGMENT_LENGTH]
            all_rows.append(segment.tolist() + [label])
            start += STRIDE
            seg_idx += 1

        print(f"✅ {csv_path.relative_to(SUBF_ROOT)} → 生成 {seg_idx} 个片段（label={label}）")

    if not all_rows:
        raise RuntimeError("未生成任何 SUBF 片段，请检查数据路径和滑动参数。")

    columns = [f"t_{i}" for i in range(SEGMENT_LENGTH)] + ["label"]
    return pd.DataFrame(all_rows, columns=columns)


def main():
    os.makedirs(OUTPUT_CSV.parent, exist_ok=True)
    df_segments = build_segments()

    df_segments.to_csv(OUTPUT_CSV, index=False, header=False)

    print("🎉 SUBF v1.0 切片完成")
    print(f"   输出文件: {OUTPUT_CSV}")
    print(f"   样本数: {len(df_segments)}，每样本长度: {SEGMENT_LENGTH}")
    label_counts = df_segments["label"].value_counts().sort_index()
    print("   每个标签的样本数：")
    for lab, cnt in label_counts.items():
        print(f"     label {lab}: {cnt}")


if __name__ == "__main__":
    main()

