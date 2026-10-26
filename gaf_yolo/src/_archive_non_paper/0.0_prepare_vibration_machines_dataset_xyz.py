import os
from pathlib import Path

import numpy as np
import pandas as pd


# === 配置区 ===
RAW_ROOT = Path(r"D:\TEP_2_repro\gaf_yolo\data\raw\Vibration based Fault Diagnosis of Machines\Data")

CONDITION_DIRS = [
    "Ideal",
    "Wear",
    "Cracking",
    "Offset_Pulley",
]

AXES = ["X", "Y", "Z"]

OUTPUT_NPY = Path(
    r"D:\TEP_2_repro\gaf_yolo\data\processed\final_dataset_vibration_machines_xyz.npy"
)

SEGMENT_LENGTH = 128
STRIDE = 64


def read_single_axis_file(csv_path: Path) -> np.ndarray:
    """读取单个轴的 M(k).csv，返回 Channel 1 时间序列。"""
    try:
        df = pd.read_csv(csv_path, header=None, skiprows=11, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, header=None, skiprows=11, encoding="cp1252")

    values = pd.to_numeric(df.iloc[:, 1], errors="coerce").dropna().to_numpy(dtype=float)
    return values


def collect_triplets_for_condition(cond: str):
    """
    收集某个工况下可配对的 (Mx, My, Mz) 三轴文件。
    返回 dict: key=M(k).csv 文件名，value={axis: path}
    """
    axis_dirs = {}
    for axis in AXES:
        axis_dir = RAW_ROOT / cond / f"{cond}_{axis}"
        axis_dirs[axis] = axis_dir

    # 以 Z 轴为基准枚举文件名，再检查 X/Y 是否都存在
    z_dir = axis_dirs["Z"]
    if not z_dir.exists():
        return {}

    triplets = {}
    for z_path in sorted(z_dir.glob("M(*).csv")):
        fname = z_path.name
        x_path = axis_dirs["X"] / fname
        y_path = axis_dirs["Y"] / fname
        if x_path.exists() and y_path.exists():
            triplets[fname] = {"X": x_path, "Y": y_path, "Z": z_path}
    return triplets


def build_segments_xyz():
    all_xyz = []
    all_labels = []

    per_label_files = {i: 0 for i in range(len(CONDITION_DIRS))}

    for label_id, cond in enumerate(CONDITION_DIRS):
        triplets = collect_triplets_for_condition(cond)
        if not triplets:
            print(f"⚠ {cond}: 未找到可配对的 XYZ 文件，跳过")
            continue

        for fname, paths in triplets.items():
            sx = read_single_axis_file(paths["X"])
            sy = read_single_axis_file(paths["Y"])
            sz = read_single_axis_file(paths["Z"])

            L = min(sx.size, sy.size, sz.size)
            if L < SEGMENT_LENGTH:
                continue

            sx = sx[:L]
            sy = sy[:L]
            sz = sz[:L]

            start = 0
            seg_idx = 0
            while start + SEGMENT_LENGTH <= L:
                x_seg = sx[start : start + SEGMENT_LENGTH]
                y_seg = sy[start : start + SEGMENT_LENGTH]
                z_seg = sz[start : start + SEGMENT_LENGTH]

                all_xyz.append(np.stack([x_seg, y_seg, z_seg], axis=0))  # (3, 128)
                all_labels.append(label_id)

                start += STRIDE
                seg_idx += 1

            per_label_files[label_id] += 1
            print(f"✅ {cond}/{fname} → 生成 {seg_idx} 个三轴片段（label={label_id}）")

    if not all_xyz:
        raise RuntimeError("未生成任何三轴样本，请检查数据目录与工况/轴配置。")

    signals_xyz = np.asarray(all_xyz, dtype=np.float32)  # (N, 3, 128)
    labels = np.asarray(all_labels, dtype=int)
    return signals_xyz, labels, per_label_files


def main():
    os.makedirs(OUTPUT_NPY.parent, exist_ok=True)
    signals_xyz, labels, per_label_files = build_segments_xyz()

    np.save(OUTPUT_NPY, {"signals_xyz": signals_xyz, "labels": labels})

    print("🎉 三轴切片完成")
    print(f"   输出文件: {OUTPUT_NPY}")
    print(f"   signals_xyz shape: {signals_xyz.shape} (N,3,{SEGMENT_LENGTH})")
    print("=== 每个标签使用的源文件数 ===")
    for lid, cnt in sorted(per_label_files.items()):
        print(f"  label {lid}: {cnt} 个 M(k).csv 文件")


if __name__ == "__main__":
    main()

