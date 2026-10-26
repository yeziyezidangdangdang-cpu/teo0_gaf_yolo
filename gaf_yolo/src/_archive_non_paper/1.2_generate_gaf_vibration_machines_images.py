"""
为 Vibration based Fault Diagnosis of Machines 数据集生成少量彩色 GAF 图像用于快速可视化验证。

- 输入：1.1_preprocess_vibration_machines.py 输出的 gaf_preprocessed_vibration_machines.npy
- 每个标签最多生成 MAX_IMAGES_PER_LABEL 组 GASF/GADF 图像（小体量）
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from utils.gaf_utils import gaf_transform


DEBUG_LOG_PATH = r"d:\TEP_2_repro\gaf_yolo\.cursor\debug.log"

INPUT_NPY = r"D:\TEP_2_repro\gaf_yolo\data\processed\gaf_preprocessed_vibration_machines.npy"
OUTPUT_DIR = r"D:\TEP_2_repro\gaf_yolo\data\images_vibration_machines"
SIGNAL_LENGTH = 128

MAX_IMAGES_PER_LABEL = 50


def agent_log(run_id, hypothesis_id, location, message, data=None):
    try:
        now_ms = int(__import__("time").time() * 1000)
        payload = {
            "id": f"log_{now_ms}",
            "timestamp": now_ms,
            "location": location,
            "message": message,
            "data": data or {},
            "runId": run_id,
            "hypothesisId": hypothesis_id,
        }
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            import json

            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def gaf_to_01(gaf: np.ndarray) -> np.ndarray:
    """
    GAF 理论范围为 [-1, 1]，使用固定映射到 [0, 1]，避免“每张图单独 min-max”
    把不同类别间的差异拉平。
    """
    return np.clip((gaf + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)


def load_signals_and_labels():
    agent_log(
        run_id="vib_machines_gaf",
        hypothesis_id="H1",
        location="1.2_generate_gaf_vibration_machines_images.py:load",
        message="About to read INPUT_NPY",
        data={"input_path": INPUT_NPY, "exists": os.path.exists(INPUT_NPY)},
    )

    npy_obj = np.load(INPUT_NPY, allow_pickle=True)

    data = None
    try:
        if isinstance(npy_obj, np.ndarray) and npy_obj.dtype == object and npy_obj.shape == ():
            data = npy_obj.item()
        elif isinstance(npy_obj, dict):
            data = npy_obj
        elif isinstance(npy_obj, np.lib.npyio.NpzFile):
            data = {"signals": npy_obj["signals"], "labels": npy_obj["labels"]}
        else:
            data = npy_obj.item()
    except Exception:
        pass

    signals = np.asarray(data["signals"], dtype=np.float32)
    labels = np.asarray(data["labels"], dtype=int)

    if signals.ndim != 2 or signals.shape[1] != SIGNAL_LENGTH:
        raise ValueError(f"signals shape {signals.shape} 不符合预期 (N, {SIGNAL_LENGTH})")
    if labels.ndim != 1 or labels.shape[0] != signals.shape[0]:
        raise ValueError(
            f"labels shape {labels.shape} 与 signals.shape[0]={signals.shape[0]} 不匹配"
        )

    return signals, labels


def main():
    signals, labels = load_signals_and_labels()

    gasf_root = os.path.join(OUTPUT_DIR, "gasf")
    gadf_root = os.path.join(OUTPUT_DIR, "gadf")
    os.makedirs(gasf_root, exist_ok=True)
    os.makedirs(gadf_root, exist_ok=True)

    per_label_counts = {}
    n = len(signals)
    print(f"[VibMachines GAF] 总样本数: {n}")

    for idx in range(n):
        signal = signals[idx]
        label = int(labels[idx])

        cnt = per_label_counts.get(label, 0)
        if cnt >= MAX_IMAGES_PER_LABEL:
            continue
        per_label_counts[label] = cnt + 1

        # 预处理阶段已做全局 [-1,1] 归一化，这里跳过 gaf_transform 内部的逐样本归一化
        gasf = gaf_transform(signal, method="s", normalize=False)
        gadf = gaf_transform(signal, method="d", normalize=False)

        gasf_norm = gaf_to_01(gasf)
        gadf_norm = gaf_to_01(gadf)

        gasf_dir = os.path.join(gasf_root, str(label))
        gadf_dir = os.path.join(gadf_root, str(label))
        os.makedirs(gasf_dir, exist_ok=True)
        os.makedirs(gadf_dir, exist_ok=True)

        gasf_path = os.path.join(gasf_dir, f"gasf_{label}_{idx:05d}.png")
        gadf_path = os.path.join(gadf_dir, f"gadf_{label}_{idx:05d}.png")

        plt.imsave(gasf_path, gasf_norm, cmap="jet")
        plt.imsave(gadf_path, gadf_norm, cmap="jet")

    print("[VibMachines GAF] 小体量 GASF/GADF 图像生成完成！")


if __name__ == "__main__":
    main()

