"""
为 Vibration based Fault Diagnosis of Machines 数据集生成少量结构增强版 GASF / GADF 图像（SA-GAF，小体量）。

- 输入：1.1_preprocess_vibration_machines.py 输出的 gaf_preprocessed_vibration_machines.npy
- 对 GAF 做对比度拉伸 + unsharp mask
- 每个标签最多生成 MAX_IMAGES_PER_LABEL 组增强图像
"""

import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils.gaf_utils import gaf_transform


DEBUG_LOG_PATH = r"d:\TEP_2_repro\gaf_yolo\.cursor\debug.log"

INPUT_NPY = r"D:\TEP_2_repro\gaf_yolo\data\processed\gaf_preprocessed_vibration_machines.npy"
OUTPUT_DIR = r"D:\TEP_2_repro\gaf_yolo\data\images_vibration_machines_enhanced"
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


def normalize_to_01(matrix: np.ndarray) -> np.ndarray:
    mat_min = float(matrix.min())
    mat_max = float(matrix.max())
    if mat_max == mat_min:
        return np.zeros_like(matrix, dtype=np.float32)
    return (matrix - mat_min) / (mat_max - mat_min)


def gaf_to_01(gaf: np.ndarray) -> np.ndarray:
    """GAF 固定 [-1,1] → [0,1] 映射，避免逐图 min-max 拉平差异。"""
    return np.clip((gaf + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)


def local_contrast_stretch(matrix: np.ndarray, low_perc: float = 2.0, high_perc: float = 98.0) -> np.ndarray:
    lo, hi = np.percentile(matrix, [low_perc, high_perc])
    if hi <= lo:
        return normalize_to_01(matrix)
    clipped = np.clip(matrix, lo, hi)
    return (clipped - lo) / (hi - lo + 1e-8)


def conv2d_valid_reflect(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    kernel = kernel.astype(np.float32)
    kh, kw = kernel.shape
    assert kh % 2 == 1 and kw % 2 == 1, "kernel 必须为奇数尺寸"

    pad_h = kh // 2
    pad_w = kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")

    h, w = image.shape
    out = np.zeros_like(image, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            region = padded[i : i + kh, j : j + kw]
            out[i, j] = float(np.sum(region * kernel))
    return out


def enhance_gaf_matrix(matrix: np.ndarray) -> np.ndarray:
    base = normalize_to_01(matrix)
    stretched = local_contrast_stretch(base)
    sharpen_kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )
    sharpened = conv2d_valid_reflect(stretched, sharpen_kernel)
    enhanced = normalize_to_01(sharpened)
    return enhanced


def load_signals_and_labels() -> Tuple[np.ndarray, np.ndarray]:
    agent_log(
        run_id="vib_machines_sa_gaf",
        hypothesis_id="H1",
        location="1.3_generate_gaf_vibration_machines_enhanced_images.py:load",
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

    n = len(signals)
    per_label_counts = {}
    print(f"[VibMachines SA-GAF] 总样本数: {n}")

    for idx in range(n):
        signal = signals[idx]
        label = int(labels[idx])

        cnt = per_label_counts.get(label, 0)
        if cnt >= MAX_IMAGES_PER_LABEL:
            continue
        per_label_counts[label] = cnt + 1

        gasf = gaf_transform(signal, method="s", normalize=False)
        gadf = gaf_transform(signal, method="d", normalize=False)

        # 注意：enhance_gaf_matrix 内部会做对比度拉伸/锐化。为了保留跨类别差异，
        # 我们先做固定映射到 [0,1] 再增强。
        gasf_enh = enhance_gaf_matrix(gaf_to_01(gasf))
        gadf_enh = enhance_gaf_matrix(gaf_to_01(gadf))

        gasf_dir = os.path.join(gasf_root, str(label))
        gadf_dir = os.path.join(gadf_root, str(label))
        os.makedirs(gasf_dir, exist_ok=True)
        os.makedirs(gadf_dir, exist_ok=True)

        gasf_path = os.path.join(gasf_dir, f"gasf_{label}_{idx:05d}.png")
        gadf_path = os.path.join(gadf_dir, f"gadf_{label}_{idx:05d}.png")

        plt.imsave(gasf_path, gasf_enh, cmap="jet")
        plt.imsave(gadf_path, gadf_enh, cmap="jet")

    print("[VibMachines SA-GAF] 小体量增强 GASF/GADF 图像生成完成！")


if __name__ == "__main__":
    main()

