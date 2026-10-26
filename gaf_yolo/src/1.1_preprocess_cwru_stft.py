"""
1.1_preprocess_cwru_stft.py

为 CWRU STFT 频谱图生成做前置预处理：
- 从 0.0_prepare_cwru_stft.py 输出的 final_dataset_cwru.csv 读取 1024 点片段
- 对每段做去均值 + 标准差归一化，输出为 numpy 字典：
  {"signals": (N, L), "labels": (N,)}

说明：
- 与 HUST 的 1.1_preprocess_custom.py 平行存在，不影响现有 HUST 管线。
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import utils.project_paths as pp  # noqa: E402

from __future__ import annotations

import os

import numpy as np
import pandas as pd


RAW_DATA_PATH = str(pp.FINAL_DATASET_CWRU)
OUTPUT_PATH = str(pp.PROCESSED / 'stft_input_cwru.npy')

SEGMENT_LENGTH = 1024


def main() -> None:
    df = pd.read_csv(RAW_DATA_PATH, header=None)
    n_cols = df.shape[1]
    expected = SEGMENT_LENGTH + 1
    if n_cols != expected:
        raise ValueError(f"期望 {expected} 列 (信号 {SEGMENT_LENGTH} + label)，实际为 {n_cols}")

    signal_cols = list(range(SEGMENT_LENGTH))
    label_col = SEGMENT_LENGTH

    signals = df.iloc[:, signal_cols].to_numpy(dtype=np.float32)  # (N,L)
    labels = df.iloc[:, label_col].to_numpy(dtype=int)

    # 每段：去均值 + 标准化
    means = signals.mean(axis=1, keepdims=True)
    stds = signals.std(axis=1, keepdims=True) + 1e-8
    signals_norm = (signals - means) / stds

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.save(OUTPUT_PATH, {"signals": signals_norm, "labels": labels})
    print(f"[CWRU-STFT] 预处理完成：N={signals_norm.shape[0]}，signals 形状={signals_norm.shape}")


if __name__ == "__main__":
    main()

