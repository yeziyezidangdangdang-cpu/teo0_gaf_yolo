# src/2_preprocess_custom.py
"""
STFT 频谱图前置预处理（HUST-bearing）：
- 从 0.0_prepare_hust_dataset.py 输出的 final_dataset_hust.csv 读取 3 轴 2048 点片段
- 对每段的每个轴做去均值 + 标准化，为后续三通道 STFT 频谱图生成做准备
"""

import sys
from pathlib import Path

import numpy as np
import os

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import utils.project_paths as pp  # noqa: E402

try:
    import pandas as pd  # optional
except Exception:  # pragma: no cover
    pd = None

RAW_DATA_PATH = str(pp.FINAL_DATASET_HUST)
STFT_OUTPUT_PATH = str(pp.STFT_INPUT_HUST)
GAF_OUTPUT_PATH = str(pp.GAF_PREPROCESSED_HUST)
GAF_AXES_OUTPUT_PATH = str(pp.GAF_INPUT_HUST_AXES)
GAF_ENV_AXES_OUTPUT_PATH = str(pp.GAF_INPUT_HUST_AXES_ENV)

SEGMENT_LENGTH = 2048


REMOVE_MEAN_FOR_GAF = True
# 全局鲁棒缩放：按每个轴的 p_low~p_high 分位裁剪并映射到 [-1, 1]
ROBUST_PLOW = 1.0
ROBUST_PHIGH = 99.0
ALSO_SAVE_ENVELOPE_INPUT = True  # 需要 scipy.signal.hilbert


def _load_csv_signals_labels(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    读取 final_dataset_hust.csv，返回：
    - signals: (N, 3, L)
    - labels:  (N,)
    优先用 pandas；若环境缺 pandas，则回退为逐行读取。
    """
    expected_cols = SEGMENT_LENGTH * 3 + 1

    if pd is not None:
        df = pd.read_csv(path, header=None)
        n_cols = df.shape[1]
        if n_cols != expected_cols:
            raise ValueError(f"期望 {expected_cols} 列 (3 轴 × {SEGMENT_LENGTH} + label)，实际为 {n_cols}")
        flat = df.iloc[:, : n_cols - 1].to_numpy(dtype=np.float32)  # (N, 3*L)
        labels = df.iloc[:, n_cols - 1].to_numpy(dtype=int)
        N = flat.shape[0]
        signals = flat.reshape(N, 3, SEGMENT_LENGTH)
        return signals, labels

    # fallback: no pandas
    rows = []
    labels = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            arr = np.fromstring(line, sep=",", dtype=np.float32)
            if arr.size != expected_cols:
                raise ValueError(f"第 {i} 行列数异常：期望 {expected_cols}，实际 {arr.size}")
            rows.append(arr[:-1])
            labels.append(int(round(float(arr[-1]))))
            if (i + 1) % 2000 == 0:
                print(f"[1.1] 已读取 {i + 1} 行…")

    flat = np.stack(rows, axis=0) if rows else np.zeros((0, expected_cols - 1), dtype=np.float32)
    labels = np.asarray(labels, dtype=int)
    N = flat.shape[0]
    signals = flat.reshape(N, 3, SEGMENT_LENGTH)
    return signals, labels


def _robust_scale_to_minus1_1(signals: np.ndarray, plow: float, phigh: float) -> tuple[np.ndarray, dict]:
    """
    对 signals (N, 3, L) 做全局鲁棒缩放到 [-1, 1]。
    每个通道独立计算分位数并裁剪，以保留跨样本/跨类别的幅值差异。
    """
    if signals.ndim != 3 or signals.shape[1] != 3:
        raise ValueError(f"signals shape {signals.shape} 不符合预期 (N, 3, L)")

    scaled = np.empty_like(signals, dtype=np.float32)
    lo_list = []
    hi_list = []
    for ch in range(3):
        flat = signals[:, ch, :].reshape(-1)
        lo = float(np.percentile(flat, plow))
        hi = float(np.percentile(flat, phigh))
        if hi <= lo:
            lo, hi = float(flat.min()), float(flat.max() + 1e-8)
        lo_list.append(lo)
        hi_list.append(hi)

        x = np.clip(signals[:, ch, :], lo, hi)
        x01 = (x - lo) / (hi - lo + 1e-8)
        scaled[:, ch, :] = (x01 * 2.0 - 1.0).astype(np.float32)

    meta = {
        "method": "global_percentile_clip_scale",
        "plow": float(plow),
        "phigh": float(phigh),
        "lo": lo_list,
        "hi": hi_list,
    }
    return scaled, meta


def main():
    # 读取 (N, 3, L) 原始切片
    signals, labels = _load_csv_signals_labels(RAW_DATA_PATH)
    N = signals.shape[0]

    # 每段每轴：去均值 + 标准化（供 STFT 使用）
    means = signals.mean(axis=2, keepdims=True)
    stds = signals.std(axis=2, keepdims=True) + 1e-8
    signals_norm = (signals - means) / stds

    # === 分支 1：三通道 STFT 输入 ===
    os.makedirs(os.path.dirname(STFT_OUTPUT_PATH), exist_ok=True)
    np.save(STFT_OUTPUT_PATH, {"signals": signals_norm, "labels": labels})

    # === 分支 2：GAF 输入（将三轴合成为单通道幅值信号） ===
    # 使用三轴归一化后信号的幅值 sqrt(x^2 + y^2 + z^2) 作为 1D 序列
    magnitudes = np.linalg.norm(signals_norm, axis=1)  # (N, L)
    np.save(GAF_OUTPUT_PATH, {"signals": magnitudes.astype(np.float32), "labels": labels})

    # === 分支 3：三通道 GAF 输入（保留跨样本幅值差异） ===
    sig_gaf = signals.astype(np.float32)
    if REMOVE_MEAN_FOR_GAF:
        sig_gaf = sig_gaf - sig_gaf.mean(axis=2, keepdims=True)

    sig_gaf_scaled, meta = _robust_scale_to_minus1_1(sig_gaf, ROBUST_PLOW, ROBUST_PHIGH)  # (N, 3, L) in [-1, 1]
    np.save(GAF_AXES_OUTPUT_PATH, {"signals": sig_gaf_scaled, "labels": labels, "scale": meta})

    # 可选：包络输入（Hilbert envelope）
    if ALSO_SAVE_ENVELOPE_INPUT:
        try:
            from scipy.signal import hilbert

            env = np.abs(hilbert(sig_gaf, axis=2)).astype(np.float32)  # (N, 3, L), >=0
            env_scaled, meta_env = _robust_scale_to_minus1_1(env, ROBUST_PLOW, ROBUST_PHIGH)
            np.save(GAF_ENV_AXES_OUTPUT_PATH, {"signals": env_scaled, "labels": labels, "scale": meta_env})
            env_msg = f"[OK] Saved HUST GAF-Envelope axes input: {GAF_ENV_AXES_OUTPUT_PATH}"
        except Exception as e:
            env_msg = f"[WARN] Envelope input not saved (needs scipy.signal.hilbert): {repr(e)}"
    else:
        env_msg = "(Envelope input generation disabled)"

    print(f"[OK] HUST STFT input saved. N={N}, STFT signals shape={signals_norm.shape}")
    print(f"[OK] (compat) HUST magnitude GAF input saved. shape={magnitudes.shape}, path={GAF_OUTPUT_PATH}")
    print(f"[OK] HUST axes GAF input saved (global robust scaling). path={GAF_AXES_OUTPUT_PATH}")
    print(env_msg)


if __name__ == "__main__":
    main()