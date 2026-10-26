"""
MCC5-THU 三轴 GAF 预处理：去均值 + 每通道全局分位数缩放到 [-1, 1]（与 HUST 1.1_preprocess_custom 一致）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.project_paths import (  # noqa: E402
    FINAL_DATASET_MCC5,
    GAF_INPUT_MCC5_AXES,
    MCC5_ROBUST_PHIGH,
    MCC5_ROBUST_PLOW,
    MCC5_SEGMENT_LENGTH,
    ensure_mcc5_dirs,
)

try:
    import pandas as pd
except ImportError as e:  # pragma: no cover
    raise ImportError("需要 pandas") from e


def load_csv_signals_labels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    expected_cols = MCC5_SEGMENT_LENGTH * 3 + 1
    df = pd.read_csv(path, header=None)
    if df.shape[1] != expected_cols:
        raise ValueError(f"期望 {expected_cols} 列，实际 {df.shape[1]}")
    flat = df.iloc[:, :-1].to_numpy(dtype=np.float32)
    labels = df.iloc[:, -1].to_numpy(dtype=int)
    signals = flat.reshape(len(df), 3, MCC5_SEGMENT_LENGTH)
    return signals, labels


def robust_scale_axes(signals: np.ndarray, plow: float, phigh: float) -> tuple[np.ndarray, dict]:
    scaled = np.empty_like(signals, dtype=np.float32)
    lo_list, hi_list = [], []
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
        "method": "global_percentile_clip_scale_per_channel",
        "plow": float(plow),
        "phigh": float(phigh),
        "lo": lo_list,
        "hi": hi_list,
    }
    return scaled, meta


def main() -> None:
    ensure_mcc5_dirs()
    signals, labels = load_csv_signals_labels(FINAL_DATASET_MCC5)
    sig = signals.astype(np.float32)
    sig = sig - sig.mean(axis=2, keepdims=True)
    scaled, meta = robust_scale_axes(sig, MCC5_ROBUST_PLOW, MCC5_ROBUST_PHIGH)

    np.save(
        GAF_INPUT_MCC5_AXES,
        {"signals": scaled, "labels": labels, "scale": meta},
        allow_pickle=True,
    )
    sidecar = GAF_INPUT_MCC5_AXES.with_suffix(".meta.json")
    sidecar.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[OK] MCC5 GAF axes input: {GAF_INPUT_MCC5_AXES}")
    print(f"     shape={scaled.shape}, labels={labels.shape}")


if __name__ == "__main__":
    main()
