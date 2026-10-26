"""
与 HUST 1.1_preprocess_custom 一致的预处理：去均值 + 全局分位数缩放到 [-1, 1]。
"""
from __future__ import annotations

from typing import Any

import numpy as np


def demean_1d_segments(signals: np.ndarray) -> np.ndarray:
    """signals: (N, L) -> 每段减均值。"""
    if signals.ndim != 2:
        raise ValueError(f"expected (N, L), got {signals.shape}")
    means = signals.mean(axis=1, keepdims=True)
    return (signals - means).astype(np.float32, copy=False)


def robust_scale_1d_minus1_1(
    signals: np.ndarray,
    plow: float = 1.0,
    phigh: float = 99.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """对 (N, L) 全体样本做全局分位数裁剪并映射到 [-1, 1]。"""
    if signals.ndim != 2:
        raise ValueError(f"expected (N, L), got {signals.shape}")
    flat = signals.reshape(-1)
    lo = float(np.percentile(flat, plow))
    hi = float(np.percentile(flat, phigh))
    if hi <= lo:
        lo, hi = float(flat.min()), float(flat.max() + 1e-8)
    clipped = np.clip(signals, lo, hi)
    scaled = (2.0 * (clipped - lo) / (hi - lo + 1e-8) - 1.0).astype(np.float32)
    meta = {
        "method": "global_percentile_clip_scale",
        "plow": float(plow),
        "phigh": float(phigh),
        "lo": lo,
        "hi": hi,
    }
    return scaled, meta


def preprocess_gaf_1d(
    signals: np.ndarray,
    plow: float = 1.0,
    phigh: float = 99.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    demeaned = demean_1d_segments(signals)
    scaled, meta = robust_scale_1d_minus1_1(demeaned, plow=plow, phigh=phigh)
    return scaled, meta
