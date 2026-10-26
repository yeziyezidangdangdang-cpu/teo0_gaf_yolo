"""
1.3_generate_cwru_gaf_enhanced_images.py

为 CWRU 生成结构感知增强版 SA-GAF（在 GAF 矩阵上做对比度拉伸 + unsharp mask）：
- 输入：`data/processed/gaf_preprocessed_cwru.npy`
- 输出：
  - data/images_enhanced/gasf/<label>/gasf_<label>_<idx>.png
  - data/images_enhanced/gadf/<label>/gadf_<label>_<idx>.png

说明：
- 与 `1.3_generate_gaf_enhanced_images.py`（HUST 版本）并存，避免互相覆盖/混淆。
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from utils.cwru_sampling import (  # noqa: E402
    build_idx_to_file,
    files_per_label,
    load_file_ranges,
    select_indices_stratified,
    summarize_selection,
)
from utils.gaf_utils import gaf_transform  # noqa: E402
from utils.project_paths import (  # noqa: E402
    CWRU_GAF_SIZE,
    FILE_RANGES_CWRU_V2,
    GADF_IMAGES_ENH_V2,
    GASF_IMAGES_ENH_V2,
    GAF_PREPROCESSED_CWRU_V2,
    ensure_v2_dirs,
)

INPUT_NPY = str(GAF_PREPROCESSED_CWRU_V2)
FILE_RANGES_CSV = str(FILE_RANGES_CWRU_V2)
OUT_GASF_DIR = str(GASF_IMAGES_ENH_V2)
OUT_GADF_DIR = str(GADF_IMAGES_ENH_V2)

CMAP = "jet"
GAF_SIZE = CWRU_GAF_SIZE
MAX_IMAGES_PER_LABEL = 1500
KEEP_MOD = 8
KEEP_PATTERN = {0, 3, 6}
CLEAR_OUTPUT_DIRS = True

# Contrast stretch percentiles
LOW_P = 2.0
HIGH_P = 98.0


def normalize_to_01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mn = float(x.min())
    mx = float(x.max())
    if mx - mn < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def local_contrast_stretch(x01: np.ndarray, low_p: float = LOW_P, high_p: float = HIGH_P) -> np.ndarray:
    x01 = x01.astype(np.float32, copy=False)
    lo = float(np.percentile(x01, low_p))
    hi = float(np.percentile(x01, high_p))
    if hi - lo < 1e-12:
        return np.clip(x01, 0.0, 1.0)
    y = (x01 - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32, copy=False)


def unsharp_mask(x01: np.ndarray) -> np.ndarray:
    """
    Apply a classic sharpening kernel with reflect padding.
    Uses scipy.ndimage.convolve if available, else a small numpy fallback.
    """
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    try:
        from scipy.ndimage import convolve  # type: ignore

        y = convolve(x01, kernel, mode="reflect")
        return y.astype(np.float32, copy=False)
    except Exception:
        pad = 1
        xpad = np.pad(x01, ((pad, pad), (pad, pad)), mode="reflect")
        out = np.empty_like(x01, dtype=np.float32)
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                region = xpad[i : i + 3, j : j + 3]
                out[i, j] = float(np.sum(region * kernel))
        return out


def enhance_gaf_matrix(gaf: np.ndarray) -> np.ndarray:
    x01 = normalize_to_01(gaf)
    x01 = local_contrast_stretch(x01)
    sharp = unsharp_mask(x01)
    return normalize_to_01(sharp)


def maybe_resample_1d(s: np.ndarray, target: int) -> np.ndarray:
    if s.shape[0] == target:
        return s.astype(np.float32, copy=False)
    x_old = np.linspace(0.0, 1.0, num=s.shape[0], dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=target, dtype=np.float32)
    return np.interp(x_new, x_old, s).astype(np.float32)


def main() -> None:
    ensure_v2_dirs()
    obj = np.load(INPUT_NPY, allow_pickle=True)
    data = obj.item() if isinstance(obj, np.ndarray) and obj.dtype == object else obj
    signals = np.asarray(data["signals"], dtype=np.float32)  # (N, L)
    labels = np.asarray(data["labels"], dtype=int)

    if CLEAR_OUTPUT_DIRS:
        for out_dir in (OUT_GASF_DIR, OUT_GADF_DIR):
            p = Path(out_dir)
            if p.exists():
                shutil.rmtree(p)
    os.makedirs(OUT_GASF_DIR, exist_ok=True)
    os.makedirs(OUT_GADF_DIR, exist_ok=True)

    ranges = load_file_ranges(Path(FILE_RANGES_CSV))
    idx_to_file = build_idx_to_file(ranges)
    files_by_label = files_per_label(ranges, labels)
    selected = select_indices_stratified(
        labels=labels,
        idx_to_file=idx_to_file,
        files_by_label=files_by_label,
        file_ranges=ranges,
        max_per_label=MAX_IMAGES_PER_LABEL,
        keep_mod=KEEP_MOD,
        keep_pattern=KEEP_PATTERN,
    )
    per_label_counts: dict[int, int] = {}
    per_file_counts = summarize_selection(selected, labels, idx_to_file)
    print("[CWRU-SA-GAF v2] Stratified per-file selection:")
    for (lab, fid), cnt in sorted(per_file_counts.items()):
        print(f"  label={lab} file_id={fid}: {cnt}")

    for idx in selected:
        lab = int(labels[idx])

        s = maybe_resample_1d(signals[idx], GAF_SIZE)
        gasf = gaf_transform(s, method="s", normalize=False)
        gadf = gaf_transform(s, method="d", normalize=False)

        gasf_enh = enhance_gaf_matrix(gasf)
        gadf_enh = enhance_gaf_matrix(gadf)

        gasf_dir = os.path.join(OUT_GASF_DIR, str(lab))
        gadf_dir = os.path.join(OUT_GADF_DIR, str(lab))
        os.makedirs(gasf_dir, exist_ok=True)
        os.makedirs(gadf_dir, exist_ok=True)

        gasf_path = os.path.join(gasf_dir, f"gasf_{lab}_{idx:05d}.png")
        gadf_path = os.path.join(gadf_dir, f"gadf_{lab}_{idx:05d}.png")

        plt.imsave(gasf_path, gasf_enh, cmap=CMAP, origin="lower")
        plt.imsave(gadf_path, gadf_enh, cmap=CMAP, origin="lower")
        per_label_counts[lab] = per_label_counts.get(lab, 0) + 1

    print(f"[CWRU-SA-GAF v2] Done. gaf_size={GAF_SIZE}, generated={len(selected)} per_label={per_label_counts}")


if __name__ == "__main__":
    main()

