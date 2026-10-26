"""
1.2_generate_cwru_gaf_images.py — CWRU v2：2048 点预处理 → GAF 256×256。
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
    CWRU_SEGMENT_LENGTH,
    FILE_RANGES_CWRU_V2,
    GADF_IMAGES_V2,
    GASF_IMAGES_V2,
    GAF_PREPROCESSED_CWRU_V2,
    ensure_v2_dirs,
)

INPUT_NPY = str(GAF_PREPROCESSED_CWRU_V2)
FILE_RANGES_CSV = str(FILE_RANGES_CWRU_V2)
OUT_GASF_DIR = str(GASF_IMAGES_V2)
OUT_GADF_DIR = str(GADF_IMAGES_V2)

GAF_SIZE = CWRU_GAF_SIZE
CMAP = "jet"
MAX_IMAGES_PER_LABEL = 1500
KEEP_MOD = 8
KEEP_PATTERN = {0, 3, 6}
CLEAR_OUTPUT_DIRS = True


def resample_1d(signal: np.ndarray, out_len: int) -> np.ndarray:
    if signal.shape[0] == out_len:
        return signal.astype(np.float32, copy=False)
    x_old = np.linspace(0.0, 1.0, num=signal.shape[0], dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=out_len, dtype=np.float32)
    return np.interp(x_new, x_old, signal).astype(np.float32)


def normalize_to_01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mn = float(x.min())
    mx = float(x.max())
    if mx - mn < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def main() -> None:
    ensure_v2_dirs()
    obj = np.load(INPUT_NPY, allow_pickle=True)
    data = obj.item() if isinstance(obj, np.ndarray) and obj.dtype == object else obj
    signals = np.asarray(data["signals"], dtype=np.float32)
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
    print("[CWRU-GAF v2] Stratified per-file selection:")
    for (lab, fid), cnt in sorted(per_file_counts.items()):
        print(f"  label={lab} file_id={fid}: {cnt}")

    for idx in selected:
        lab = int(labels[idx])
        s = resample_1d(signals[idx], GAF_SIZE)

        gasf = gaf_transform(s, method="s", normalize=False)
        gadf = gaf_transform(s, method="d", normalize=False)
        gasf01 = normalize_to_01(gasf)
        gadf01 = normalize_to_01(gadf)

        gasf_dir = os.path.join(OUT_GASF_DIR, str(lab))
        gadf_dir = os.path.join(OUT_GADF_DIR, str(lab))
        os.makedirs(gasf_dir, exist_ok=True)
        os.makedirs(gadf_dir, exist_ok=True)

        gasf_path = os.path.join(gasf_dir, f"gasf_{lab}_{idx:05d}.png")
        gadf_path = os.path.join(gadf_dir, f"gadf_{lab}_{idx:05d}.png")
        plt.imsave(gasf_path, gasf01, cmap=CMAP, origin="lower")
        plt.imsave(gadf_path, gadf01, cmap=CMAP, origin="lower")
        per_label_counts[lab] = per_label_counts.get(lab, 0) + 1

    print(
        f"[CWRU-GAF v2] Done. segment={CWRU_SEGMENT_LENGTH}, gaf_size={GAF_SIZE}, "
        f"generated={len(selected)} per_label={per_label_counts}"
    )


if __name__ == "__main__":
    main()
