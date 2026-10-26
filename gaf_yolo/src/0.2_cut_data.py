import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.project_paths import (  # noqa: E402
    CWRU_CSV_DIR,
    CWRU_SEGMENT_LENGTH,
    CWRU_STRIDE,
    FILE_RANGES_CWRU_V2,
    MERGED_SEGMENTS_V2,
    ensure_v2_dirs,
)

SEGMENT_LENGTH = CWRU_SEGMENT_LENGTH
STRIDE = CWRU_STRIDE
CSV_DIR = str(CWRU_CSV_DIR)
OUTPUT_CSV = str(MERGED_SEGMENTS_V2)
FILE_RANGES_CSV = str(FILE_RANGES_CWRU_V2)


def main() -> None:
    ensure_v2_dirs()
    all_segments = []
    file_ranges = []

    for filename in os.listdir(CSV_DIR):
        if not filename.endswith(".csv"):
            continue
        file_path = os.path.join(CSV_DIR, filename)
        try:
            signal = pd.read_csv(file_path, header=None).iloc[:, 0].values
        except Exception as e:
            print(f"[WARN] Read failed: {filename} - {e}")
            continue

        if len(signal) < SEGMENT_LENGTH:
            print(f"[WARN] Skip {filename}: length({len(signal)}) < {SEGMENT_LENGTH}")
            continue

        start_idx = len(all_segments)
        segments = []
        start = 0
        seg_id = 0
        while start + SEGMENT_LENGTH <= len(signal):
            segment = signal[start : start + SEGMENT_LENGTH]
            segments.append([filename, seg_id] + segment.tolist())
            start += STRIDE
            seg_id += 1
        all_segments.extend(segments)
        end_idx = len(all_segments)
        file_id = len(file_ranges)
        file_ranges.append((file_id, start_idx, end_idx))
        print(f"[OK] {filename} -> {seg_id} segments (global idx [{start_idx},{end_idx}))")

    columns = ["filename", "segment_id"] + [f"t_{i}" for i in range(SEGMENT_LENGTH)]
    df = pd.DataFrame(all_segments, columns=columns)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[OK] Merged done: {len(df)} rows x {len(df.columns)} cols")
    print(f"[OK] Output: {OUTPUT_CSV}")

    if file_ranges:
        ranges_df = pd.DataFrame(file_ranges, columns=["file_id", "start_idx", "end_idx"])
        ranges_df.to_csv(FILE_RANGES_CSV, index=False)
        print(f"[OK] File ranges: {FILE_RANGES_CSV} ({len(file_ranges)} files)")


if __name__ == "__main__":
    main()
