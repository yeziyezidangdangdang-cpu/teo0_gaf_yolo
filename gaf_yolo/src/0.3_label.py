import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.project_paths import (  # noqa: E402
    FINAL_DATASET_CWRU_V2,
    MERGED_SEGMENTS_V2,
    ensure_v2_dirs,
)

input_csv = str(MERGED_SEGMENTS_V2)
output_csv = str(FINAL_DATASET_CWRU_V2)


def map_fault_to_label(filename: str) -> int:
    name = filename
    if "." in name:
        name = name.split(".", 1)[0]
    token = name.split("_", 1)[0]

    if token.startswith("N"):
        return 0
    if token.startswith("IR"):
        return 1
    if token.startswith("OR"):
        return 2
    if token.startswith("B"):
        return 3
    raise ValueError(f"无法从文件名映射标签: {filename}")


def main():
    ensure_v2_dirs()
    df = pd.read_csv(input_csv)
    labels = df["filename"].apply(map_fault_to_label)
    time_columns = df.columns[2:]
    df_final = df[time_columns].copy()
    df_final["label"] = labels.values
    df_final.to_csv(output_csv, index=False, header=False)

    print("[OK] Label mapping done (CWRU v2).")
    print("  - map: Normal->0, IR->1, OR->2, B->3")
    print(f"  - cols: {df_final.shape[1]} (signal + label)")
    print(f"  - rows: {df_final.shape[0]}")
    print(f"  - out:  {output_csv}")


if __name__ == "__main__":
    main()
