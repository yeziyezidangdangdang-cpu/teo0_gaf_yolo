"""
Vibration based Fault Diagnosis of Machines 数据集预处理：
- 0.0_prepare_vibration_machines_dataset.py 已将各工况 CSV 切成 128 点片段 + 标签
- 本脚本对每个片段做：
  - 中值滤波去趋势 → 残差信号（突出冲击）
  - 全局 [-1, 1] 归一化
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import utils.project_paths as pp  # noqa: E402

import json
import time

import numpy as np
import pandas as pd
from scipy.signal import medfilt


DEBUG_LOG_PATH = r"d:\TEP_2_repro\gaf_yolo\.cursor\debug.log"

RAW_DATA_PATH = r"D:/TEP_2_repro/gaf_yolo/data/processed/final_dataset_vibration_machines.csv"
OUTPUT_PATH = r"D:/TEP_2_repro/gaf_yolo/data/processed/gaf_preprocessed_vibration_machines.npy"

SEGMENT_LENGTH = 128


def agent_log(run_id, hypothesis_id, location, message, data=None):
    try:
        now_ms = int(time.time() * 1000)
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
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def preprocess_signal(signal: np.ndarray) -> np.ndarray:
    trend = medfilt(signal, kernel_size=31)
    residual = signal - trend
    return residual


def main():
    df = pd.read_csv(RAW_DATA_PATH, header=None)
    agent_log(
        run_id="vib_machines",
        hypothesis_id="H1",
        location="1.1_preprocess_vibration_machines.py:main",
        message="Loaded RAW_DATA_PATH",
        data={"shape": list(df.shape)},
    )

    signals_list = []
    labels_list = []

    for _, row in df.iterrows():
        signal = row.iloc[:SEGMENT_LENGTH].values.astype(float)
        label = int(row.iloc[SEGMENT_LENGTH])
        residual = preprocess_signal(signal)
        signals_list.append(residual)
        labels_list.append(label)

    signals = np.array(signals_list, dtype=float)
    labels = np.array(labels_list, dtype=int)

    global_min = float(signals.min())
    global_max = float(signals.max())

    agent_log(
        run_id="vib_machines",
        hypothesis_id="H2",
        location="1.1_preprocess_vibration_machines.py:stats",
        message="Global stats before normalization (residual signals)",
        data={"global_min": global_min, "global_max": global_max},
    )

    if global_max == global_min:
        signals_norm = np.zeros_like(signals)
    else:
        signals_norm = 2 * (signals - global_min) / (global_max - global_min) - 1

    signals = signals_norm

    np.save(OUTPUT_PATH, {"signals": signals, "labels": labels})
    print(f"✅ Vibration Machines 预处理完成！共 {len(signals)} 个样本，已保存到 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

