import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import utils.project_paths as pp  # noqa: E402
import os
import scipy.io
import numpy as np
import pandas as pd

# 设置路径（CSV 输出到 CWRU/csv，与 0.2 读取路径一致）
mat_dir = r'D:/TEP_2_repro/gaf_yolo/data/raw/CWRU/cwrn'
base_dir = os.path.dirname(mat_dir)  # CWRU 目录
csv_dir = os.path.join(base_dir, 'csv')
os.makedirs(csv_dir, exist_ok=True)

def extract_1d_signal(mat_data):
    """从 mat 字典中提取第一个有效的一维数值信号"""
    for key, val in mat_data.items():
        if key.startswith('__'):  # 跳过 MATLAB 元数据
            continue
        if isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.number):
            signal = val.flatten()
            if signal.size > 10:  # 至少10个点才算有效
                return signal
    return None

# 遍历 .mat 文件
for filename in os.listdir(mat_dir):
    if not filename.endswith('.mat'):
        continue

    mat_path = os.path.join(mat_dir, filename)
    csv_filename = os.path.splitext(filename)[0] + '.csv'
    csv_path = os.path.join(csv_dir, csv_filename)

    try:
        # 加载 .mat
        mat_data = scipy.io.loadmat(mat_path)
        signal = extract_1d_signal(mat_data)

        if signal is None:
            print(f"[WARN] Skip {filename}: no valid 1D signal found")
            continue

        # 转为 DataFrame（单列，无 header，无 index）
        df = pd.DataFrame(signal)
        df.to_csv(csv_path, index=False, header=False)

        print(f"[OK] Wrote {csv_filename} ({len(signal)} points)")

    except Exception as e:
        print(f"[ERR] Failed {filename}: {e}")

print(f"\n[OK] Done. CSV saved to:\n{csv_dir}")