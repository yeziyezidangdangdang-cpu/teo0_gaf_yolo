import os
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.signal import detrend


# === 配置区 ===
# 0.0_prepare_subf_dataset.py 输出
RAW_DATA_PATH = Path(r"D:\TEP_2_repro\gaf_yolo\data\processed\final_dataset_subf.csv")
# AREEMD 预处理后的输出（仍为 CSV，供 1.1 使用）
OUTPUT_CSV = Path(r"D:\TEP_2_repro\gaf_yolo\data\processed\final_dataset_subf_areemd.csv")

# 相对能量阈值（论文中使用 15%）
REL_ENERGY_THRESH = 0.15


def require_pyemd():
    try:
        from PyEMD import EMD  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "需要安装 PyEMD 才能运行 AREEMD 预处理。\n"
            "请先执行：pip install EMD-signal"
        ) from e


def areemd_reconstruct(signal: np.ndarray, rel_thresh: float = REL_ENERGY_THRESH) -> np.ndarray:
    """
    使用 EMD + 相对能量阈值近似实现 AREEMD：
    - 对信号做轻度去趋势
    - EMD 分解得到 IMFs
    - 计算每个 IMF 的能量占比，保留相对能量 ≥ rel_thresh 的 IMFs
    - 将保留的 IMFs 累加作为重构信号；若全部被过滤，则退化为原始信号
    """
    from PyEMD import EMD

    # 简单去趋势，减小直流分量的影响
    sig = detrend(signal.astype(float))

    emd = EMD()
    imfs = emd(sig)

    if imfs.size == 0:
        return sig

    energies = np.sum(imfs**2, axis=1)
    total_energy = float(np.sum(energies))
    if total_energy <= 0:
        return sig

    rel_energies = energies / total_energy
    mask = rel_energies >= rel_thresh

    if not np.any(mask):
        # 若所有 IMF 相对能量都很低，则退化为“保留全部 IMF”
        recon = np.sum(imfs, axis=0)
    else:
        recon = np.sum(imfs[mask, :], axis=0)

    return recon.astype(float)


def main():
    require_pyemd()

    df = pd.read_csv(RAW_DATA_PATH, header=None)

    # 假设每行: t_0..t_1023, label
    n_cols = df.shape[1]
    if n_cols < 2:
        raise ValueError(f"RAW_DATA_PATH 列数过少: {n_cols}")

    signal_cols = list(range(n_cols - 1))
    label_col = n_cols - 1

    signals = df.iloc[:, signal_cols].to_numpy(dtype=float)
    labels = df.iloc[:, label_col].to_numpy(dtype=int)

    recon_list = []
    for idx, sig in enumerate(signals):
        recon = areemd_reconstruct(sig, rel_thresh=REL_ENERGY_THRESH)
        recon_list.append(recon)
        if (idx + 1) % 200 == 0:
            print(f"✅ AREEMD 预处理 {idx + 1} / {len(signals)} 段")

    recon_arr = np.vstack(recon_list)
    df_out = pd.DataFrame(recon_arr)
    df_out["label"] = labels

    os.makedirs(OUTPUT_CSV.parent, exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False, header=False)

    print("🎉 AREEMD 预处理完成")
    print(f"   输入: {RAW_DATA_PATH}")
    print(f"   输出: {OUTPUT_CSV}")
    print(f"   样本数: {len(df_out)}，每样本长度: {recon_arr.shape[1]}")


if __name__ == "__main__":
    main()

