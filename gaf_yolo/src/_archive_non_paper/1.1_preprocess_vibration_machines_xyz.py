import numpy as np
from scipy.signal import medfilt


INPUT_NPY = r"D:\TEP_2_repro\gaf_yolo\data\processed\final_dataset_vibration_machines_xyz.npy"
OUTPUT_NPY = r"D:\TEP_2_repro\gaf_yolo\data\processed\gaf_preprocessed_vibration_machines_xyz.npy"

SEGMENT_LENGTH = 128


def preprocess_signal_1d(signal: np.ndarray) -> np.ndarray:
    trend = medfilt(signal, kernel_size=31)
    return signal - trend


def main():
    npy_obj = np.load(INPUT_NPY, allow_pickle=True)
    data = npy_obj.item() if isinstance(npy_obj, np.ndarray) else npy_obj

    signals_xyz = np.asarray(data["signals_xyz"], dtype=np.float32)  # (N,3,L)
    labels = np.asarray(data["labels"], dtype=int)

    if signals_xyz.ndim != 3 or signals_xyz.shape[1] != 3 or signals_xyz.shape[2] != SEGMENT_LENGTH:
        raise ValueError(f"signals_xyz shape {signals_xyz.shape} 不符合预期 (N,3,{SEGMENT_LENGTH})")

    # 逐轴做残差
    residual_xyz = np.empty_like(signals_xyz, dtype=np.float32)
    for a in range(3):
        # 向量化处理每条样本
        for i in range(signals_xyz.shape[0]):
            residual_xyz[i, a] = preprocess_signal_1d(signals_xyz[i, a])

    # 全局 min/max（跨轴跨样本），保留能量/幅值差异
    gmin = float(residual_xyz.min())
    gmax = float(residual_xyz.max())
    if gmax == gmin:
        signals_norm = np.zeros_like(residual_xyz, dtype=np.float32)
    else:
        signals_norm = 2 * (residual_xyz - gmin) / (gmax - gmin) - 1

    np.save(OUTPUT_NPY, {"signals_xyz": signals_norm.astype(np.float32), "labels": labels})
    print(f"✅ 三轴预处理完成！N={signals_norm.shape[0]}，已保存到 {OUTPUT_NPY}")


if __name__ == "__main__":
    main()

