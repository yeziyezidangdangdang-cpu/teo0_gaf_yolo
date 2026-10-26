"""
从 SUBF 预处理信号中提取简化版 cAR 特征（cepstral autoregressive），用于小体量数值验证。

流程（对应论文的 AREEMD + cAR 思路的后半部分）：
- 输入：1.1_preprocess_custom.py 生成的 gaf_preprocessed_subf.npy（signals, labels）
- 对每段预处理后的时间序列：
  - 计算实值倒谱：cep = IFFT(log |FFT(x)|)
  - 在倒谱上拟合 5 阶 AR 模型（Yule–Walker 近似）
  - 将 5 个 AR 系数作为 cAR 特征向量
- 为节约时间：每个类别最多采样 MAX_SAMPLES_PER_CLASS 段。
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import utils.project_paths as pp  # noqa: E402

import os
from pathlib import Path

import numpy as np


INPUT_NPY = Path(r"D:/TEP_2_repro/gaf_yolo/data/processed/gaf_preprocessed_subf.npy")
OUTPUT_NPZ = Path(r"D:/TEP_2_repro/gaf_yolo/data/processed/car_features_subf.npz")

AR_ORDER = 5
MAX_SAMPLES_PER_CLASS = 500  # 每类最多采样 500 段用于快速实验


def compute_cepstrum(x: np.ndarray) -> np.ndarray:
    """计算实值倒谱：cep = IFFT(log |FFT(x)|)。"""
    eps = 1e-12
    X = np.fft.rfft(x)
    log_mag = np.log(np.abs(X) + eps)
    cep = np.fft.irfft(log_mag, n=x.size)
    return np.real(cep)


def yule_walker_ar(x: np.ndarray, order: int) -> np.ndarray:
    """
    使用 Yule–Walker 方程估计 AR(order) 系数。
    返回长度为 order 的系数向量 phi。
    """
    x = x - np.mean(x)
    N = x.size
    if N <= order:
        # 样本太短时退化为零系数
        return np.zeros(order, dtype=float)

    r = np.correlate(x, x, mode="full")
    mid = len(r) // 2
    r = r[mid : mid + order + 1] / N  # r[0..order]

    R = np.empty((order, order), dtype=float)
    for i in range(order):
        R[i] = r[abs(i - np.arange(order))]

    rhs = r[1 : order + 1]
    try:
        phi = np.linalg.solve(R, rhs)
    except np.linalg.LinAlgError:
        phi = np.zeros(order, dtype=float)
    return phi


def main():
    if not INPUT_NPY.exists():
        raise FileNotFoundError(f"未找到预处理 NPY: {INPUT_NPY}")

    npy_obj = np.load(INPUT_NPY, allow_pickle=True)
    data = None
    if isinstance(npy_obj, dict):
        data = npy_obj
    else:
        # np.save(dict) 的情形
        data = npy_obj.item()

    signals = np.asarray(data["signals"], dtype=float)
    labels = np.asarray(data["labels"], dtype=int)

    N, L = signals.shape
    print(f"[3.1] 载入 signals 形状: {signals.shape}, 标签数: {len(labels)}")

    # 为每类下采样，避免特征数过大
    rng = np.random.default_rng(42)
    unique_labels = np.unique(labels)

    selected_indices = []
    for lab in unique_labels:
        idxs = np.where(labels == lab)[0]
        if idxs.size > MAX_SAMPLES_PER_CLASS:
            idxs = rng.choice(idxs, size=MAX_SAMPLES_PER_CLASS, replace=False)
        selected_indices.extend(idxs.tolist())

    selected_indices = np.array(sorted(selected_indices))
    signals_sel = signals[selected_indices]
    labels_sel = labels[selected_indices]
    print(
        f"[3.1] 采样后样本数: {signals_sel.shape[0]} "
        f"(每类最多 {MAX_SAMPLES_PER_CLASS} 条)"
    )

    feats = np.zeros((signals_sel.shape[0], AR_ORDER), dtype=float)

    for i, sig in enumerate(signals_sel):
        cep = compute_cepstrum(sig)
        phi = yule_walker_ar(cep, AR_ORDER)
        feats[i] = phi

        if (i + 1) % 500 == 0:
            print(f"[3.1] 已提取 {i + 1} / {signals_sel.shape[0]} 条样本的 cAR 特征")

    os.makedirs(OUTPUT_NPZ.parent, exist_ok=True)
    np.savez_compressed(
        OUTPUT_NPZ,
        features=feats,
        labels=labels_sel,
        ar_order=AR_ORDER,
        max_samples_per_class=MAX_SAMPLES_PER_CLASS,
    )

    print(f"[3.1] 已保存 cAR 特征到: {OUTPUT_NPZ}")
    print(f"      特征形状: {feats.shape}")


if __name__ == "__main__":
    main()

