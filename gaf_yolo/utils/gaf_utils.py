# GAF transformation functions (GASF/GADF)
# utils/gaf_utils.py
import numpy as np


def gaf_transform(series, method="s", normalize: bool = True):
    """
    将一维时间序列转换为 Gramian Angular Field (GAF) 图像。
    论文方法复现：先归一化到 [-1, 1]，再映射到极坐标。

    Args:
        series: 一维 numpy 数组 (shape: [L])
        method: 's' for GASF (Summation), 'd' for GADF (Difference)
        normalize: 是否对输入序列做 min-max 归一化到 [-1, 1]。
                   - True（默认）：每条序列各自归一化（传统做法）
                   - False：认为输入已在 [-1, 1]（例如你在预处理阶段做了“全局”归一化），
                            则跳过此步骤以保留跨样本/跨类别的幅值差异。

    Returns:
        gaf_matrix: 二维 GAF 图像 (shape: [L, L])
    """
    series = np.asarray(series, dtype=float)

    # Step 1: 归一化到 [-1, 1] (论文公式1)
    if normalize:
        min_val, max_val = np.min(series), np.max(series)
        if max_val - min_val == 0:
            normalized = np.zeros_like(series)
        else:
            normalized = (2 * series - max_val - min_val) / (max_val - min_val)
    else:
        normalized = series

    # Step 2: 映射到极坐标 (论文公式2)
    # φ = arccos(x̃_t), where x̃_t ∈ [-1, 1]
    phi = np.arccos(np.clip(normalized, -1, 1))  # θ in paper

    # Step 3: 构建 GAF 矩阵 (论文公式4)
    if method == 's':  # GASF: cos(φ_i + φ_j)
        gaf = np.cos(phi[:, None] + phi[None, :])
    elif method == 'd':  # GADF: sin(φ_i - φ_j)
        gaf = np.sin(phi[:, None] - phi[None, :])
    else:
        raise ValueError("method must be 's' (GASF) or 'd' (GADF)")

    return gaf