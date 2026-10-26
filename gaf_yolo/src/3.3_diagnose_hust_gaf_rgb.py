from __future__ import annotations

"""
诊断 HUST 三通道 GAF-RGB 结果为何颜色偏“绿紫”且区分弱。

做法：
- 读取 1.1 输出的 gaf_input_hust_axes.npy（signals 已全局鲁棒缩放到 [-1,1]）
- 按 label 统计：
  - 三轴信号的均值/标准差/分位数（看是否已被压扁）
  - 生成少量 GASF/GADF（每轴各一张）后，统计矩阵像素分布与通道相关性
- 同时读取已生成的 PNG（可选）统计 RGB 像素分布

输出：只打印诊断信息，不修改数据。
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import utils.project_paths as pp  # noqa: E402

import os
from pathlib import Path
import random

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_AXES_NPY = str(pp.GAF_INPUT_HUST_AXES)
GAF_RGB_DIR = str(pp.GADF_RGB_HUST)  # 任选 gasf/gadf 都可

SEGMENT_LENGTH = 2048
GAF_SIZE = 256
SAMPLES_PER_LABEL = 5
RANDOM_SEED = 42


def load_npy_dict(path: str) -> dict:
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        return obj.item()
    if isinstance(obj, dict):
        return obj
    return obj.item()


def resample_1d(signal: np.ndarray, out_len: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    in_len = signal.shape[0]
    if in_len == out_len:
        return signal
    x_old = np.arange(in_len, dtype=np.float32)
    x_new = np.linspace(0, in_len - 1, out_len, dtype=np.float32)
    return np.interp(x_new, x_old, signal).astype(np.float32)


def pct_stats(x: np.ndarray, p=(1, 5, 50, 95, 99)) -> dict:
    x = np.asarray(x, dtype=np.float32)
    out = {f"p{pp}": float(np.percentile(x, pp)) for pp in p}
    out["mean"] = float(np.mean(x))
    out["std"] = float(np.std(x))
    out["min"] = float(np.min(x))
    out["max"] = float(np.max(x))
    return out


def corr3(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-8)
    return float(np.sum(a * b) / denom)


def main() -> None:
    from utils.gaf_utils import gaf_transform  # local import to ensure path

    rng = random.Random(RANDOM_SEED)

    data = load_npy_dict(INPUT_AXES_NPY)
    sig = np.asarray(data["signals"], dtype=np.float32)  # (N,3,L) in [-1,1]
    lab = np.asarray(data["labels"], dtype=int)
    assert sig.ndim == 3 and sig.shape[1] == 3

    labels = sorted(set(int(x) for x in lab.tolist()))
    print(f"[DIAG] input={INPUT_AXES_NPY}")
    print(f"[DIAG] signals={sig.shape}, labels={labels}")
    if "scale" in data:
        print(f"[DIAG] scale meta: {data['scale']}")

    # 1) 时域信号统计：按 label & channel
    for y in labels:
        idxs = np.where(lab == y)[0]
        print(f"\n[DIAG] label={y} n={idxs.size}")
        for ch, name in enumerate(["X", "Y", "Z"]):
            flat = sig[idxs, ch, :].reshape(-1)
            st = pct_stats(flat)
            print(f"  ch={name}: mean={st['mean']:.4f} std={st['std']:.4f} "
                  f"p1={st['p1']:.3f} p50={st['p50']:.3f} p99={st['p99']:.3f}")

    # 2) GAF 统计：抽样每类若干条，计算每轴 GAF 的像素分布与通道相关性
    print("\n[DIAG] GAF matrix statistics (sampled)")
    for y in labels:
        idxs = np.where(lab == y)[0].tolist()
        rng.shuffle(idxs)
        take = idxs[:SAMPLES_PER_LABEL]
        if not take:
            continue
        gasf_mats = []
        gadf_mats = []
        for idx in take:
            mats_s = []
            mats_d = []
            for ch in range(3):
                s = resample_1d(sig[idx, ch, :], GAF_SIZE)
                ms = gaf_transform(s, method="s", normalize=False).astype(np.float32)
                md = gaf_transform(s, method="d", normalize=False).astype(np.float32)
                mats_s.append(ms)
                mats_d.append(md)
            gasf_mats.append(np.stack(mats_s, axis=0))
            gadf_mats.append(np.stack(mats_d, axis=0))

        gasf = np.stack(gasf_mats, axis=0)  # (K,3,H,W) in [-1,1]
        gadf = np.stack(gadf_mats, axis=0)

        def _print_one(name: str, m: np.ndarray):
            # per channel distribution
            print(f"\n  [{name}] label={y} samples={m.shape[0]} size={GAF_SIZE}")
            for ch, cname in enumerate(["R(X)", "G(Y)", "B(Z)"]):
                flat = m[:, ch, :, :].reshape(-1)
                st = pct_stats(flat, p=(1, 50, 99))
                print(f"    {cname}: mean={st['mean']:.4f} std={st['std']:.4f} "
                      f"p1={st['p1']:.3f} p50={st['p50']:.3f} p99={st['p99']:.3f}")
            # channel correlation on mean matrix
            mean_m = m.mean(axis=0)  # (3,H,W)
            c_rg = corr3(mean_m[0], mean_m[1])
            c_rb = corr3(mean_m[0], mean_m[2])
            c_gb = corr3(mean_m[1], mean_m[2])
            print(f"    corr(mean): R-G={c_rg:.3f} R-B={c_rb:.3f} G-B={c_gb:.3f}")

        _print_one("GASF", gasf)
        _print_one("GADF", gadf)

    # 3) PNG 像素统计（可选）
    if os.path.exists(GAF_RGB_DIR):
        try:
            import matplotlib.image as mpimg

            print(f"\n[DIAG] PNG pixel statistics from {GAF_RGB_DIR}")
            root = Path(GAF_RGB_DIR)
            for y in labels:
                files = sorted((root / str(y)).glob("*.png"))
                if not files:
                    continue
                sel = files[: min(3, len(files))]
                for fp in sel:
                    im = mpimg.imread(fp)  # float in [0,1] typically
                    if im.ndim == 3 and im.shape[2] >= 3:
                        rgb = im[:, :, :3].astype(np.float32)
                        st = pct_stats(rgb.reshape(-1, 3))
                        # channel-wise std
                        stds = rgb.reshape(-1, 3).std(axis=0)
                        means = rgb.reshape(-1, 3).mean(axis=0)
                        print(f"  {fp.name}: meanRGB={means.round(3).tolist()} stdRGB={stds.round(3).tolist()} "
                              f"min={float(rgb.min()):.3f} max={float(rgb.max()):.3f}")
        except Exception as e:
            print(f"[DIAG] PNG stats skipped: {repr(e)}")


if __name__ == "__main__":
    # ensure project root on sys.path for utils import
    import sys

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    main()

