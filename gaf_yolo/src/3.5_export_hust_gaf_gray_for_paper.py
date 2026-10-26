from __future__ import annotations

"""
导出适合《机械工程学报》黑白印刷的 HUST GAF 图像：
- 从 data/gaf_rgb/gasf, data/gaf_rgb/gadf 读取现有 GASF/GADF-RGB 图
- 转为高对比度灰度图（仅用于论文插图，不参与训练）

用法：
    cd D:\\TEP_2\\gaf_yolo\\src
    python 3.5_export_hust_gaf_gray_for_paper.py

输出：
    docs/paper_gaf_gray/
        gasf/<label>/gasf_gray_<label>_<idx>.png
        gadf/<label>/gadf_gray_<label>_<idx>.png
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GASF_RGB_ROOT = PROJECT_ROOT / "data" / "gaf_rgb" / "gasf"
GADF_RGB_ROOT = PROJECT_ROOT / "data" / "gaf_rgb" / "gadf"

OUT_ROOT = PROJECT_ROOT / "docs" / "paper_gaf_gray"

# 每类导出的示例张数（论文插图一般只需要少量代表性样本）
MAX_IMAGES_PER_LABEL = 12

# 灰度对比度拉伸百分位（增强纹理层次）
STRETCH_LOW_PERC = 5.0
STRETCH_HIGH_PERC = 95.0


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _rgb_to_gray_high_contrast(rgb: np.ndarray) -> np.ndarray:
    """
    rgb: (H, W, 3) in [0,1] or [0,255]
    返回: (H, W) in [0,1]，经过百分位对比度拉伸，适合黑白印刷。
    """
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.max() > 1.0 + 1e-3:
        arr = arr / 255.0

    if arr.ndim == 3 and arr.shape[2] >= 3:
        gray = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
    else:
        gray = arr[..., 0] if arr.ndim == 3 else arr

    # 百分位拉伸，增强纹理层次
    lo, hi = np.percentile(gray, [STRETCH_LOW_PERC, STRETCH_HIGH_PERC])
    if hi <= lo:
        lo, hi = float(gray.min()), float(gray.max() + 1e-8)
    gray = np.clip(gray, lo, hi)
    gray = (gray - lo) / (hi - lo + 1e-8)
    return gray.astype(np.float32)


def _export_branch(src_root: Path, out_root: Path, prefix: str) -> None:
    if not src_root.exists():
        print(f"[WARN] 源目录不存在，跳过: {src_root}")
        return

    print(f"=== 导出 {prefix.upper()} 分支灰度 GAF 图（用于论文插图） ===")
    _ensure_dir(out_root)

    label_dirs = sorted(p for p in src_root.iterdir() if p.is_dir())
    for lab_dir in label_dirs:
        lab = lab_dir.name
        out_lab_dir = out_root / lab
        _ensure_dir(out_lab_dir)

        pngs = sorted(p for p in lab_dir.glob("*.png") if p.is_file())
        if not pngs:
            print(f"  ⚠ 标签 {lab} 无 PNG，跳过")
            continue

        count = 0
        for src in pngs:
            if count >= MAX_IMAGES_PER_LABEL:
                break

            img = plt.imread(src)
            gray = _rgb_to_gray_high_contrast(img)

            dst_name = f"{prefix}_gray_{lab}_{src.stem.split('_')[-1]}.png"
            dst_path = out_lab_dir / dst_name
            plt.imsave(dst_path, gray, cmap="gray", origin="lower")
            count += 1

        print(f"  标签 {lab}: 导出 {count} 张灰度图 -> {out_lab_dir}")


def main() -> None:
    _export_branch(GASF_RGB_ROOT, OUT_ROOT / "gasf", "gasf")
    _export_branch(GADF_RGB_ROOT, OUT_ROOT / "gadf", "gadf")
    print(f"\n[OK] 已导出适合黑白印刷的 GAF 灰度图到: {OUT_ROOT}")


if __name__ == "__main__":
    # 避免在某些环境下当前工作目录不是 src/
    os.chdir(PROJECT_ROOT / "src")
    main()

