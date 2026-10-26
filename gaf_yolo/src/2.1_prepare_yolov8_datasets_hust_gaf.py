"""
基于 HUST-bearing 的 SA-GAF 图像，整理为 YOLOv8 分类任务可直接使用的数据集。

前置步骤：
- 0.0_prepare_hust_dataset.py 生成 final_dataset_hust.csv
- 1.1_preprocess_custom.py 生成 gaf_preprocessed_hust.npy
- 1.3_generate_gaf_enhanced_images.py 生成 SA-GAF 图像：
  - data/images_enhanced/gasf/<label>/*.png
  - data/images_enhanced/gadf/<label>/*.png

本脚本将上述图像划分为 train/val/test 三个子集：
- GASF 数据集根目录: yolo_datasets/hust_gasf_cls
- GADF 数据集根目录: yolo_datasets/hust_gadf_cls
"""

import shutil
from pathlib import Path
import random


RANDOM_SEED = 42
# 按 7:1.5:1.5 ≈ 70% / 15% / 15% 划分
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


def prepare_one_branch(images_root: Path, out_root: Path) -> None:
    """
    针对 GAF 图像构建 YOLOv8 分类数据集目录。

    images_root:
      - 结构: images_root/<label>/*.png
    out_root:
      - 结构: out_root/{train,val,test}/<label>/*.png
    """
    if not images_root.exists():
        raise FileNotFoundError(f"源图像目录不存在: {images_root}")

    # 每次运行前清空输出目录，避免重复累积
    if out_root.exists():
        shutil.rmtree(out_root)

    rng = random.Random(RANDOM_SEED)

    summary = {}

    # 自动发现标签子目录
    label_dirs = sorted([p for p in images_root.iterdir() if p.is_dir()])
    labels = [p.name for p in label_dirs]

    # 先创建输出目录结构
    for split in SPLIT_RATIOS.keys():
        for lab in labels:
            (out_root / split / lab).mkdir(parents=True, exist_ok=True)

    for lab_dir in label_dirs:
        lab = lab_dir.name
        src_dir = lab_dir
        if not src_dir.exists():
            raise FileNotFoundError(f"标签目录不存在: {src_dir}")

        all_imgs = sorted([p for p in src_dir.glob("*.png") if p.is_file()])
        if not all_imgs:
            print(f"⚠️ 标签 {lab} 在 {src_dir} 下未找到 PNG 图像")
            continue

        rng.shuffle(all_imgs)
        n = len(all_imgs)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["val"])
        n_test = n - n_train - n_val

        splits = {
            "train": all_imgs[:n_train],
            "val": all_imgs[n_train : n_train + n_val],
            "test": all_imgs[n_train + n_val :],
        }

        summary[lab] = {k: len(v) for k, v in splits.items()}

        for split_name, files in splits.items():
            dst_base = out_root / split_name / lab
            for src in files:
                dst = dst_base / src.name
                shutil.copy2(src, dst)

    print(f"\n[INFO] 已整理数据集: {images_root} -> {out_root}")
    print("划分统计（每类 train/val/test 张数）：")
    for lab in labels:
        if lab in summary:
            s = summary[lab]
            print(f"  类别 {lab}: train={s['train']}, val={s['val']}, test={s['test']}")
        else:
            print(f"  类别 {lab}: 无图像")


def main():
    project_root = Path(__file__).resolve().parents[1]

    gasf_root = project_root / "data" / "images_enhanced" / "gasf"
    gadf_root = project_root / "data" / "images_enhanced" / "gadf"

    out_root = project_root / "yolo_datasets"
    gasf_out = out_root / "hust_gasf_cls"
    gadf_out = out_root / "hust_gadf_cls"

    print("=== 准备 HUST GASF 分类数据集 ===")
    prepare_one_branch(gasf_root, gasf_out)

    print("\n=== 准备 HUST GADF 分类数据集 ===")
    prepare_one_branch(gadf_root, gadf_out)

    print("\n[INFO] YOLOv8 HUST GAF 分类数据集已准备完成。")
    print(f"   GASF root: {gasf_out}")
    print(f"   GADF root: {gadf_out}")


if __name__ == "__main__":
    main()

