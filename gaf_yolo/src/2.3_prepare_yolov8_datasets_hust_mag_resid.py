"""
基于 HUST 幅值+高通残差 STFT 频谱图（1.6 生成）整理 YOLOv8 分类数据集。

输入：
- data/spectrograms_mag_resid/<label>/spec_mr_<label>_<idx>.png

输出：
- yolo_datasets/hust_spec_mag_resid_cls/train/<label>/*.png
- yolo_datasets/hust_spec_mag_resid_cls/val/<label>/*.png
- yolo_datasets/hust_spec_mag_resid_cls/test/<label>/*.png
"""

import shutil
from pathlib import Path
import random


RANDOM_SEED = 42
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


def prepare_one_branch(images_root: Path, out_root: Path) -> None:
    if not images_root.exists():
        raise FileNotFoundError(f"源图像目录不存在: {images_root}")

    if out_root.exists():
        shutil.rmtree(out_root)

    rng = random.Random(RANDOM_SEED)
    summary = {}

    label_dirs = sorted([p for p in images_root.iterdir() if p.is_dir()])
    labels = [p.name for p in label_dirs]

    for split in SPLIT_RATIOS.keys():
        for lab in labels:
            (out_root / split / lab).mkdir(parents=True, exist_ok=True)

    for lab_dir in label_dirs:
        lab = lab_dir.name
        all_imgs = sorted([p for p in lab_dir.glob("*.png") if p.is_file()])
        if not all_imgs:
            print(f"⚠️ 标签 {lab} 在 {lab_dir} 下未找到 PNG 图像")
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
                shutil.copy2(src, dst_base / src.name)

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
    images_root = project_root / "data" / "spectrograms_mag_resid"
    out_root = project_root / "yolo_datasets" / "hust_spec_mag_resid_cls"

    print("=== 准备 HUST 幅值+高通残差 STFT 频谱图分类数据集 ===")
    prepare_one_branch(images_root, out_root)

    print("\n[INFO] YOLOv8 HUST Mag+Resid STFT 分类数据集已准备完成。")
    print(f"   root: {out_root}")


if __name__ == "__main__":
    main()

