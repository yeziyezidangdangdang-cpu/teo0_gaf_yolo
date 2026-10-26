"""
为 HUST 三个单轴 STFT 频谱图（X/Y/Z）分别整理 YOLOv8 分类数据集。

输入根目录：
- data/spec_x/<label>/*.png
- data/spec_y/<label>/*.png
- data/spec_z/<label>/*.png

输出：
- yolo_datasets/hust_spec_x_cls/train|val|test/<label>/*.png
- yolo_datasets/hust_spec_y_cls/...
- yolo_datasets/hust_spec_z_cls/...
"""

import shutil
from pathlib import Path
import random


RANDOM_SEED = 42
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


def prepare_one_branch(images_root: Path, out_root: Path) -> None:
    if not images_root.exists():
        print(f"[AXIS-DATA] 源图像目录不存在，跳过: {images_root}")
        return

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
            print(f"[AXIS-DATA] ⚠️ 标签 {lab} 在 {lab_dir} 下未找到 PNG 图像")
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

    print(f"\n[AXIS-DATA] 已整理数据集: {images_root} -> {out_root}")
    print("划分统计（每类 train/val/test 张数）：")
    for lab in labels:
        if lab in summary:
            s = summary[lab]
            print(f"  类别 {lab}: train={s['train']}, val={s['val']}, test={s['test']}")
        else:
            print(f"  类别 {lab}: 无图像")


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data"
    out_root = project_root / "yolo_datasets"

    cfgs = [
        ("spec_x", "hust_spec_x_cls"),
        ("spec_y", "hust_spec_y_cls"),
        ("spec_z", "hust_spec_z_cls"),
    ]

    for subdir, out_name in cfgs:
        images_root = data_root / subdir
        axis_out_root = out_root / out_name
        print(f"\n=== 准备 HUST {subdir} 单轴 STFT 数据集 ===")
        prepare_one_branch(images_root, axis_out_root)

    print("\n[AXIS-DATA] 三个单轴 STFT YOLOv8 数据集已准备完成。")


if __name__ == "__main__":
    main()

