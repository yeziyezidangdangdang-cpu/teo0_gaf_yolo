"""
HUST GAF-RGB 随机划分 YOLO 数据集（D 点劣化对照）。

每类内 shuffle 后 70/15/15，用于与 strict temporal（2.2）定量对比，
回应审稿人「与随机划分对比」要求。

用法：
  python src/2.1_prepare_yolov8_datasets_hust_gaf_rgb_random.py
"""
from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.project_paths import (  # noqa: E402
    GADF_RGB_HUST,
    GASF_RGB_HUST,
    YOLO_DS_HUST,
    YOLO_GADF_RGB_HUST_RANDOM,
    YOLO_GASF_RGB_HUST_RANDOM,
    ensure_hust_dirs,
)

RANDOM_SEED = 42
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


def prepare_one_branch(images_root: Path, out_root: Path) -> dict[str, dict[str, int]]:
    if not images_root.is_dir():
        raise FileNotFoundError(f"源图像目录不存在: {images_root}")

    if out_root.exists():
        shutil.rmtree(out_root)

    rng = random.Random(RANDOM_SEED)
    summary: dict[str, dict[str, int]] = {}

    label_dirs = sorted([p for p in images_root.iterdir() if p.is_dir()])
    labels = [p.name for p in label_dirs]

    for split in SPLIT_RATIOS:
        for lab in labels:
            (out_root / split / lab).mkdir(parents=True, exist_ok=True)

    for lab_dir in label_dirs:
        lab = lab_dir.name
        all_imgs = sorted([p for p in lab_dir.glob("*.png") if p.is_file()])
        if not all_imgs:
            print(f"[RANDOM-GAF] 标签 {lab} 无 PNG")
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

    print(f"\n[RANDOM-GAF] {images_root.name} -> {out_root}")
    for lab in labels:
        s = summary.get(lab, {"train": 0, "val": 0, "test": 0})
        print(f"  类别 {lab}: train={s['train']}, val={s['val']}, test={s['test']}")
    return summary


def main() -> None:
    ensure_hust_dirs()
    if not GASF_RGB_HUST.is_dir() or not GADF_RGB_HUST.is_dir():
        raise FileNotFoundError(
            f"缺少 HUST GAF-RGB 源图，请先运行 1.4_generate_hust_gaf_rgb_images.py\n"
            f"  期望: {GASF_RGB_HUST.parent}"
        )

    print(f"[RANDOM-GAF] seed={RANDOM_SEED}, split=70/15/15 per label (shuffle)")
    print("\n=== GASF-RGB 随机划分 ===")
    prepare_one_branch(GASF_RGB_HUST, YOLO_GASF_RGB_HUST_RANDOM)
    print("\n=== GADF-RGB 随机划分 ===")
    prepare_one_branch(GADF_RGB_HUST, YOLO_GADF_RGB_HUST_RANDOM)

    sidecar = YOLO_DS_HUST / "split_mode_random_hust.json"
    sidecar.write_text(
        json.dumps(
            {
                "exp_point": "d",
                "split_mode": "random",
                "random_seed": RANDOM_SEED,
                "gasf_out": str(YOLO_GASF_RGB_HUST_RANDOM),
                "gadf_out": str(YOLO_GADF_RGB_HUST_RANDOM),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n[RANDOM-GAF] 完成 | sidecar={sidecar}")


if __name__ == "__main__":
    main()
