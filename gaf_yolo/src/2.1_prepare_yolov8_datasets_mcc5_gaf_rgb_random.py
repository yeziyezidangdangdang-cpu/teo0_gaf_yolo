"""
MCC5 GAF-RGB 随机划分 YOLO 数据集（D 点劣化对照）。

每类内 shuffle（seed=42）后 70/15/15，与 whole_file（B）/ temporal（C）对比。
前提：已运行 1.4（B 点分桶或全局 1500/类均可，每类图像池一致即可）。

用法：
  python src/2.1_prepare_yolov8_datasets_mcc5_gaf_rgb_random.py
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
    GADF_RGB_MCC5,
    GASF_RGB_MCC5,
    YOLO_DS_MCC5,
    YOLO_GADF_RGB_MCC5_RANDOM,
    YOLO_GASF_RGB_MCC5_RANDOM,
    ensure_mcc5_dirs,
)

RANDOM_SEED = 42
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


def prepare_one_branch(images_root: Path, out_root: Path) -> None:
    if not images_root.is_dir():
        raise FileNotFoundError(f"源图像目录不存在: {images_root}")
    if out_root.exists():
        shutil.rmtree(out_root)

    rng = random.Random(RANDOM_SEED)
    label_dirs = sorted([p for p in images_root.iterdir() if p.is_dir()])
    labels = [p.name for p in label_dirs]
    for split in SPLIT_RATIOS:
        for lab in labels:
            (out_root / split / lab).mkdir(parents=True, exist_ok=True)

    for lab_dir in label_dirs:
        lab = lab_dir.name
        all_imgs = sorted(lab_dir.glob("*.png"))
        if not all_imgs:
            continue
        rng.shuffle(all_imgs)
        n = len(all_imgs)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["val"])
        splits = {
            "train": all_imgs[:n_train],
            "val": all_imgs[n_train : n_train + n_val],
            "test": all_imgs[n_train + n_val :],
        }
        for split_name, files in splits.items():
            for src in files:
                shutil.copy2(src, out_root / split_name / lab / src.name)

    print(f"\n[MCC5-RANDOM] {images_root.name} -> {out_root}")
    for lab in labels:
        counts = {s: len(list((out_root / s / lab).glob("*.png"))) for s in SPLIT_RATIOS}
        print(f"  类别 {lab}: train={counts['train']}, val={counts['val']}, test={counts['test']}")


def main() -> None:
    ensure_mcc5_dirs()
    if not GASF_RGB_MCC5.is_dir():
        raise FileNotFoundError(f"请先运行 1.4_generate_mcc5_gaf_rgb_images.py: {GASF_RGB_MCC5.parent}")

    print(f"[MCC5-RANDOM] seed={RANDOM_SEED}, 每类 shuffle 后 70/15/15")
    print("=== GASF ===")
    prepare_one_branch(GASF_RGB_MCC5, YOLO_GASF_RGB_MCC5_RANDOM)
    print("=== GADF ===")
    prepare_one_branch(GADF_RGB_MCC5, YOLO_GADF_RGB_MCC5_RANDOM)

    sidecar = YOLO_DS_MCC5 / "split_mode_random.json"
    sidecar.write_text(
        json.dumps(
            {
                "exp_point": "d",
                "split_mode": "random",
                "random_seed": RANDOM_SEED,
                "gasf_out": str(YOLO_GASF_RGB_MCC5_RANDOM),
                "gadf_out": str(YOLO_GADF_RGB_MCC5_RANDOM),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[MCC5-RANDOM] 完成 | {sidecar}")


if __name__ == "__main__":
    main()
