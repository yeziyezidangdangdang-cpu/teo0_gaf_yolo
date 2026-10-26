"""
Train dual YOLOv8 classifiers for HUST GAF-RGB (GASF/GADF).

  main — strict temporal（2.2）
  d    — random shuffle（2.1，劣化对照）

Usage:
    cd D:\\TEP_2_repro\\YOLOv8
    python train_hust_gaf_rgb_dual.py --exp-point d
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO

from hust_paths import (
    GADF_DATA_ROOT,
    GASF_DATA_ROOT,
    HUST_IMGSZ,
    HUST_PROJECT,
    get_exp_point,
    set_exp_point,
)


def train_branch(
    branch: str,
    data_root,
    model_size: str = "n",
    epochs: int = 40,
    fraction: float = 1.0,
    batch: int = 32,
    workers: int = 2,
):
    import hust_paths as hp

    model = YOLO(f"yolov8{model_size}-cls.pt")
    return model.train(
        data=str(data_root),
        epochs=epochs,
        imgsz=hp.HUST_IMGSZ,
        batch=batch,
        project=hp.HUST_PROJECT,
        name=f"hust_{branch}",
        exist_ok=True,
        verbose=True,
        fraction=fraction,
        workers=workers,
        device="cpu",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HUST dual YOLOv8 GAF-RGB classifiers")
    parser.add_argument("--exp-point", choices=("main", "d"), default="main")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--model-size", type=str, default="n", choices=["n", "s", "m", "l", "x"])
    args = parser.parse_args()

    set_exp_point(args.exp_point)
    import hust_paths as hp

    print(
        f"[train_hust_gaf_rgb_dual] exp_point={get_exp_point().upper()}, "
        f"epochs={args.epochs}, project={hp.HUST_PROJECT}"
    )
    print(f"  GASF data: {hp.GASF_DATA_ROOT}")
    print(f"  GADF data: {hp.GADF_DATA_ROOT}")

    train_branch(
        "gasf_rgb",
        hp.GASF_DATA_ROOT,
        model_size=args.model_size,
        epochs=args.epochs,
        fraction=args.fraction,
        batch=args.batch,
        workers=args.workers,
    )
    train_branch(
        "gadf_rgb",
        hp.GADF_DATA_ROOT,
        model_size=args.model_size,
        epochs=args.epochs,
        fraction=args.fraction,
        batch=args.batch,
        workers=args.workers,
    )
