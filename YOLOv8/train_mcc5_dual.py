"""
Train dual YOLOv8 classifiers for MCC5-THU GASF-RGB / GADF-RGB (B 点三轴泛化验证).

Data: gaf_yolo/yolo_datasets_mcc5/mcc5_{gasf,gadf}_rgb_cls_strict
Weights: runs/classify/6_edit/mcc5_cls/mcc5_{gasf,gadf}/
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO

from mcc5_paths import GADF_DATA_ROOT, GASF_DATA_ROOT, MCC5_IMGSZ, MCC5_PROJECT, set_exp_point, set_split_mode


def train_branch(
    branch: str,
    data_root,
    model_size: str = "n",
    epochs: int = 40,
    fraction: float = 1.0,
    batch: int = 32,
    workers: int = 2,
):
    model = YOLO(f"yolov8{model_size}-cls.pt")
    import mcc5_paths as mp

    return model.train(
        data=str(data_root),
        epochs=epochs,
        imgsz=mp.MCC5_IMGSZ,
        batch=batch,
        project=mp.MCC5_PROJECT,
        name=f"mcc5_{branch}",
        exist_ok=True,
        verbose=True,
        fraction=fraction,
        workers=workers,
        device="cpu",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MCC5 dual YOLOv8 GAF-RGB classifiers")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--model-size", type=str, default="n", choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--exp-point", choices=("c", "b", "d"), default="b", help="B=whole_file; C=temporal; D=random")
    parser.add_argument("--split-mode", choices=("temporal", "whole_file", "random"), default=None, help="覆盖 --exp-point")
    args = parser.parse_args()

    if args.split_mode:
        set_split_mode(args.split_mode)
    else:
        set_exp_point(args.exp_point)
    import mcc5_paths as mp

    print(
        f"[train_mcc5_dual] exp_point={mp.get_exp_point().upper()}, epochs={args.epochs}, "
        f"imgsz={mp.MCC5_IMGSZ}, project={mp.MCC5_PROJECT}"
    )
    print(f"  GASF data: {mp.GASF_DATA_ROOT}")
    print(f"  GADF data: {mp.GADF_DATA_ROOT}")
    train_branch("gasf", mp.GASF_DATA_ROOT, model_size=args.model_size, epochs=args.epochs,
                 fraction=args.fraction, batch=args.batch, workers=args.workers)
    train_branch("gadf", mp.GADF_DATA_ROOT, model_size=args.model_size, epochs=args.epochs,
                 fraction=args.fraction, batch=args.batch, workers=args.workers)
