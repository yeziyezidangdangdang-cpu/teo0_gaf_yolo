"""
Train dual YOLOv8 classifiers for CWRU v2 GASF/GADF (2048/512 + robust preprocess).

Data: D:/TEP_2_repro/gaf_yolo/yolo_datasets_v2/{gasf,gadf}_cls_strict
Weights: runs/classify/6_edit/cwru_v2_cls/cwru_v2_{gasf,gadf}/
"""

import argparse

from ultralytics import YOLO

from cwru_paths import CWRU_V2_IMGSZ, CWRU_V2_PROJECT, GADF_DATA_ROOT, GASF_DATA_ROOT, RUNS_DIR


def train_branch(
    branch: str,
    data_root,
    model_size: str = "n",
    epochs: int = 40,
    fraction: float = 1.0,
    batch: int = 32,
    workers: int = 2,
):
    model_name = f"yolov8{model_size}-cls.pt"
    model = YOLO(model_name)

    return model.train(
        data=str(data_root),
        epochs=epochs,
        imgsz=CWRU_V2_IMGSZ,
        batch=batch,
        project=CWRU_V2_PROJECT,
        name=f"cwru_v2_{branch}",
        exist_ok=True,
        verbose=True,
        fraction=fraction,
        workers=workers,
        device="cpu",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CWRU v2 dual YOLOv8 classifiers")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--model-size", type=str, default="n", choices=["n", "s", "m", "l", "x"])
    args = parser.parse_args()

    print(
        f"[train_cwru_dual v2] epochs={args.epochs}, imgsz={CWRU_V2_IMGSZ}, "
        f"fraction={args.fraction}, batch={args.batch}"
    )

    train_branch(
        "gasf",
        GASF_DATA_ROOT,
        model_size=args.model_size,
        epochs=args.epochs,
        fraction=args.fraction,
        batch=args.batch,
        workers=args.workers,
    )
    train_branch(
        "gadf",
        GADF_DATA_ROOT,
        model_size=args.model_size,
        epochs=args.epochs,
        fraction=args.fraction,
        batch=args.batch,
        workers=args.workers,
    )
