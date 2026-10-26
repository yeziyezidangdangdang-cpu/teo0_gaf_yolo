"""Train CWRU v2 SA-GAF dual classifiers."""

import argparse

from ultralytics import YOLO

from cwru_paths import CWRU_V2_IMGSZ, CWRU_V2_PROJECT, GADF_DATA_ROOT_ENH, GASF_DATA_ROOT_ENH


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
    parser = argparse.ArgumentParser(description="Train CWRU v2 enhanced dual classifiers")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--model-size", type=str, default="n", choices=["n", "s", "m", "l", "x"])
    args = parser.parse_args()

    train_branch("gasf_enh", GASF_DATA_ROOT_ENH, args.model_size, args.epochs, args.fraction, args.batch, args.workers)
    train_branch("gadf_enh", GADF_DATA_ROOT_ENH, args.model_size, args.epochs, args.fraction, args.batch, args.workers)
