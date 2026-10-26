"""
Train a YOLOv8 classification model on HUST single-axis STFT spectrograms (X/Y/Z).

数据集结构由 `D:\TEP_2_repro\gaf_yolo\src\1.2_x/y/z, 2.4` 脚本准备：
- X 轴: D:/TEP_2_repro/gaf_yolo/yolo_datasets/hust_spec_x_cls
- Y 轴: D:/TEP_2_repro/gaf_yolo/yolo_datasets/hust_spec_y_cls
- Z 轴: D:/TEP_2_repro/gaf_yolo/yolo_datasets/hust_spec_z_cls

示例：
    cd D:\TEP_2_repro\YOLOv8
    python train_hust_spec_axis.py --axis x --epochs 40 --fraction 1.0
"""

import repro_paths as pp  # noqa: E402

import argparse

from ultralytics import YOLO


DATA_ROOTS = {
    "x": str(pp.YOLO_DS / 'hust_spec_x_cls'),
    "y": str(pp.YOLO_DS / 'hust_spec_y_cls'),
    "z": str(pp.YOLO_DS / 'hust_spec_z_cls'),
}


def train_hust_spec_axis(
    axis: str = 'x',
    model_size: str = "n",
    epochs: int = 40,
    fraction: float = 1.0,
    batch: int = 32,
    workers: int = 2,
):
    """
    训练单个 YOLOv8 分类模型（HUST 单轴 STFT）。
    axis: 'x' / 'y' / 'z'
    """
    data_root = DATA_ROOTS[axis]
    model_name = f"yolov8{model_size}-cls.pt"
    model = YOLO(model_name)

    results = model.train(
        data=data_root,
        epochs=epochs,
        imgsz=128,
        batch=batch,
        project=f"runs/hust_spec_{axis}",
        name=f"hust_spec_{axis}_cls",
        exist_ok=True,
        verbose=True,
        fraction=fraction,
        workers=workers,
        device="cpu",  # 若有 GPU 可改为 "0"
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 classifier on HUST single-axis STFT spectrograms")
    parser.add_argument("--axis", type=str, required=True, choices=["x", "y", "z"], help="axis to train on")
    parser.add_argument("--epochs", type=int, default=40, help="epochs (default 40)")
    parser.add_argument("--fraction", type=float, default=1.0, help="data fraction 0~1 (default 1.0)")
    parser.add_argument("--batch", type=int, default=32, help="batch size (default 32)")
    parser.add_argument("--workers", type=int, default=2, help="dataloader workers (default 2)")
    parser.add_argument(
        "--model-size", type=str, default="n", choices=["n", "s", "m", "l", "x"], help="YOLOv8 model size"
    )
    args = parser.parse_args()

    print(
        f"[train_hust_spec_axis] 训练 HUST 单轴 STFT 分类模型，"
        f"axis={args.axis}, epochs={args.epochs}, fraction={args.fraction}, "
        f"batch={args.batch}, workers={args.workers}, model={args.model_size}"
    )

    train_hust_spec_axis(
        axis=args.axis,
        model_size=args.model_size,
        epochs=args.epochs,
        fraction=args.fraction,
        batch=args.batch,
        workers=args.workers,
    )

