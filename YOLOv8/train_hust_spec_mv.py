"""
Train a YOLOv8 classification model on HUST-bearing multi-view STFT spectrograms.

数据集结构由 `D:\TEP_2_repro\gaf_yolo\src\1.5/2.1` 脚本准备：
- 频谱图根目录: D:/TEP_2_repro/gaf_yolo/data/spectrograms_mv/<label>/*.png
- YOLOv8 分类数据集根目录: D:/TEP_2_repro/gaf_yolo/yolo_datasets/hust_spec_mv_cls
  - train/<label>
  - val/<label>
  - test/<label>

快速小体量试跑示例：
    cd D:\TEP_2_repro\YOLOv8
    python train_hust_spec_mv.py --epochs 5 --fraction 0.1 --batch 16
"""

import repro_paths as pp  # noqa: E402

import argparse

from ultralytics import YOLO


DATA_ROOT = str(pp.YOLO_DS / 'hust_spec_mv_cls')


def train_hust_spec_mv(
    data_root: str = DATA_ROOT,
    model_size: str = "n",
    epochs: int = 40,
    fraction: float = 1.0,
    batch: int = 32,
    workers: int = 2,
):
    """
    训练单个 YOLOv8 分类模型（HUST 多视角 STFT 频谱图）。
    """
    model_name = f"yolov8{model_size}-cls.pt"
    model = YOLO(model_name)

    results = model.train(
        data=data_root,
        epochs=epochs,
        imgsz=128,
        batch=batch,
        project="runs/hust_spec_mv",
        name="hust_spec_mv_cls",
        exist_ok=True,
        verbose=True,
        fraction=fraction,
        workers=workers,
        device="cpu",  # 若有 GPU 可改为 "0"
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 classifier on HUST multi-view STFT spectrograms")
    parser.add_argument("--epochs", type=int, default=40, help="epochs (default 40)")
    parser.add_argument("--fraction", type=float, default=1.0, help="data fraction 0~1 (default 1.0)")
    parser.add_argument("--batch", type=int, default=32, help="batch size (default 32)")
    parser.add_argument("--workers", type=int, default=2, help="dataloader workers (default 2)")
    parser.add_argument(
        "--model-size", type=str, default="n", choices=["n", "s", "m", "l", "x"], help="YOLOv8 model size"
    )
    args = parser.parse_args()

    print(
        f"[train_hust_spec_mv] 训练 HUST 多视角 STFT 分类模型："
        f"epochs={args.epochs}, fraction={args.fraction}, batch={args.batch}, workers={args.workers}, "
        f"model={args.model_size}"
    )

    train_hust_spec_mv(
        data_root=DATA_ROOT,
        model_size=args.model_size,
        epochs=args.epochs,
        fraction=args.fraction,
        batch=args.batch,
        workers=args.workers,
    )

