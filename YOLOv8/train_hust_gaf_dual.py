"""
Train two YOLOv8 classification models for HUST-bearing GASF and GADF images（双分支对比 STFT）.

数据集由 `D:\TEP_2_repro\gaf_yolo\src\0.0/1.1/1.3/2.1` 几个脚本准备：
- GASF: D:/TEP_2_repro/gaf_yolo/yolo_datasets/hust_gasf_cls
- GADF: D:/TEP_2_repro/gaf_yolo/yolo_datasets/hust_gadf_cls

每个根目录下都有 train/val/test/{0,1,2,3} 子文件夹。

快速小体量试跑示例：
    cd D:\TEP_2_repro\YOLOv8
    python train_hust_gaf_dual.py --epochs 5 --fraction 0.1
"""

import argparse

from ultralytics import YOLO


GASF_DATA_ROOT = r"D:\TEP_2_repro\gaf_yolo\yolo_datasets\hust_gasf_cls"
GADF_DATA_ROOT = r"D:\TEP_2_repro\gaf_yolo\yolo_datasets\hust_gadf_cls"


def train_branch(
    branch: str,
    data_root: str,
    model_size: str = "n",
    epochs: int = 40,
    fraction: float = 1.0,
    batch: int = 32,
    workers: int = 2,
):
    """
    训练单个 YOLOv8 分类模型（GASF 或 GADF 分支）。

    Args:
        branch: "gasf" or "gadf"
        data_root: 数据集根路径（含 train/val/test 子目录）
        model_size: ['n', 's', 'm', 'l', 'x'] 之一
        epochs: 训练轮数
        fraction: 使用的数据比例（0~1），用于小体量快速试跑
        batch: batch size
        workers: dataloader workers
    """
    model_name = f"yolov8{model_size}-cls.pt"
    model = YOLO(model_name)

    results = model.train(
        data=data_root,
        epochs=epochs,
        imgsz=128,
        batch=batch,
        project="runs/hust_gaf_cls",
        name=f"hust_{branch}",
        exist_ok=True,
        verbose=True,
        fraction=fraction,
        workers=workers,
        device="cpu",  # 若有 GPU 可改为 "0"
    )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train dual YOLOv8 classifiers for HUST GASF/GADF")
    parser.add_argument("--epochs", type=int, default=40, help="epochs per branch (default 40)")
    parser.add_argument("--fraction", type=float, default=1.0, help="data fraction 0~1 (default 1.0)")
    parser.add_argument("--batch", type=int, default=32, help="batch size (default 32)")
    parser.add_argument("--workers", type=int, default=2, help="dataloader workers (default 2)")
    parser.add_argument(
        "--model-size", type=str, default="n", choices=["n", "s", "m", "l", "x"], help="YOLOv8 model size"
    )
    args = parser.parse_args()

    print(
        f"[train_hust_gaf_dual] 将依次训练 GASF、GADF 两个分支，"
        f"每分支 epochs={args.epochs}, fraction={args.fraction}, batch={args.batch}, workers={args.workers}。"
    )
    print("(GASF 跑完会自动接着跑 GADF，请勿中途关掉)\n")

    # Train GASF branch
    print(
        f"=== Training GASF classifier (branch: gasf), "
        f"epochs={args.epochs}, fraction={args.fraction}, model={args.model_size}, "
        f"batch={args.batch}, workers={args.workers} ==="
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

    # Train GADF branch
    print(
        f"\n=== Training GADF classifier (branch: gadf), "
        f"epochs={args.epochs}, fraction={args.fraction}, model={args.model_size}, "
        f"batch={args.batch}, workers={args.workers} ==="
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

