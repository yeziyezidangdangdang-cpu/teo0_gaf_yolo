"""
Train a single YOLOv8 classification model for HUST-bearing GAF-RGB images（GAF 主线简化版实验）.

用途：
- 作为多轴 GAF 双分支方案的**简化对照模型**，仅使用 GASF-RGB 或 GADF-RGB 其中一支，
  其它预处理与数据集准备步骤与主线保持一致，用于做“单分支 vs 双分支”的结构消融。

数据集由 `D:\TEP_2_repro\gaf_yolo\src\1.1/1.4/2.2` 脚本准备：
- GASF-RGB: D:/TEP_2_repro/gaf_yolo/yolo_datasets/hust_gasf_rgb_cls
- GADF-RGB: D:/TEP_2_repro/gaf_yolo/yolo_datasets/hust_gadf_rgb_cls
"""

import repro_paths as pp  # noqa: E402

import argparse

from ultralytics import YOLO


GASF_DATA_ROOT = str(pp.YOLO_DS / 'hust_gasf_rgb_cls')
GADF_DATA_ROOT = str(pp.YOLO_DS / 'hust_gadf_rgb_cls')


def train_single_branch(
    branch: str,
    data_root: str,
    model_size: str = "n",
    epochs: int = 40,
    fraction: float = 1.0,
    batch: int = 32,
    workers: int = 2,
):
    """
    训练单分支 YOLOv8‑cls 模型（GASF 或 GADF）。
    """
    model_name = f"yolov8{model_size}-cls.pt"
    model = YOLO(model_name)

    results = model.train(
        data=data_root,
        epochs=epochs,
        imgsz=128,
        batch=batch,
        project="runs/hust_gaf_rgb_cls",
        name=f"hust_{branch}_single",
        exist_ok=True,
        verbose=True,
        fraction=fraction,
        workers=workers,
        device="cpu",
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train single-branch YOLOv8 classifier for HUST GAF-RGB (simplified baseline)"
    )
    parser.add_argument(
        "--branch",
        type=str,
        default="gasf",
        choices=["gasf", "gadf"],
        help="which GAF branch to use: gasf (default) or gadf",
    )
    parser.add_argument("--epochs", type=int, default=40, help="training epochs (default 40)")
    parser.add_argument("--fraction", type=float, default=1.0, help="data fraction 0~1 (default 1.0)")
    parser.add_argument("--batch", type=int, default=32, help="batch size (default 32)")
    parser.add_argument("--workers", type=int, default=2, help="dataloader workers (default 2)")
    parser.add_argument(
        "--model-size", type=str, default="n", choices=["n", "s", "m", "l", "x"], help="YOLOv8 model size"
    )
    args = parser.parse_args()

    if args.branch == "gasf":
        data_root = GASF_DATA_ROOT
    else:
        data_root = GADF_DATA_ROOT

    print(
        f"[train_hust_gaf_single] 训练单分支 {args.branch.upper()}-RGB 分类器，"
        f"data_root={data_root}, epochs={args.epochs}, fraction={args.fraction}, "
        f"batch={args.batch}, workers={args.workers}, model={args.model_size}"
    )

    train_single_branch(
        branch=args.branch,
        data_root=data_root,
        model_size=args.model_size,
        epochs=args.epochs,
        fraction=args.fraction,
        batch=args.batch,
        workers=args.workers,
    )

