"""
Late fusion evaluation for HUST single-axis STFT classifiers (方案一融合评估).

思路：
- 已分别训练好 X/Y/Z 三个单轴 STFT 分类模型：
  - runs/hust_spec_x/hust_spec_x_cls/weights/best.pt
  - runs/hust_spec_y/hust_spec_y_cls/weights/best.pt
  - runs/hust_spec_z/hust_spec_z_cls/weights/best.pt
- 数据集结构由 2.4_prepare_yolov8_datasets_hust_axes.py 准备：
  - D:/TEP_2_repro/gaf_yolo/yolo_datasets/hust_spec_{x,y,z}_cls/test/<label>/*.png
- 三个轴的数据集划分使用相同随机种子，文件名一一对应，
  因此可以按文件名将 test 集的同一片段在 X/Y/Z 三个模型上做预测，然后做 late fusion。

本脚本：
- 读取三个 best.pt
- 遍历 X 轴 test 集图像，找到对应 Y/Z 轴图像
- 取三个模型的 softmax 概率，做「简单平均」得到融合概率
- 统计单轴 / 融合 的 test accuracy
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from ultralytics import YOLO


# 对于 YOLOv8 项目，本脚本所在目录即项目根目录（包含 runs、train_*.py 等）
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_ROOTS = {
    "x": PROJECT_ROOT.parent / "gaf_yolo" / "yolo_datasets" / "hust_spec_x_cls",
    "y": PROJECT_ROOT.parent / "gaf_yolo" / "yolo_datasets" / "hust_spec_y_cls",
    "z": PROJECT_ROOT.parent / "gaf_yolo" / "yolo_datasets" / "hust_spec_z_cls",
}

WEIGHTS = {
    # 注意：Ultralytics 默认分类运行目录为 runs/classify/runs/...
    "x": PROJECT_ROOT / "runs" / "classify" / "runs" / "hust_spec_x" / "hust_spec_x_cls" / "weights" / "best.pt",
    "y": PROJECT_ROOT / "runs" / "classify" / "runs" / "hust_spec_y" / "hust_spec_y_cls" / "weights" / "best.pt",
    "z": PROJECT_ROOT / "runs" / "classify" / "runs" / "hust_spec_z" / "hust_spec_z_cls" / "weights" / "best.pt",
}


def load_models() -> dict[str, YOLO]:
    models: dict[str, YOLO] = {}
    for axis, w in WEIGHTS.items():
        if not w.exists():
            raise FileNotFoundError(f"未找到 {axis} 轴权重: {w}")
        print(f"[LF] Loading {axis}-axis model from {w}")
        models[axis] = YOLO(str(w))
    return models


def get_test_images(axis: str) -> list[Path]:
    root = DATA_ROOTS[axis]
    test_root = root / "test"
    if not test_root.exists():
        raise FileNotFoundError(f"{axis} 轴 test 目录不存在: {test_root}")
    paths: list[Path] = []
    for lab_dir in sorted(test_root.iterdir()):
        if not lab_dir.is_dir():
            continue
        for img in sorted(lab_dir.glob("*.png")):
            paths.append(img)
    return paths


def predict_probs(model: YOLO, img_path: Path) -> np.ndarray:
    # Ultralytics 分类预测，返回 1×num_classes 概率
    res = model.predict(source=str(img_path), imgsz=128, verbose=False)
    if not res:
        raise RuntimeError(f"模型未返回结果: {img_path}")
    r = res[0]
    if r.probs is None:
        raise RuntimeError(f"结果中缺少 probs: {img_path}")
    # r.probs.data 是 1D 张量
    return r.probs.data.cpu().numpy().astype(np.float32)


def main() -> None:
    models = load_models()

    # 以 X 轴 test 集为基准，按文件名在 Y/Z 中寻找对应
    x_tests = get_test_images("x")
    print(f"[LF] X-axis test images: {len(x_tests)}")

    correct_x = 0
    correct_y = 0
    correct_z = 0
    correct_fused = 0

    for img_x in x_tests:
        # 标签从父目录名解析
        lab = int(img_x.parent.name)

        # 文件名模板：spec_x_<lab>_<idx>.png，对应 Y/Z 轴需要改前缀
        name_x = img_x.name  # e.g. spec_x_0_00256.png
        if not name_x.startswith("spec_x_"):
            raise ValueError(f"意外的 X 轴文件名格式: {name_x}")

        name_y = "spec_y_" + name_x[len("spec_x_") :]
        name_z = "spec_z_" + name_x[len("spec_x_") :]

        img_y = DATA_ROOTS["y"] / "test" / str(lab) / name_y
        img_z = DATA_ROOTS["z"] / "test" / str(lab) / name_z

        if not img_y.exists() or not img_z.exists():
            raise FileNotFoundError(f"未找到对应 Y/Z 轴图像: {img_y}, {img_z}")

        px = predict_probs(models["x"], img_x)
        py = predict_probs(models["y"], img_y)
        pz = predict_probs(models["z"], img_z)

        # 单轴预测
        pred_x = int(px.argmax())
        pred_y = int(py.argmax())
        pred_z = int(pz.argmax())

        # 简单平均融合
        pfused = (px + py + pz) / 3.0
        pred_fused = int(pfused.argmax())

        if pred_x == lab:
            correct_x += 1
        if pred_y == lab:
            correct_y += 1
        if pred_z == lab:
            correct_z += 1
        if pred_fused == lab:
            correct_fused += 1

    n = len(x_tests)
    acc_x = correct_x / n if n else 0.0
    acc_y = correct_y / n if n else 0.0
    acc_z = correct_z / n if n else 0.0
    acc_fused = correct_fused / n if n else 0.0

    print("\n[LF] Test accuracy (单模型 vs late fusion)：")
    print(f"  X-axis:   {acc_x:.4f}")
    print(f"  Y-axis:   {acc_y:.4f}")
    print(f"  Z-axis:   {acc_z:.4f}")
    print(f"  Fused(均值): {acc_fused:.4f}")


if __name__ == "__main__":
    main()

