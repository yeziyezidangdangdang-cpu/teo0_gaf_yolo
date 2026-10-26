"""
Weighted late fusion evaluation for HUST single-axis STFT classifiers.

在 eval_hust_spec_late_fusion.py 基础上增加「加权融合」：
- 仍然使用 X/Y/Z 三个单轴模型的 best.pt
- 对 test 集同一片段，计算：
  - 单轴预测精度（X/Y/Z）
  - 简单平均融合精度
  - 加权融合精度：p_fused = wx * px + wy * py + wz * pz

权重可通过命令行指定：
    python eval_hust_spec_late_fusion_weighted.py --wx 0.5 --wy 0.3 --wz 0.2

若未归一化，脚本会自动按 wx+wy+wz 归一化。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_ROOTS = {
    "x": PROJECT_ROOT.parent / "gaf_yolo" / "yolo_datasets" / "hust_spec_x_cls",
    "y": PROJECT_ROOT.parent / "gaf_yolo" / "yolo_datasets" / "hust_spec_y_cls",
    "z": PROJECT_ROOT.parent / "gaf_yolo" / "yolo_datasets" / "hust_spec_z_cls",
}

WEIGHTS_PATHS = {
    "x": PROJECT_ROOT / "runs" / "classify" / "runs" / "hust_spec_x" / "hust_spec_x_cls" / "weights" / "best.pt",
    "y": PROJECT_ROOT / "runs" / "classify" / "runs" / "hust_spec_y" / "hust_spec_y_cls" / "weights" / "best.pt",
    "z": PROJECT_ROOT / "runs" / "classify" / "runs" / "hust_spec_z" / "hust_spec_z_cls" / "weights" / "best.pt",
}


def load_models() -> dict[str, YOLO]:
    models: dict[str, YOLO] = {}
    for axis, w in WEIGHTS_PATHS.items():
        if not w.exists():
            raise FileNotFoundError(f"未找到 {axis} 轴权重: {w}")
        print(f"[LF-W] Loading {axis}-axis model from {w}")
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
    res = model.predict(source=str(img_path), imgsz=128, verbose=False)
    if not res:
        raise RuntimeError(f"模型未返回结果: {img_path}")
    r = res[0]
    if r.probs is None:
        raise RuntimeError(f"结果中缺少 probs: {img_path}")
    return r.probs.data.cpu().numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Weighted late fusion for HUST single-axis STFT classifiers")
    parser.add_argument("--wx", type=float, default=1.0, help="weight for X-axis model (default 1.0)")
    parser.add_argument("--wy", type=float, default=1.0, help="weight for Y-axis model (default 1.0)")
    parser.add_argument("--wz", type=float, default=1.0, help="weight for Z-axis model (default 1.0)")
    args = parser.parse_args()

    # 归一化权重
    w = np.array([args.wx, args.wy, args.wz], dtype=np.float32)
    s = float(w.sum())
    if s <= 0:
        raise ValueError(f"权重之和必须大于 0，当前: {w.tolist()}")
    w /= s
    wx, wy, wz = w.tolist()
    print(f"[LF-W] 使用加权融合权重: wx={wx:.3f}, wy={wy:.3f}, wz={wz:.3f} (已归一化)")

    models = load_models()

    # 以 X 轴 test 集为基准
    x_tests = get_test_images("x")
    print(f"[LF-W] X-axis test images: {len(x_tests)}")

    correct_x = correct_y = correct_z = 0
    correct_mean = correct_w = 0

    for img_x in x_tests:
        lab = int(img_x.parent.name)

        name_x = img_x.name  # spec_x_0_00256.png
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

        pred_x = int(px.argmax())
        pred_y = int(py.argmax())
        pred_z = int(pz.argmax())

        p_mean = (px + py + pz) / 3.0
        pred_mean = int(p_mean.argmax())

        p_w = wx * px + wy * py + wz * pz
        pred_w = int(p_w.argmax())

        if pred_x == lab:
            correct_x += 1
        if pred_y == lab:
            correct_y += 1
        if pred_z == lab:
            correct_z += 1
        if pred_mean == lab:
            correct_mean += 1
        if pred_w == lab:
            correct_w += 1

    n = len(x_tests)
    def acc(c: int) -> float:
        return c / n if n else 0.0

    print("\n[LF-W] Test accuracy：")
    print(f"  X-axis:        {acc(correct_x):.4f}")
    print(f"  Y-axis:        {acc(correct_y):.4f}")
    print(f"  Z-axis:        {acc(correct_z):.4f}")
    print(f"  Fused(mean):   {acc(correct_mean):.4f}")
    print(f"  Fused(weight): {acc(correct_w):.4f}")


if __name__ == "__main__":
    main()

