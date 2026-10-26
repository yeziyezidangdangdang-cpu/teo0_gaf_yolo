"""
Export confusion matrix (counts + normalized) for HUST STFT-RGB classifier.

This is for generating Table-1 (GAF-RGB vs STFT-RGB) numbers.

Expected assets:
- Weights: YOLOv8/runs/classify/runs/hust_spec/hust_spec_cls/weights/best.pt
- Dataset: D:/TEP_2_repro/gaf_yolo/yolo_datasets/spec_cls_strict/test/<label>/*.png

Outputs:
- YOLOv8/runs/hust_spec/eval_spec/confusion_matrix_counts.csv
- YOLOv8/runs/hust_spec/eval_spec/confusion_matrix_normalized.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO

import repro_paths as pp

PROJECT_ROOT = pp.YOLO_ROOT

# 稳态时序隔离划分训练权重（Log 2 / 3.3 主线）；勿用 spec_cls 随机划分权重。
_WEIGHT_CANDIDATES = [
    PROJECT_ROOT / "runs" / "classify" / "runs" / "hust_spec" / "hust_spec_cls" / "weights" / "best.pt",
    PROJECT_ROOT / "runs" / "classify" / "hust_spec" / "hust_spec_cls" / "weights" / "best.pt",
]

_TEST_CANDIDATES = [
    pp.YOLO_SPEC_STRICT / "test",
]


def _first_existing(candidates: list[Path]) -> Path:
    for p in candidates:
        if p.is_file() or p.is_dir():
            return p
    raise FileNotFoundError("None of:\n" + "\n".join(str(p) for p in candidates))
OUT_DIR = PROJECT_ROOT / "runs" / "hust_spec" / "eval_spec"
NUM_CLASSES = 4


def main() -> None:
    weights = _first_existing(_WEIGHT_CANDIDATES)
    test_root = _first_existing(_TEST_CANDIDATES)
    if not weights.is_file():
        raise FileNotFoundError(f"weights not found: {weights}")

    model = YOLO(str(weights))
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for lab_dir in sorted(test_root.iterdir()):
        if not lab_dir.is_dir():
            continue
        y_true = int(lab_dir.name)
        for img in sorted(lab_dir.glob("*.png")):
            res = model.predict(source=str(img), imgsz=128, verbose=False)
            if not res or res[0].probs is None:
                raise RuntimeError(f"no probs for {img}")
            y_pred = int(res[0].probs.top1)
            cm[y_true, y_pred] += 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savetxt(OUT_DIR / "confusion_matrix_counts.csv", cm, fmt="%d", delimiter=",")
    cm_norm = cm.astype(np.float64)
    row_sum = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sum, out=np.zeros_like(cm_norm), where=row_sum != 0)
    np.savetxt(OUT_DIR / "confusion_matrix_normalized.csv", cm_norm, fmt="%.6f", delimiter=",")

    acc = float(np.trace(cm) / max(cm.sum(), 1))
    rec = np.divide(np.diag(cm), cm.sum(axis=1), out=np.zeros(NUM_CLASSES, dtype=np.float64), where=cm.sum(axis=1) != 0)
    print(f"[OK] Saved -> {OUT_DIR}")
    print(f"overall_acc(%): {acc * 100:.2f}")
    print("per_class_recall(%): " + ", ".join(f"{r * 100:.2f}" for r in rec))


if __name__ == "__main__":
    main()

