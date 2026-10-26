"""
Compute Table-1 numbers for HUST:
- Multi-axis GAF-RGB (dual-branch attention fusion): from saved confusion matrix counts.
- Multi-axis STFT-RGB (late fusion X/Y/Z): run inference on test set and compute confusion matrix.

Outputs overall accuracy and per-class recall (%) for labels 0..3.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent

# ---------- GAF-RGB (attention fusion) ----------
GAF_CM_COUNTS = PROJECT_ROOT / "runs" / "hust_gaf_cls" / "eval_attn" / "confusion_matrix_attn_counts.csv"

# ---------- STFT-RGB (late fusion) ----------
DATA_ROOTS = {
    "x": PROJECT_ROOT.parent / "gaf_yolo" / "yolo_datasets" / "hust_spec_x_cls",
    "y": PROJECT_ROOT.parent / "gaf_yolo" / "yolo_datasets" / "hust_spec_y_cls",
    "z": PROJECT_ROOT.parent / "gaf_yolo" / "yolo_datasets" / "hust_spec_z_cls",
}

WEIGHTS = {
    "x": PROJECT_ROOT / "runs" / "classify" / "runs" / "hust_spec_x" / "hust_spec_x_cls" / "weights" / "best.pt",
    "y": PROJECT_ROOT / "runs" / "classify" / "runs" / "hust_spec_y" / "hust_spec_y_cls" / "weights" / "best.pt",
    "z": PROJECT_ROOT / "runs" / "classify" / "runs" / "hust_spec_z" / "hust_spec_z_cls" / "weights" / "best.pt",
}

NUM_CLASSES = 4


def _acc_and_recall(cm: np.ndarray) -> tuple[float, list[float]]:
    cm = cm.astype(np.float64)
    total = float(cm.sum())
    overall = float(np.trace(cm) / total) if total > 0 else 0.0
    recalls: list[float] = []
    for i in range(NUM_CLASSES):
        row = float(cm[i, :].sum())
        recalls.append(float(cm[i, i] / row) if row > 0 else 0.0)
    return overall, recalls


def _read_cm_counts(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.loadtxt(str(path), delimiter=",").astype(np.int64)


def _predict_probs(model: YOLO, img_path: Path) -> np.ndarray:
    res = model.predict(source=str(img_path), imgsz=128, verbose=False)
    if not res or res[0].probs is None:
        raise RuntimeError(f"No probs returned for {img_path}")
    return res[0].probs.data.cpu().numpy().astype(np.float32)


def _collect_test_images(axis: str) -> list[Path]:
    test_root = DATA_ROOTS[axis] / "test"
    if not test_root.is_dir():
        raise FileNotFoundError(test_root)
    paths: list[Path] = []
    for lab_dir in sorted(test_root.iterdir()):
        if not lab_dir.is_dir():
            continue
        paths.extend(sorted(lab_dir.glob("*.png")))
    return paths


def compute_stft_late_fusion_cm() -> np.ndarray:
    for axis, w in WEIGHTS.items():
        if not w.is_file():
            raise FileNotFoundError(f"Missing {axis}-axis weights: {w}")

    models = {k: YOLO(str(v)) for k, v in WEIGHTS.items()}
    x_tests = _collect_test_images("x")
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for img_x in x_tests:
        y_true = int(img_x.parent.name)
        name_x = img_x.name
        if not name_x.startswith("spec_x_"):
            raise ValueError(f"Unexpected filename: {name_x}")
        name_y = "spec_y_" + name_x[len("spec_x_") :]
        name_z = "spec_z_" + name_x[len("spec_x_") :]
        img_y = DATA_ROOTS["y"] / "test" / str(y_true) / name_y
        img_z = DATA_ROOTS["z"] / "test" / str(y_true) / name_z
        if not img_y.is_file() or not img_z.is_file():
            raise FileNotFoundError(f"Missing paired images: {img_y} / {img_z}")

        px = _predict_probs(models["x"], img_x)
        py = _predict_probs(models["y"], img_y)
        pz = _predict_probs(models["z"], img_z)
        pf = (px + py + pz) / 3.0
        y_pred = int(pf.argmax())
        cm[y_true, y_pred] += 1

    return cm


def main() -> None:
    # GAF-RGB attention fusion
    gaf_cm = _read_cm_counts(GAF_CM_COUNTS)
    gaf_overall, gaf_recalls = _acc_and_recall(gaf_cm)

    # STFT-RGB late fusion
    stft_cm = compute_stft_late_fusion_cm()
    stft_overall, stft_recalls = _acc_and_recall(stft_cm)

    def fmt(x: float) -> str:
        return f"{x * 100:.2f}"

    print("=== Table-1 numbers (HUST) ===")
    print("[GAF-RGB dual-branch + attention] (from confusion matrix counts)")
    print(f"overall_acc(%): {fmt(gaf_overall)}")
    print("per_class_recall(%): " + ", ".join(fmt(r) for r in gaf_recalls))
    print("")
    print("[STFT-RGB 3-axis late fusion] (computed from test inference)")
    print(f"overall_acc(%): {fmt(stft_overall)}")
    print("per_class_recall(%): " + ", ".join(fmt(r) for r in stft_recalls))
    print("")
    print("Confusion matrices saved in memory only. If you need CSV outputs, tell me.")


if __name__ == "__main__":
    main()

