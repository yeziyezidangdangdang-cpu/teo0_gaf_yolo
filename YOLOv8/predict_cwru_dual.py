"""
Dual-branch inference script for CWRU GASF + GADF classifiers (方案二：双模型后端融合).

默认使用与 eval_cwru_dual.py 一致的「最后一组完整配对」：cwru_gasf3 + cwru_gadf3。
可选：python predict_cwru_dual.py --run 2 --gasf-img ... --gadf-img ...
"""

import argparse
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from cwru_paths import CWRU_V2_IMGSZ, branch_weight_path

DEFAULT_RUN = 1  # 1=固定一对 cwru_gasf/cwru_gadf（与训练 exist_ok 覆盖一致）

CLASS_NAMES = {
    0: "Normal",
    1: "Inner race fault",
    2: "Outer race fault",
    3: "Rolling element fault",
}


def get_weights_paths(run: int):
    return branch_weight_path("gasf", run), branch_weight_path("gadf", run)


def load_models(run: int = DEFAULT_RUN):
    gasf_pt, gadf_pt = get_weights_paths(run)
    if not gasf_pt.is_file() or not gadf_pt.is_file():
        raise FileNotFoundError(f"Need both weights: {gasf_pt}, {gadf_pt}")
    return YOLO(str(gasf_pt)), YOLO(str(gadf_pt))


def predict_pair(gasf_model: YOLO, gadf_model: YOLO, gasf_img: Path, gadf_img: Path):
    # Single‑image prediction for each branch
    r1 = gasf_model(str(gasf_img), imgsz=CWRU_V2_IMGSZ, verbose=False)[0]
    r2 = gadf_model(str(gadf_img), imgsz=CWRU_V2_IMGSZ, verbose=False)[0]

    probs1 = r1.probs.data.cpu().numpy()
    probs2 = r2.probs.data.cpu().numpy()

    probs = (probs1 + probs2) / 2.0
    cls_id = int(np.argmax(probs))
    score = float(probs[cls_id])

    return cls_id, score, probs


def main():
    parser = argparse.ArgumentParser(description="CWRU dual-branch GASF+GADF classifier")
    parser.add_argument("--run", type=int, default=DEFAULT_RUN, help=f"Weights run index (default {DEFAULT_RUN} = gasf3+gadf3)")
    parser.add_argument("--gasf-img", type=str, required=True, help="Path to GASF image")
    parser.add_argument("--gadf-img", type=str, required=True, help="Path to GADF image")
    args = parser.parse_args()

    gasf_path = Path(args.gasf_img)
    gadf_path = Path(args.gadf_img)

    if not gasf_path.is_file():
        raise FileNotFoundError(f"GASF image not found: {gasf_path}")
    if not gadf_path.is_file():
        raise FileNotFoundError(f"GADF image not found: {gadf_path}")

    gasf_model, gadf_model = load_models(args.run)
    cls_id, score, probs = predict_pair(gasf_model, gadf_model, gasf_path, gadf_path)

    name = CLASS_NAMES.get(cls_id, str(cls_id))
    print(f"Predicted class: {cls_id} ({name}), score={score:.4f}")
    print("Probabilities:")
    for i in range(len(probs)):
        cname = CLASS_NAMES.get(i, str(i))
        print(f"  {i} ({cname}): {probs[i]:.4f}")


if __name__ == "__main__":
    main()

