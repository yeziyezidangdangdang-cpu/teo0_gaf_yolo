"""
2.0_prepare_yolov8_datasets_cwru_gaf.py

基于 CWRU GAF 图像整理 YOLOv8 分类数据集（GASF/GADF 两个分支）：

输入：
- 原始 GAF：
  - data/images/gasf/<label>/*.png
  - data/images/gadf/<label>/*.png
- 可选（若存在）增强 SA-GAF：
  - data/images_enhanced/gasf/<label>/*.png
  - data/images_enhanced/gadf/<label>/*.png

输出：
- yolo_datasets/gasf_cls/{train,val,test}/<label>/*.png
- yolo_datasets/gadf_cls/{train,val,test}/<label>/*.png
- 若检测到增强图像，则额外输出：
  - yolo_datasets_enhanced/gasf_cls/{train,val,test}/<label>/*.png
  - yolo_datasets_enhanced/gadf_cls/{train,val,test}/<label>/*.png
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Dict
from concurrent.futures import ThreadPoolExecutor, as_completed


RANDOM_SEED = 42
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}

# I/O 优化：
# - Windows/同盘场景，硬链接会比逐文件 copy 快非常多（也能节省空间）
# - 若硬链接失败（不同磁盘/权限），自动退回 copy2
LINK_MODE = "hardlink"  # "hardlink" | "copy"
MAX_WORKERS = 8


def _split_counts(n: int) -> Dict[str, int]:
    # Ensure val/test are non-zero when possible (n>=3)
    if n <= 0:
        return {"train": 0, "val": 0, "test": 0}
    if n == 1:
        return {"train": 1, "val": 0, "test": 0}
    if n == 2:
        return {"train": 1, "val": 1, "test": 0}

    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])
    n_test = n - n_train - n_val
    if n_val < 1:
        n_val = 1
    if n_test < 1:
        n_test = 1
    n_train = max(0, n - n_val - n_test)
    return {"train": n_train, "val": n_val, "test": n_test}


def _safe_link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if LINK_MODE == "hardlink":
        try:
            # hardlink requires same filesystem
            dst.hardlink_to(src)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)


def _bulk_materialize(pairs: list[tuple[Path, Path]], desc: str) -> None:
    if not pairs:
        return
    done = 0
    total = len(pairs)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(_safe_link_or_copy, s, d) for s, d in pairs]
        for f in as_completed(futs):
            _ = f.result()
            done += 1
            if done % 500 == 0 or done == total:
                print(f"[CWRU-GAF] {desc}: {done}/{total}")


def prepare_one_branch(images_root: Path, out_root: Path, seed: int = RANDOM_SEED) -> None:
    if not images_root.exists():
        raise FileNotFoundError(f"源图像目录不存在: {images_root}")

    if out_root.exists():
        shutil.rmtree(out_root)

    rng = random.Random(seed)

    label_dirs = sorted([p for p in images_root.iterdir() if p.is_dir()])
    labels = [p.name for p in label_dirs]
    for split in SPLIT_RATIOS.keys():
        for lab in labels:
            (out_root / split / lab).mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Dict[str, int]] = {}

    for lab_dir in label_dirs:
        lab = lab_dir.name
        imgs = sorted([p for p in lab_dir.glob("*.png") if p.is_file()])
        if not imgs:
            print(f"[CWRU-GAF] WARN: no images under {lab_dir}")
            continue

        rng.shuffle(imgs)
        counts = _split_counts(len(imgs))
        n_train, n_val = counts["train"], counts["val"]

        splits = {
            "train": imgs[:n_train],
            "val": imgs[n_train : n_train + n_val],
            "test": imgs[n_train + n_val :],
        }
        summary[lab] = {k: len(v) for k, v in splits.items()}

        for split_name, files in splits.items():
            dst_dir = out_root / split_name / lab
            pairs = [(src, dst_dir / src.name) for src in files]
            _bulk_materialize(pairs, f"{out_root.name}/{split_name}/class_{lab}")

    print(f"\n[CWRU-GAF] Prepared dataset: {images_root} -> {out_root}")
    for lab in labels:
        s = summary.get(lab)
        if not s:
            print(f"  class {lab}: no images")
        else:
            print(f"  class {lab}: train={s['train']}, val={s['val']}, test={s['test']}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    # Raw GAF
    gasf_root = project_root / "data" / "images" / "gasf"
    gadf_root = project_root / "data" / "images" / "gadf"
    out_root = project_root / "yolo_datasets"
    print(f"[CWRU-GAF] link_mode={LINK_MODE}, workers={MAX_WORKERS}")
    prepare_one_branch(gasf_root, out_root / "gasf_cls")
    prepare_one_branch(gadf_root, out_root / "gadf_cls")

    # Enhanced SA-GAF (optional)
    gasf_enh_root = project_root / "data" / "images_enhanced" / "gasf"
    gadf_enh_root = project_root / "data" / "images_enhanced" / "gadf"
    if gasf_enh_root.exists() and gadf_enh_root.exists():
        out_root_enh = project_root / "yolo_datasets_enhanced"
        prepare_one_branch(gasf_enh_root, out_root_enh / "gasf_cls")
        prepare_one_branch(gadf_enh_root, out_root_enh / "gadf_cls")
        print(f"[CWRU-GAF] Enhanced datasets also prepared under: {out_root_enh}")
    else:
        print("[CWRU-GAF] Enhanced images not found; skipped enhanced dataset preparation.")


if __name__ == "__main__":
    main()

