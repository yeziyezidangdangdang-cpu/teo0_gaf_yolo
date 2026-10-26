"""
评估 HUST GAF 双分支（GASF-RGB + GADF-RGB）+ 分支注意力融合在 test 集上的表现。

对比基线：简单平均 p_fused = (p_gasf + p_gadf) / 2
本脚本：使用训练好的注意力模块做自适应加权融合。

数据与权重：
- YOLO: runs/classify/runs/hust_gaf_rgb_cls/hust_gasf_rgb/weights/best.pt
        runs/classify/runs/hust_gaf_rgb_cls/hust_gadf_rgb/weights/best.pt
- 注意力: runs/hust_gaf_cls/branch_attn/branch_attn.pt (或 branch_attn_env.pt)

运行示例：
    cd D:\TEP_2_repro\YOLOv8
    python eval_hust_gaf_dual_attn.py
    python eval_hust_gaf_dual_attn.py --use-envelope
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from ultralytics import YOLO


YOLO_ROOT = Path(__file__).resolve().parent
RUNS_DIR = YOLO_ROOT / "runs" / "classify" / "hust_gaf_rgb_cls"
GASF_DATA_ROOT = Path(r"D:\TEP_2_repro\gaf_yolo\yolo_datasets\hust_gasf_rgb_cls_strict")
GADF_DATA_ROOT = Path(r"D:\TEP_2_repro\gaf_yolo\yolo_datasets\hust_gadf_rgb_cls_strict")
GASF_DATA_ROOT_ENV = Path(r"D:\TEP_2_repro\gaf_yolo\yolo_datasets\hust_gasf_rgb_env_cls")
GADF_DATA_ROOT_ENV = Path(r"D:\TEP_2_repro\gaf_yolo\yolo_datasets\hust_gadf_rgb_env_cls")

NUM_CLASSES = 4
CLASS_NAMES = {0: "Healthy", 1: "Ball", 2: "Inner race", 3: "Outer race"}
ATTN_DIR = YOLO_ROOT / "runs" / "hust_gaf_cls" / "branch_attn"


def get_weights_paths() -> Tuple[Path, Path]:
    gasf = RUNS_DIR / "hust_gasf_rgb" / "weights" / "best.pt"
    gadf = RUNS_DIR / "hust_gadf_rgb" / "weights" / "best.pt"
    return gasf, gadf


def load_yolo_models() -> Tuple[YOLO, YOLO]:
    gasf_pt, gadf_pt = get_weights_paths()
    if not gasf_pt.is_file():
        raise FileNotFoundError(f"GASF-RGB weights not found: {gasf_pt}")
    if not gadf_pt.is_file():
        raise FileNotFoundError(f"GADF-RGB weights not found: {gadf_pt}")
    return YOLO(str(gasf_pt)), YOLO(str(gadf_pt))


class BranchAttention(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, hidden_dim: int = 16) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_classes * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, probs_gasf: torch.Tensor, probs_gadf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat([probs_gasf, probs_gadf], dim=1)
        logits = self.mlp(z)
        attn = torch.softmax(logits, dim=1)
        w_gasf = attn[:, 0].unsqueeze(1)
        w_gadf = attn[:, 1].unsqueeze(1)
        p_fused = w_gasf * probs_gasf + w_gadf * probs_gadf
        p_fused = p_fused / (p_fused.sum(dim=1, keepdim=True) + 1e-8)
        return p_fused, attn


def load_attention_model(use_envelope: bool, device: torch.device) -> BranchAttention:
    ckpt_name = "branch_attn_env.pt" if use_envelope else "branch_attn.pt"
    ckpt_path = ATTN_DIR / ckpt_name
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Attention checkpoint not found: {ckpt_path}\nRun: python train_hust_gaf_dual_attn.py first")
    state = torch.load(ckpt_path, map_location=device)
    model = BranchAttention(
        num_classes=int(state.get("num_classes", NUM_CLASSES)),
        hidden_dim=int(state.get("hidden_dim", 16)),
    ).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


def forward_yolo_probs(model: YOLO, img_paths: List[str], device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        results = model(img_paths, imgsz=128, verbose=False)
    probs_list = []
    for r in results:
        p = r.probs.data
        if p.dim() == 1:
            p = p.unsqueeze(0)
        probs_list.append(p)
    return torch.cat(probs_list, dim=0).to(device)


def collect_test_pairs(root_gasf: Path, root_gadf: Path) -> List[Tuple[Path, Path, int]]:
    samples = []
    gasf_test = root_gasf / "test"
    gadf_test = root_gadf / "test"
    if not gasf_test.is_dir() or not gadf_test.is_dir():
        raise FileNotFoundError(f"Missing test dir: {gasf_test} or {gadf_test}")

    for lab in range(NUM_CLASSES):
        gasf_dir = gasf_test / str(lab)
        gadf_dir = gadf_test / str(lab)
        if not gasf_dir.is_dir() or not gadf_dir.is_dir():
            continue
        gasf_imgs = sorted([p for p in gasf_dir.glob("*.png") if p.is_file()])
        gadf_map = {p.name: p for p in gadf_dir.glob("*.png") if p.is_file()}
        for gasf_img in gasf_imgs:
            gadf_name = gasf_img.name.replace("gasf_rgb_", "gadf_rgb_", 1)
            gadf_img = gadf_map.get(gadf_name)
            if gadf_img is not None:
                samples.append((gasf_img, gadf_img, lab))

    if not samples:
        raise RuntimeError(f"No paired test samples in {root_gasf} / {root_gadf}")
    return samples


def evaluate(gasf_m, gadf_m, attn_m, samples: List[Tuple[Path, Path, int]], device, batch_size=32):
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    batch_gasf, batch_gadf, batch_lab = [], [], []

    def flush():
        nonlocal cm, batch_gasf, batch_gadf, batch_lab
        if not batch_gasf:
            return
        labels = torch.tensor(batch_lab, dtype=torch.long, device=device)
        p_gasf = forward_yolo_probs(gasf_m, batch_gasf, device)
        p_gadf = forward_yolo_probs(gadf_m, batch_gadf, device)
        with torch.no_grad():
            p_fused, _ = attn_m(p_gasf, p_gadf)
        preds = p_fused.argmax(dim=1).cpu().numpy()
        labs = labels.cpu().numpy()
        for t, p in zip(labs, preds):
            cm[t, p] += 1
        batch_gasf, batch_gadf, batch_lab = [], [], []

    for gasf_path, gadf_path, lab in samples:
        batch_gasf.append(str(gasf_path))
        batch_gadf.append(str(gadf_path))
        batch_lab.append(lab)
        if len(batch_gasf) >= batch_size:
            flush()
    flush()

    return cm


def save_confusion_matrix(cm: np.ndarray, out_dir: Path, title_suffix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / "confusion_matrix_attn_counts.csv", cm, fmt="%d", delimiter=",")
    row_sums = cm.sum(axis=1, keepdims=True)
    safe = np.where(row_sums == 0, 1, row_sums)
    cm_norm = cm.astype(float) / safe
    np.savetxt(out_dir / "confusion_matrix_attn_normalized.csv", cm_norm, fmt="%.4f", delimiter=",")

    fig, ax = plt.subplots(figsize=(5, 4))
    plot_mat = np.nan_to_num(cm_norm, nan=0.0, posinf=0.0, neginf=0.0)
    im = ax.imshow(plot_mat, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels([f"{i}\n{CLASS_NAMES[i]}" for i in range(NUM_CLASSES)])
    ax.set_yticklabels([f"{i}:{CLASS_NAMES[i]}" for i in range(NUM_CLASSES)])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"HUST GAF dual-branch + attention {title_suffix}")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = plot_mat[i, j]
            ax.text(j, i, f"{val:.2f}" if row_sums[i, 0] > 0 else "-", ha="center", va="center",
                    color="white" if val > 0.5 else "black")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix_attn.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate HUST GAF dual-branch with attention fusion")
    parser.add_argument("--use-envelope", action="store_true", help="Use envelope GAF-RGB and branch_attn_env.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.use_envelope and GASF_DATA_ROOT_ENV.is_dir() and GADF_DATA_ROOT_ENV.is_dir():
        gasf_root, gadf_root = GASF_DATA_ROOT_ENV, GADF_DATA_ROOT_ENV
        title_suffix = "(attention, envelope GAF-RGB)"
    else:
        gasf_root, gadf_root = GASF_DATA_ROOT, GADF_DATA_ROOT
        title_suffix = "(attention, raw GAF-RGB)"

    gasf_model, gadf_model = load_yolo_models()
    attn_model = load_attention_model(args.use_envelope, device)

    samples = collect_test_pairs(gasf_root, gadf_root)
    print(f"[eval_hust_gaf_dual_attn] Test samples: {len(samples)}")

    cm = evaluate(gasf_model, gadf_model, attn_model, samples, device)
    total = cm.sum()
    overall_acc = cm.trace() / total if total > 0 else 0.0

    print(f"Total: {total}, Overall accuracy (attention): {overall_acc:.4f}")
    for lab in range(NUM_CLASSES):
        n = cm[lab, :].sum()
        acc = cm[lab, lab] / n if n > 0 else 0.0
        print(f"  Class {lab} ({CLASS_NAMES[lab]}): {acc:.4f} ({cm[lab, lab]}/{n})")

    out_dir = YOLO_ROOT / "runs" / "hust_gaf_cls" / ("eval_attn_env" if args.use_envelope else "eval_attn")
    save_confusion_matrix(cm, out_dir, title_suffix)
    print(f"Confusion matrix saved to {out_dir}")


if __name__ == "__main__":
    main()
