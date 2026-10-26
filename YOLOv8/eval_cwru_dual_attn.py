"""
Evaluate dual-branch (GASF + GADF) classifiers with branch-attentive fusion on the CWRU test set.

对比基线：
- 基线：eval_cwru_dual.py 中使用两分支概率简单平均
- 本脚本：加载训练好的分支注意力融合模块，对每个样本做自适应加权融合

数据与权重：
- YOLO GASF / GADF 权重：与 train_cwru_dual.py 一致（runs/classify/runs/cwru_cls 下）
- 注意力模块权重：runs/cwru_cls/branch_attn/branch_attn.pt
  或增强版 runs/cwru_cls/branch_attn/branch_attn_enh.pt
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from ultralytics import YOLO

from cwru_paths import branch_weight_path


DEFAULT_RUN = 1

GASF_DATA_ROOT = Path(r"D:\TEP_2_repro\gaf_yolo\yolo_datasets\gasf_cls_strict")
GADF_DATA_ROOT = Path(r"D:\TEP_2_repro\gaf_yolo\yolo_datasets\gadf_cls_strict")

GASF_DATA_ROOT_ENH = Path(r"D:\TEP_2_repro\gaf_yolo\yolo_datasets_enhanced\gasf_cls_strict")
GADF_DATA_ROOT_ENH = Path(r"D:\TEP_2_repro\gaf_yolo\yolo_datasets_enhanced\gadf_cls_strict")

NUM_CLASSES = 4
LABELS = [0, 1, 2, 3]
CLASS_NAMES = {
    0: "Normal",
    1: "Inner race fault",
    2: "Outer race fault",
    3: "Rolling element fault",
}

ATTN_RUN_DIR = Path(r"D:\TEP_2_repro\YOLOv8\runs\cwru_cls\branch_attn")


def get_yolo_weights_paths(run: int, use_enhanced: bool = False) -> Tuple[Path, Path]:
    if use_enhanced:
        return branch_weight_path("gasf_enh", run), branch_weight_path("gadf_enh", run)
    return branch_weight_path("gasf", run), branch_weight_path("gadf", run)


def load_yolo_models(run: int = DEFAULT_RUN) -> Tuple[YOLO, YOLO]:
    gasf_pt, gadf_pt = get_yolo_weights_paths(run)
    if not gasf_pt.is_file():
        raise FileNotFoundError(f"GASF weights not found: {gasf_pt}")
    if not gadf_pt.is_file():
        raise FileNotFoundError(f"GADF weights not found: {gadf_pt}")
    gasf_model = YOLO(str(gasf_pt))
    gadf_model = YOLO(str(gadf_pt))
    gasf_model.model.eval()
    gadf_model.model.eval()
    return gasf_model, gadf_model


class BranchAttention(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, hidden_dim: int = 16) -> None:
        super().__init__()
        in_dim = num_classes * 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, probs_gasf: torch.Tensor, probs_gadf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat([probs_gasf, probs_gadf], dim=1)  # (B, 2C)
        logits = self.mlp(z)  # (B, 2)
        attn = torch.softmax(logits, dim=1)  # (B, 2)
        w_gasf = attn[:, 0].unsqueeze(1)
        w_gadf = attn[:, 1].unsqueeze(1)
        p_fused = w_gasf * probs_gasf + w_gadf * probs_gadf
        p_fused = p_fused / (p_fused.sum(dim=1, keepdim=True) + 1e-8)
        return p_fused, attn


def load_attention_model(use_enhanced: bool, device: torch.device) -> BranchAttention:
    ckpt_name = "branch_attn_enh.pt" if use_enhanced else "branch_attn.pt"
    ckpt_path = ATTN_RUN_DIR / ckpt_name
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Attention model checkpoint not found: {ckpt_path}")

    # 我们自己保存的纯 state dict，安全可控；显式设置 weights_only=False 避免未来版本默认变更的警告。
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    num_classes = int(state.get("num_classes", NUM_CLASSES))
    hidden_dim = int(state.get("hidden_dim", 16))

    model = BranchAttention(num_classes=num_classes, hidden_dim=hidden_dim).to(device)
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
    probs = torch.cat(probs_list, dim=0)
    return probs.to(device)


def collect_test_pairs(root_gasf: Path, root_gadf: Path) -> List[Tuple[Path, Path, int]]:
    samples: List[Tuple[Path, Path, int]] = []
    gasf_test_root = root_gasf / "test"
    gadf_test_root = root_gadf / "test"

    if not gasf_test_root.is_dir() or not gadf_test_root.is_dir():
        raise FileNotFoundError(f"缺少 test 目录: {gasf_test_root} 或 {gadf_test_root}")

    for lab in LABELS:
        lab_str = str(lab)
        gasf_dir = gasf_test_root / lab_str
        gadf_dir = gadf_test_root / lab_str
        if not gasf_dir.is_dir() or not gadf_dir.is_dir():
            continue

        gasf_imgs = sorted([p for p in gasf_dir.glob("*.png") if p.is_file()])
        gadf_map = {p.name: p for p in gadf_dir.glob("*.png") if p.is_file()}

        for gasf_img in gasf_imgs:
            gadf_name = gasf_img.name.replace("gasf_", "gadf_", 1)
            gadf_img = gadf_map.get(gadf_name)
            if gadf_img is None:
                continue
            samples.append((gasf_img, gadf_img, lab))

    if not samples:
        raise RuntimeError(f"{root_gasf} / {root_gadf} 下 test split 未找到任何可配对样本")
    return samples


def evaluate(
    gasf_model: YOLO,
    gadf_model: YOLO,
    attn_model: BranchAttention,
    samples: List[Tuple[Path, Path, int]],
    device: torch.device,
) -> Tuple[np.ndarray, int]:
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    total = 0

    # 简单逐条或小批量评估
    batch_paths_gasf: List[str] = []
    batch_paths_gadf: List[str] = []
    batch_labels: List[int] = []
    batch_size = 32
    processed = 0

    def flush_batch():
        nonlocal cm, total, batch_paths_gasf, batch_paths_gadf, batch_labels, processed
        if not batch_paths_gasf:
            return
        labels = torch.as_tensor(batch_labels, dtype=torch.long, device=device)
        probs_gasf = forward_yolo_probs(gasf_model, batch_paths_gasf, device)
        probs_gadf = forward_yolo_probs(gadf_model, batch_paths_gadf, device)
        with torch.no_grad():
            p_fused, _ = attn_model(probs_gasf, probs_gadf)
        preds = p_fused.argmax(dim=1).cpu().numpy()
        labs = labels.cpu().numpy()
        for t, p in zip(labs, preds):
            cm[t, p] += 1
            total += 1
            processed += 1
        if processed % 1000 == 0:
            print(f"[eval_cwru_dualyi_attn] 已评估样本数: {processed}")
        batch_paths_gasf = []
        batch_paths_gadf = []
        batch_labels = []

    for gasf_img, gadf_img, lab in samples:
        batch_paths_gasf.append(str(gasf_img))
        batch_paths_gadf.append(str(gadf_img))
        batch_labels.append(lab)
        if len(batch_paths_gasf) >= batch_size:
            flush_batch()
    flush_batch()

    return cm, total


def save_confusion_matrix(cm: np.ndarray, total: int, out_dir: Path, title_suffix: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    counts_csv = out_dir / "confusion_matrix_counts_attn.csv"
    np.savetxt(counts_csv, cm, fmt="%d", delimiter=",")

    true_row_sums = cm.sum(axis=1, keepdims=True)
    safe_sums = np.where(true_row_sums == 0, 1, true_row_sums)
    cm_norm = cm.astype(float) / safe_sums
    norm_csv = out_dir / "confusion_matrix_normalized_attn.csv"
    np.savetxt(norm_csv, cm_norm, fmt="%.4f", delimiter=",")

    plot_mat = np.nan_to_num(cm_norm, nan=0.0, posinf=0.0, neginf=0.0)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(plot_mat, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels([f"{i}\n{CLASS_NAMES[i]}" for i in LABELS])
    ax.set_yticklabels([f"{i}:{CLASS_NAMES[i]}" for i in LABELS])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Dual-branch (GASF+GADF) with attention {title_suffix}")

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = plot_mat[i, j]
            txt = f"{val:.2f}" if true_row_sums[i, 0] > 0 else "-"
            ax.text(j, i, txt, ha="center", va="center", color="white" if val > 0.5 else "black")

    fig.tight_layout()
    fig_path = out_dir / "confusion_matrix_attn.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate dual-branch (GASF+GADF) with branch-attentive fusion on CWRU test set"
    )
    parser.add_argument("--run", type=int, default=DEFAULT_RUN, help=f"YOLO weights run index (default {DEFAULT_RUN})")
    parser.add_argument(
        "--use-enhanced",
        action="store_true",
        help="使用结构增强版 SA-GAF 数据集与对应的注意力模型（需先训练 train_cwru_dual_attn.py --use-enhanced）",
    )
    args = parser.parse_args()

    device = torch.device("cpu")

    # 选择数据根目录
    if args.use_enhanced and GASF_DATA_ROOT_ENH.is_dir() and GADF_DATA_ROOT_ENH.is_dir():
        gasf_root = GASF_DATA_ROOT_ENH
        gadf_root = GADF_DATA_ROOT_ENH
        print(f"[eval_cwru_dual_attn] 使用增强版 SA-GAF 测试集：\n  GASF_ENH: {gasf_root}\n  GADF_ENH: {gadf_root}")
        title_suffix = "(attention, SA-GAF, normalized)"
    else:
        if args.use_enhanced:
            print(
                "[eval_cwru_dual_attn] 警告：指定了 --use-enhanced，但未检测到增强版数据集，"
                "将退回使用原始数据集与原始注意力模型。"
            )
        gasf_root = GASF_DATA_ROOT
        gadf_root = GADF_DATA_ROOT
        title_suffix = "(attention, original GAF, normalized)"
        print(f"[eval_cwru_dual_attn] 使用原始 GAF 测试集：\n  GASF: {gasf_root}\n  GADF: {gadf_root}")

    # 加载 YOLO 模型与注意力模块
    gasf_model, gadf_model = load_yolo_models(args.run)
    attn_model = load_attention_model(args.use_enhanced, device)

    # 收集 test 样本配对
    samples = collect_test_pairs(gasf_root, gadf_root)
    print(f"[eval_cwru_dual_attn] 测试集中可配对样本数: {len(samples)}")

    # 评估
    cm, total = evaluate(gasf_model, gadf_model, attn_model, samples, device)
    overall_acc = cm.trace() / total if total > 0 else 0.0
    print(f"Total samples: {total}")
    print(f"Overall accuracy (attention fusion): {overall_acc:.4f}")

    for lab in LABELS:
        true_total = cm[lab, :].sum()
        acc = cm[lab, lab] / true_total if true_total > 0 else 0.0
        print(f"Class {lab} ({CLASS_NAMES[lab]}): accuracy = {acc:.4f} ({cm[lab, lab]}/{true_total})")

    out_dir = Path("runs/cwru_cls/cwru_dual_eval_attn_enh" if args.use_enhanced else "runs/cwru_cls/cwru_dual_eval_attn")
    save_confusion_matrix(cm, total, out_dir, title_suffix)
    print(f"Confusion matrix (with attention) saved to: {out_dir}")


if __name__ == "__main__":
    main()

