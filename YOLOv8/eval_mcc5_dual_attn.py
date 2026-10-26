"""
Evaluate MCC5 dual-branch with attention fusion on test set.

Usage:
    cd D:\\TEP_2_repro\\YOLOv8
    python eval_mcc5_dual_attn.py --exp-point b
    python eval_mcc5_dual_attn.py --exp-point d
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from ultralytics import YOLO

from mcc5_paths import get_exp_point, set_exp_point, set_split_mode

LABELS = [0, 1, 2, 3]
CLASS_NAMES = {0: "Healthy", 1: "Ball", 2: "Inner race", 3: "Outer race"}


class BranchAttention(nn.Module):
    def __init__(self, num_classes: int = 4, hidden_dim: int = 16) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_classes * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, probs_gasf, probs_gadf):
        z = torch.cat([probs_gasf, probs_gadf], dim=1)
        attn = torch.softmax(self.mlp(z), dim=1)
        w_gasf = attn[:, 0].unsqueeze(1)
        w_gadf = attn[:, 1].unsqueeze(1)
        p_fused = w_gasf * probs_gasf + w_gadf * probs_gadf
        p_fused = p_fused / (p_fused.sum(dim=1, keepdim=True) + 1e-8)
        return p_fused, attn


def load_models(device):
    import mcc5_paths as mp

    gasf_pt, gadf_pt = mp.branch_weight_path("gasf"), mp.branch_weight_path("gadf")
    if not mp.ATTN_CKPT.is_file():
        raise FileNotFoundError(
            f"Attention ckpt not found: {mp.ATTN_CKPT}\n"
            f"Run train_mcc5_dual_attn.py --exp-point {mp.get_exp_point()}"
        )
    gasf_model, gadf_model = YOLO(str(gasf_pt)), YOLO(str(gadf_pt))
    ckpt = torch.load(mp.ATTN_CKPT, map_location=device)
    attn = BranchAttention(num_classes=ckpt.get("num_classes", 4)).to(device)
    attn.load_state_dict(ckpt["model_state"])
    attn.eval()
    return gasf_model, gadf_model, attn


def evaluate_attn(device) -> tuple[np.ndarray, int]:
    import mcc5_paths as mp

    gasf_model, gadf_model, attn_model = load_models(device)
    gasf_test = mp.GASF_DATA_ROOT / "test"
    gadf_test = mp.GADF_DATA_ROOT / "test"
    cm = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    total = 0

    for lab in LABELS:
        gd = gasf_test / str(lab)
        dd = gadf_test / str(lab)
        if not gd.is_dir() or not dd.is_dir():
            continue
        gadf_map = {p.name: p for p in dd.glob("*.png")}
        for gasf_img in sorted(gd.glob("*.png")):
            gadf_name = gasf_img.name.replace("gasf_", "gadf_", 1)
            gadf_img = gadf_map.get(gadf_name)
            if gadf_img is None:
                continue
            r1 = gasf_model(str(gasf_img), imgsz=mp.MCC5_IMGSZ, verbose=False)[0]
            r2 = gadf_model(str(gadf_img), imgsz=mp.MCC5_IMGSZ, verbose=False)[0]
            p1 = r1.probs.data.unsqueeze(0).to(device)
            p2 = r2.probs.data.unsqueeze(0).to(device)
            with torch.no_grad():
                p_fused, _ = attn_model(p1, p2)
            pred = int(p_fused.argmax(dim=1).item())
            cm[lab, pred] += 1
            total += 1
    return cm, total


def save_cm(cm: np.ndarray, out_dir: Path) -> float:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / "confusion_matrix_counts_attn.csv", cm, fmt="%d", delimiter=",")
    row_sums = cm.sum(axis=1, keepdims=True)
    safe = np.where(row_sums == 0, 1, row_sums)
    cm_norm = cm.astype(float) / safe
    np.savetxt(out_dir / "confusion_matrix_attn_normalized.csv", cm_norm, fmt="%.4f", delimiter=",")
    total = cm.sum()
    overall = cm.trace() / total if total > 0 else 0.0

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(np.nan_to_num(cm_norm), cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set_title("MCC5 dual attention (test)")
    fig.savefig(out_dir / "confusion_matrix_attn.png", dpi=200)
    plt.close(fig)
    return overall


def metrics_from_cm(cm: np.ndarray) -> dict:
    total = cm.sum()
    overall = cm.trace() / total if total > 0 else 0.0
    recalls = []
    for i in range(len(LABELS)):
        row = cm[i, :].sum()
        recalls.append(cm[i, i] / row if row > 0 else 0.0)
    return {"overall": overall, "recalls": recalls}


def append_table3_attn(attn_metrics: dict, split_label: str) -> None:
    import mcc5_paths as mp

    rows = []
    if mp.TABLE3_CSV.is_file():
        with mp.TABLE3_CSV.open("r", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
    else:
        rows = [
            ["方法", "总体准确率(%)", "Healthy", "Ball", "Inner", "Outer", "数据来源"],
        ]
    rows = [r for r in rows if not (r and "注意力" in r[0])]
    rows.insert(
        4 if len(rows) >= 4 else len(rows),
        [
            "MCC5 双分支注意力融合",
            f"{attn_metrics['overall']*100:.2f}",
            *[f"{r*100:.2f}" for r in attn_metrics["recalls"]],
            "测试集",
        ],
    )
    with mp.TABLE3_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(rows)

    mean_line = next((r[1] for r in rows if len(r) > 1 and "简单平均" in r[0]), "—")
    attn_line = f"{attn_metrics['overall']*100:.2f}"
    mp.TABLE3_MD.write_text(
        "\n".join([
            f"# MCC5-THU 双分支 GAF-RGB 性能（{split_label}）",
            "",
            f"- 划分：{split_label}",
            f"- 测试集双分支简单平均：{mean_line}%",
            f"- 测试集双分支注意力融合：{attn_line}%",
            f"- 注意力相对简单平均：{float(attn_line) - float(mean_line):+.2f} pp",
        ]),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-point", choices=("c", "b", "d"), default="b")
    parser.add_argument("--split-mode", choices=("temporal", "whole_file", "random"), default=None)
    args = parser.parse_args()
    if args.split_mode:
        set_split_mode(args.split_mode)
    else:
        set_exp_point(args.exp_point)

    import mcc5_paths as mp

    split_labels = {"b": "whole_file (B点)", "c": "temporal (C点)", "d": "random shuffle (D点)"}
    split_label = split_labels.get(get_exp_point(), get_exp_point())
    print(f"[eval_mcc5_dual_attn] exp_point={get_exp_point().upper()} split={split_label}")
    print(f"  eval out: {mp.EVAL_DUAL_ATTN_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cm, total = evaluate_attn(device)
    overall = save_cm(cm, mp.EVAL_DUAL_ATTN_DIR)
    m = metrics_from_cm(cm)
    append_table3_attn(m, split_label)
    print(f"[eval_mcc5_dual_attn] test overall={overall:.4f} (N={total})")
    for lab in LABELS:
        row = cm[lab, :].sum()
        rec = cm[lab, lab] / row if row > 0 else 0.0
        print(f"  Class {lab} ({CLASS_NAMES[lab]}): {rec:.4f} ({cm[lab, lab]}/{row})")
    print(f"Saved: {mp.EVAL_DUAL_ATTN_DIR}")


if __name__ == "__main__":
    main()
