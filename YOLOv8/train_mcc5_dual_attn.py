"""
Train branch-attention fusion for MCC5 GASF-RGB + GADF-RGB (frozen YOLO branches).

先运行 train_mcc5_dual.py，再本脚本。默认 whole_file 划分数据。

Usage:
    cd D:\\TEP_2_repro\\YOLOv8
    python train_mcc5_dual_attn.py --split-mode whole_file --epochs 10
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO

from mcc5_paths import (
    ATTN_CKPT,
    ATTN_DIR,
    GADF_DATA_ROOT,
    GASF_DATA_ROOT,
    MCC5_IMGSZ,
    set_split_mode,
    set_exp_point,
    get_exp_point,
)

NUM_CLASSES = 4
CLASS_NAMES = {0: "Healthy", 1: "Ball", 2: "Inner race", 3: "Outer race"}


def get_weights_paths() -> Tuple[Path, Path]:
    from mcc5_paths import branch_weight_path

    gasf = branch_weight_path("gasf")
    gadf = branch_weight_path("gadf")
    return gasf, gadf


def load_yolo_models() -> Tuple[YOLO, YOLO]:
    gasf_pt, gadf_pt = get_weights_paths()
    if not gasf_pt.is_file():
        raise FileNotFoundError(f"GASF weights not found: {gasf_pt}\nRun train_mcc5_dual.py first")
    if not gadf_pt.is_file():
        raise FileNotFoundError(f"GADF weights not found: {gadf_pt}")
    gasf_model = YOLO(str(gasf_pt))
    gadf_model = YOLO(str(gadf_pt))
    for p in gasf_model.model.parameters():
        p.requires_grad = False
    for p in gadf_model.model.parameters():
        p.requires_grad = False
    gasf_model.model.eval()
    gadf_model.model.eval()
    return gasf_model, gadf_model


class PairedMCC5GAFDataset(Dataset):
    def __init__(self, root_gasf: Path, root_gadf: Path, split: str = "train") -> None:
        assert split in ("train", "val")
        self.samples: List[Tuple[Path, Path, int]] = []

        def _collect(split_name: str) -> List[Tuple[Path, Path, int]]:
            out: List[Tuple[Path, Path, int]] = []
            sg, sd = root_gasf / split_name, root_gadf / split_name
            if not sg.is_dir() or not sd.is_dir():
                return out
            for lab in range(NUM_CLASSES):
                gs, gd = sg / str(lab), sd / str(lab)
                if not gs.is_dir() or not gd.is_dir():
                    continue
                gadf_map = {p.name: p for p in gd.glob("*.png")}
                for gasf_img in sorted(gs.glob("*.png")):
                    gadf_name = gasf_img.name.replace("gasf_rgb_", "gadf_rgb_", 1)
                    gadf_img = gadf_map.get(gadf_name)
                    if gadf_img:
                        out.append((gasf_img, gadf_img, lab))
            return out

        self.samples = _collect(split)
        if not self.samples and split == "val":
            print("[PairedMCC5] val 为空，回退使用 train 作注意力验证")
            self.samples = _collect("train")
        if not self.samples:
            raise RuntimeError(f"No paired samples for split={split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        gasf_img, gadf_img, lab = self.samples[idx]
        return str(gasf_img), str(gadf_img), lab


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
        attn = torch.softmax(self.mlp(z), dim=1)
        w_gasf = attn[:, 0].unsqueeze(1)
        w_gadf = attn[:, 1].unsqueeze(1)
        p_fused = w_gasf * probs_gasf + w_gadf * probs_gadf
        p_fused = p_fused / (p_fused.sum(dim=1, keepdim=True) + 1e-8)
        return p_fused, attn


def forward_yolo_probs(model: YOLO, img_paths: List[str], device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        results = model(img_paths, imgsz=MCC5_IMGSZ, verbose=False)
    probs_list = []
    for r in results:
        p = r.probs.data
        if p.dim() == 1:
            p = p.unsqueeze(0)
        probs_list.append(p)
    return torch.cat(probs_list, dim=0).to(device)


def train_epoch(gasf_m, gadf_m, attn_m, loader, optimizer, device) -> Tuple[float, float]:
    attn_m.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for gasf_paths, gadf_paths, labels in loader:
        labels = torch.tensor(labels, dtype=torch.long, device=device)
        probs_gasf = forward_yolo_probs(gasf_m, list(gasf_paths), device)
        probs_gadf = forward_yolo_probs(gadf_m, list(gadf_paths), device)
        p_fused, _ = attn_m(probs_gasf, probs_gadf)
        loss = -torch.log(p_fused[torch.arange(p_fused.size(0)), labels] + 1e-8).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        total_correct += int((p_fused.argmax(dim=1) == labels).sum().item())
        total += labels.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)


def eval_epoch(gasf_m, gadf_m, attn_m, loader, device) -> Tuple[float, float]:
    attn_m.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.no_grad():
        for gasf_paths, gadf_paths, labels in loader:
            labels = torch.tensor(labels, dtype=torch.long, device=device)
            probs_gasf = forward_yolo_probs(gasf_m, list(gasf_paths), device)
            probs_gadf = forward_yolo_probs(gadf_m, list(gadf_paths), device)
            p_fused, _ = attn_m(probs_gasf, probs_gadf)
            loss = -torch.log(p_fused[torch.arange(p_fused.size(0)), labels] + 1e-8).mean()
            total_loss += float(loss.item()) * labels.size(0)
            total_correct += int((p_fused.argmax(dim=1) == labels).sum().item())
            total += labels.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MCC5 branch-attention fusion")
    parser.add_argument("--exp-point", choices=("c", "b", "d"), default="b")
    parser.add_argument("--split-mode", choices=("temporal", "whole_file"), default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    if args.split_mode:
        set_split_mode(args.split_mode)
    else:
        set_exp_point(args.exp_point)
    print(f"[train_mcc5_dual_attn] exp_point={get_exp_point().upper()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = PairedMCC5GAFDataset(GASF_DATA_ROOT, GADF_DATA_ROOT, "train")
    val_ds = PairedMCC5GAFDataset(GASF_DATA_ROOT, GADF_DATA_ROOT, "val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    gasf_model, gadf_model = load_yolo_models()
    attn_model = BranchAttention().to(device)
    optimizer = torch.optim.Adam(attn_model.parameters(), lr=args.lr)

    ATTN_DIR.mkdir(parents=True, exist_ok=True)
    results_csv = ATTN_DIR / "results_attn.csv"
    results_csv.write_text(
        "epoch,train/loss,metrics/accuracy_top1,val/loss,val/accuracy_top1\n",
        encoding="utf-8",
    )
    best_val = 0.0

    print(f"[train_mcc5_dual_attn] exp_point={get_exp_point().upper()}, epochs={args.epochs}, device={device}")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(gasf_model, gadf_model, attn_model, train_loader, optimizer, device)
        va_loss, va_acc = eval_epoch(gasf_model, gadf_model, attn_model, val_loader, device)
        print(f"Epoch {epoch:02d} | train_acc={tr_acc:.4f} val_acc={va_acc:.4f}")
        with results_csv.open("a", encoding="utf-8") as f:
            f.write(f"{epoch},{tr_loss:.6f},{tr_acc:.6f},{va_loss:.6f},{va_acc:.6f}\n")
        if va_acc >= best_val:
            best_val = va_acc
            torch.save(
                {
                    "model_state": attn_model.state_dict(),
                    "num_classes": NUM_CLASSES,
                    "exp_point": get_exp_point(),
                },
                ATTN_CKPT,
            )
    print(f"Done. best_val_acc={best_val:.4f} -> {ATTN_CKPT}")


if __name__ == "__main__":
    main()
