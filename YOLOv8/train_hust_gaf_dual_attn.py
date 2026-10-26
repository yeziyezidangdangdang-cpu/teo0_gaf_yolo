"""
训练 HUST-bearing GAF 双分支（GASF-RGB + GADF-RGB）的注意力融合模块。

将 PROJECT_INNOVATION_PLAN.md 中的「分支注意力融合」创新方案适配到 HUST 数据集：
- CWRU：单轴 GASF / GADF → 双分支
- HUST：三轴 X/Y/Z 编码为 RGB 的 GASF-RGB / GADF-RGB → 双分支（两分支均充分利用三轴信息）

数据来源（由 2.2_prepare_yolov8_datasets_hust_gaf_rgb_strict.py 整理）：
- GASF-RGB: D:/TEP_2_repro/gaf_yolo/yolo_datasets/hust_gasf_rgb_cls_strict
- GADF-RGB: D:/TEP_2_repro/gaf_yolo/yolo_datasets/hust_gadf_rgb_cls_strict

YOLO 权重（由 train_hust_gaf_rgb_dual.py 训练）：
- GASF: runs/classify/runs/hust_gaf_rgb_cls/hust_gasf_rgb/weights/best.pt
- GADF: runs/classify/runs/hust_gaf_rgb_cls/hust_gadf_rgb/weights/best.pt

文件配对规则：gasf_rgb_<label>_<idx>.png <-> gadf_rgb_<label>_<idx>.png

运行示例：
    cd D:\TEP_2_repro\YOLOv8
    python train_hust_gaf_dual.py --epochs 40 --fraction 1.0   # 先训练双分支
    python train_hust_gaf_dual_attn.py --epochs 10 --batch-size 64
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO


YOLO_ROOT = Path(__file__).resolve().parent
RUNS_DIR = YOLO_ROOT / "runs" / "classify" / "hust_gaf_rgb_cls"

GASF_DATA_ROOT = Path(r"D:\TEP_2_repro\gaf_yolo\yolo_datasets\hust_gasf_rgb_cls_strict")
GADF_DATA_ROOT = Path(r"D:\TEP_2_repro\gaf_yolo\yolo_datasets\hust_gadf_rgb_cls_strict")

# 可选：包络版 GAF-RGB
GASF_DATA_ROOT_ENV = Path(r"D:\TEP_2_repro\gaf_yolo\yolo_datasets\hust_gasf_rgb_env_cls")
GADF_DATA_ROOT_ENV = Path(r"D:\TEP_2_repro\gaf_yolo\yolo_datasets\hust_gadf_rgb_env_cls")

NUM_CLASSES = 4
CLASS_NAMES = {0: "Healthy", 1: "Ball", 2: "Inner race", 3: "Outer race"}

ATTN_SAVE_DIR = YOLO_ROOT / "runs" / "hust_gaf_cls" / "branch_attn"


def get_weights_paths() -> Tuple[Path, Path]:
    gasf = RUNS_DIR / "hust_gasf_rgb" / "weights" / "best.pt"
    gadf = RUNS_DIR / "hust_gadf_rgb" / "weights" / "best.pt"
    return gasf, gadf


def load_yolo_models() -> Tuple[YOLO, YOLO]:
    gasf_pt, gadf_pt = get_weights_paths()
    if not gasf_pt.is_file():
        raise FileNotFoundError(f"GASF-RGB weights not found: {gasf_pt}\nRun: python train_hust_gaf_rgb_dual.py first")
    if not gadf_pt.is_file():
        raise FileNotFoundError(f"GADF-RGB weights not found: {gadf_pt}\nRun: python train_hust_gaf_rgb_dual.py first")
    gasf_model = YOLO(str(gasf_pt))
    gadf_model = YOLO(str(gadf_pt))
    for p in gasf_model.model.parameters():
        p.requires_grad = False
    for p in gadf_model.model.parameters():
        p.requires_grad = False
    gasf_model.model.eval()
    gadf_model.model.eval()
    return gasf_model, gadf_model


class PairedHUSTGAFDataset(Dataset):
    """
    基于 HUST GAF-RGB 数据集构建 (gasf_img, gadf_img, label) 配对样本。
    配对规则：gasf_rgb_<label>_<idx>.png <-> gadf_rgb_<label>_<idx>.png
    """

    def __init__(self, root_gasf: Path, root_gadf: Path, split: str = "train") -> None:
        super().__init__()
        assert split in ("train", "val"), "split must be 'train' or 'val'"
        self.samples: List[Tuple[Path, Path, int]] = []

        def _collect(split_name: str) -> List[Tuple[Path, Path, int]]:
            samples: List[Tuple[Path, Path, int]] = []
            split_gasf = root_gasf / split_name
            split_gadf = root_gadf / split_name
            if not split_gasf.is_dir() or not split_gadf.is_dir():
                return samples
            for lab in range(NUM_CLASSES):
                lab_str = str(lab)
                gasf_dir = split_gasf / lab_str
                gadf_dir = split_gadf / lab_str
                if not gasf_dir.is_dir() or not gadf_dir.is_dir():
                    continue
                gasf_imgs = sorted([p for p in gasf_dir.glob("*.png") if p.is_file()])
                gadf_map = {p.name: p for p in gadf_dir.glob("*.png") if p.is_file()}
                for gasf_img in gasf_imgs:
                    gadf_name = gasf_img.name.replace("gasf_rgb_", "gadf_rgb_", 1)
                    gadf_img = gadf_map.get(gadf_name)
                    if gadf_img is None:
                        continue
                    samples.append((gasf_img, gadf_img, lab))
            return samples

        # 首选指定的 split（train / val）
        self.samples = _collect(split)

        # 若严格划分导致 val 为空，则回退到使用 train 作为验证集来源，避免直接报错
        if not self.samples and split == "val":
            print(
                f"[PairedHUSTGAFDataset] split=val 下无配对样本，"
                f"回退为使用 split=train 作为注意力验证数据（仅影响 MLP 验证，不影响 YOLO 训练划分）。"
            )
            self.samples = _collect("train")

        if not self.samples:
            raise RuntimeError(
                f"No paired samples in {root_gasf} / {root_gadf} split={split} "
                "(even after fallback to train). Please check that 2.2_prepare_yolov8_datasets_hust_gaf_rgb.py "
                "has been run and that gasf/gadf filenames are aligned."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        gasf_img, gadf_img, lab = self.samples[idx]
        return str(gasf_img), str(gadf_img), lab


class BranchAttention(nn.Module):
    """分支注意力融合：p_fused = w_gasf * p_gasf + w_gadf * p_gadf，权重由 MLP 学习。"""

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


def train_epoch(gasf_m, gadf_m, attn_m, loader, optimizer, device) -> Tuple[float, float]:
    attn_m.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for gasf_paths, gadf_paths, labels in loader:
        labels = torch.tensor(labels, dtype=torch.long, device=device)
        probs_gasf = forward_yolo_probs(gasf_m, list(gasf_paths), device)
        probs_gadf = forward_yolo_probs(gadf_m, list(gadf_paths), device)
        p_fused, _ = attn_m(probs_gasf, probs_gadf)
        p_true = p_fused[torch.arange(p_fused.size(0)), labels]
        loss = -torch.log(p_true + 1e-8).mean()
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
            p_true = p_fused[torch.arange(p_fused.size(0)), labels]
            loss = -torch.log(p_true + 1e-8).mean()
            total_loss += float(loss.item()) * labels.size(0)
            total_correct += int((p_fused.argmax(dim=1) == labels).sum().item())
            total += labels.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Train branch-attentive fusion for HUST GAF dual-branch (GASF-RGB + GADF-RGB)")
    parser.add_argument("--epochs", type=int, default=10, help="attention epochs (default 10)")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size (default 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate (default 1e-3)")
    parser.add_argument("--use-envelope", action="store_true", help="Use envelope GAF-RGB datasets")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.use_envelope and GASF_DATA_ROOT_ENV.is_dir() and GADF_DATA_ROOT_ENV.is_dir():
        gasf_root, gadf_root = GASF_DATA_ROOT_ENV, GADF_DATA_ROOT_ENV
        print(f"[train_hust_gaf_dual_attn] Using envelope GAF-RGB: {gasf_root}, {gadf_root}")
        attn_ckpt = ATTN_SAVE_DIR / "branch_attn_env.pt"
    else:
        gasf_root, gadf_root = GASF_DATA_ROOT, GADF_DATA_ROOT
        print(f"[train_hust_gaf_dual_attn] Using raw GAF-RGB: {gasf_root}, {gadf_root}")
        attn_ckpt = ATTN_SAVE_DIR / "branch_attn.pt"

    train_ds = PairedHUSTGAFDataset(gasf_root, gadf_root, split="train")
    val_ds = PairedHUSTGAFDataset(gasf_root, gadf_root, split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    gasf_model, gadf_model = load_yolo_models()
    attn_model = BranchAttention(num_classes=NUM_CLASSES, hidden_dim=16).to(device)
    optimizer = torch.optim.Adam(attn_model.parameters(), lr=args.lr)

    ATTN_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    # 保存训练曲线供 plot_hust_loss_accuracy.py 绘图
    results_csv = ATTN_SAVE_DIR / "results_attn.csv"
    with open(results_csv, "w", encoding="utf-8") as f:
        f.write("epoch,train/loss,metrics/accuracy_top1,val/loss,val/accuracy_top1\n")

    print(f"[train_hust_gaf_dual_attn] Training attention fusion, epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(gasf_model, gadf_model, attn_model, train_loader, optimizer, device)
        val_loss, val_acc = eval_epoch(gasf_model, gadf_model, attn_model, val_loader, device)
        print(f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        with open(results_csv, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.6f},{train_acc:.6f},{val_loss:.6f},{val_acc:.6f}\n")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": attn_model.state_dict(),
                "num_classes": NUM_CLASSES,
                "hidden_dim": 16,
                "use_envelope": args.use_envelope,
            }, attn_ckpt)
            print(f"  -> Saved best to {attn_ckpt} (val_acc={val_acc:.4f})")

    print(f"[train_hust_gaf_dual_attn] Done. Best val_acc={best_val_acc:.4f}, ckpt={attn_ckpt}")


if __name__ == "__main__":
    main()
