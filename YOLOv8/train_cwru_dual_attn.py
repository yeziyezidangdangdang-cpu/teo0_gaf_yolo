"""
Train a branch-attentive fusion module on top of two YOLOv8 classification models (GASF + GADF).

现有方案：两个 YOLOv8 分类模型分别对 GASF / GADF 图像输出概率，最后做简单平均。
本脚本在此基础上增加一个轻量「分支注意力融合模块」，实现可学习的自适应加权：

- 输入：p_gasf ∈ R^C, p_gadf ∈ R^C（来自两个已训练好的 YOLOv8 分类器的概率）
- 拼接：z = concat(p_gasf, p_gadf) ∈ R^{2C}
- 通过一个小型 MLP 预测两个分支权重 α_gasf, α_gadf（softmax 归一化）
- 输出融合概率：p_fused = α_gasf * p_gasf + α_gadf * p_gadf
- 损失：对 p_fused 使用交叉熵（-log p_fused[y]）训练注意力模块参数（冻结 YOLO 权重）

数据来源：
- 默认使用 GAF 项目整理好的 YOLOv8 分类数据集：
    - GASF: D:/TEP_2_repro/gaf_yolo/yolo_datasets/gasf_cls
    - GADF: D:/TEP_2_repro/gaf_yolo/yolo_datasets/gadf_cls
- 可选：--use-enhanced 时，如存在结构增强版数据集：
    - GASF_ENH: D:/TEP_2_repro/gaf_yolo/yolo_datasets_enhanced/gasf_cls
    - GADF_ENH: D:/TEP_2_repro/gaf_yolo/yolo_datasets_enhanced/gadf_cls

运行示例（原始 GAF）：
    cd D:/TEP_2_repro/YOLOv8
    python train_cwru_dual_attn.py --epochs 10 --batch-size 64

运行示例（增强 SA-GAF，如已生成并整理）：
    python train_cwru_dual_attn.py --epochs 10 --batch-size 64 --use-enhanced
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO


# 与 eval/predict 脚本保持一致的权重目录结构
from cwru_paths import branch_weight_path

DEFAULT_RUN = 1  # 1 = cwru_gasf / cwru_gadf（训练脚本默认覆盖这对）

# GAF 数据集根目录（原始，严格划分）
GASF_DATA_ROOT = Path(r"D:\TEP_2_repro\gaf_yolo\yolo_datasets\gasf_cls_strict")
GADF_DATA_ROOT = Path(r"D:\TEP_2_repro\gaf_yolo\yolo_datasets\gadf_cls_strict")

# SA-GAF 数据集根目录（增强版，可选，严格划分）
GASF_DATA_ROOT_ENH = Path(r"D:\TEP_2_repro\gaf_yolo\yolo_datasets_enhanced\gasf_cls_strict")
GADF_DATA_ROOT_ENH = Path(r"D:\TEP_2_repro\gaf_yolo\yolo_datasets_enhanced\gadf_cls_strict")

NUM_CLASSES = 4
CLASS_NAMES = {
    0: "Normal",
    1: "Inner race fault",
    2: "Outer race fault",
    3: "Rolling element fault",
}

ATTN_RUN_DIR = Path(r"D:\TEP_2_repro\YOLOv8\runs\cwru_cls\branch_attn")


def get_weights_paths(run: int) -> Tuple[Path, Path]:
    return branch_weight_path("gasf", run), branch_weight_path("gadf", run)


def load_yolo_models(run: int = DEFAULT_RUN) -> Tuple[YOLO, YOLO]:
    gasf_pt, gadf_pt = get_weights_paths(run)
    if not gasf_pt.is_file():
        raise FileNotFoundError(f"GASF weights not found: {gasf_pt}")
    if not gadf_pt.is_file():
        raise FileNotFoundError(f"GADF weights not found: {gadf_pt}")
    gasf_model = YOLO(str(gasf_pt))
    gadf_model = YOLO(str(gadf_pt))
    # 冻结 YOLO 权重，只训练注意力模块
    for p in gasf_model.model.parameters():
        p.requires_grad = False
    for p in gadf_model.model.parameters():
        p.requires_grad = False
    gasf_model.model.eval()
    gadf_model.model.eval()
    return gasf_model, gadf_model


class PairedGAFDataset(Dataset):
    """
    基于已有的 YOLOv8 分类数据集目录，构建 (gasf_img, gadf_img, label) 配对样本。

    root_gasf / root_gadf:
        - train / 0-3 / *.png
        - val   / 0-3 / *.png
    文件命名：gasf_<label>_<idx>.png 与 gadf_<label>_<idx>.png 一一对应。
    """

    def __init__(self, root_gasf: Path, root_gadf: Path, split: str = "train") -> None:
        super().__init__()
        assert split in ("train", "val"), "split 必须是 'train' 或 'val'"
        self.samples: List[Tuple[Path, Path, int]] = []

        split_gasf = root_gasf / split
        split_gadf = root_gadf / split

        if not split_gasf.is_dir() or not split_gadf.is_dir():
            raise FileNotFoundError(f"缺少 split 目录: {split_gasf} 或 {split_gadf}")

        for lab in range(NUM_CLASSES):
            lab_str = str(lab)
            gasf_dir = split_gasf / lab_str
            gadf_dir = split_gadf / lab_str
            if not gasf_dir.is_dir() or not gadf_dir.is_dir():
                continue

            gasf_imgs = sorted([p for p in gasf_dir.glob("*.png") if p.is_file()])
            gadf_map = {p.name: p for p in gadf_dir.glob("*.png") if p.is_file()}

            for gasf_img in gasf_imgs:
                gadf_name = gasf_img.name.replace("gasf_", "gadf_", 1)
                gadf_img = gadf_map.get(gadf_name)
                if gadf_img is None:
                    continue
                self.samples.append((gasf_img, gadf_img, lab))

        if not self.samples:
            raise RuntimeError(f"{root_gasf} / {root_gadf} 下 split={split} 未找到任何可配对样本")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        gasf_img, gadf_img, lab = self.samples[idx]
        return str(gasf_img), str(gadf_img), lab


class BranchAttention(nn.Module):
    """
    分支注意力融合模块：
    - 输入：p_gasf, p_gadf（概率向量）
    - 输出：p_fused（融合概率）
    """

    def __init__(self, num_classes: int = NUM_CLASSES, hidden_dim: int = 16) -> None:
        super().__init__()
        in_dim = num_classes * 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, probs_gasf: torch.Tensor, probs_gadf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            probs_gasf: (B, C)
            probs_gadf: (B, C)
        Returns:
            p_fused: (B, C)
            attn_weights: (B, 2) 对应 [w_gasf, w_gadf]
        """
        z = torch.cat([probs_gasf, probs_gadf], dim=1)  # (B, 2C)
        logits = self.mlp(z)  # (B, 2)
        attn = torch.softmax(logits, dim=1)  # (B, 2)
        w_gasf = attn[:, 0].unsqueeze(1)  # (B, 1)
        w_gadf = attn[:, 1].unsqueeze(1)  # (B, 1)
        p_fused = w_gasf * probs_gasf + w_gadf * probs_gadf
        # 为数值稳定性做一个归一化（防止浮点累积误差）
        p_fused = p_fused / (p_fused.sum(dim=1, keepdim=True) + 1e-8)
        return p_fused, attn


def forward_yolo_probs(model: YOLO, img_paths: List[str], device: torch.device) -> torch.Tensor:
    """
    使用 YOLOv8 分类模型对一批图像前向，返回概率张量 (B, C)。
    注意：这里仅用于推理，梯度不回传到 YOLO。
    """
    # Ultralytics YOLO 接收列表路径即可批推理
    with torch.no_grad():
        results = model(img_paths, imgsz=128, verbose=False)
    probs_list = []
    for r in results:
        # r.probs.data 是 (C,) 或 (B,C)；在单图模式下是 (C,)
        p = r.probs.data
        if p.dim() == 1:
            p = p.unsqueeze(0)
        probs_list.append(p)
    probs = torch.cat(probs_list, dim=0)  # (B, C)
    return probs.to(device)


def train_epoch(
    gasf_model: YOLO,
    gadf_model: YOLO,
    attn_model: BranchAttention,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    attn_model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for gasf_paths, gadf_paths, labels in loader:
        # labels 可能已经是 Tensor（DataLoader 默认 collate 常见情况）。
        # 用 as_tensor/to 避免 torch.tensor(Tensor) 的拷贝警告与额外开销。
        if isinstance(labels, torch.Tensor):
            labels = labels.to(device=device, dtype=torch.long)
        else:
            labels = torch.as_tensor(labels, dtype=torch.long, device=device)

        # 1. YOLO 前向（冻结，仅推理）
        probs_gasf = forward_yolo_probs(gasf_model, list(gasf_paths), device)
        probs_gadf = forward_yolo_probs(gadf_model, list(gadf_paths), device)

        # 2. 注意力融合
        p_fused, _ = attn_model(probs_gasf, probs_gadf)  # (B, C)

        # 3. 交叉熵损失：-log p_fused[y]
        eps = 1e-8
        p_true = p_fused[torch.arange(p_fused.size(0)), labels]
        loss = -torch.log(p_true + eps).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        preds = p_fused.argmax(dim=1)
        total_correct += int((preds == labels).sum().item())
        total_samples += labels.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    acc = total_correct / max(total_samples, 1)
    return avg_loss, acc


def eval_epoch(
    gasf_model: YOLO,
    gadf_model: YOLO,
    attn_model: BranchAttention,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    attn_model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for gasf_paths, gadf_paths, labels in loader:
            if isinstance(labels, torch.Tensor):
                labels = labels.to(device=device, dtype=torch.long)
            else:
                labels = torch.as_tensor(labels, dtype=torch.long, device=device)

            probs_gasf = forward_yolo_probs(gasf_model, list(gasf_paths), device)
            probs_gadf = forward_yolo_probs(gadf_model, list(gadf_paths), device)

            p_fused, _ = attn_model(probs_gasf, probs_gadf)

            eps = 1e-8
            p_true = p_fused[torch.arange(p_fused.size(0)), labels]
            loss = -torch.log(p_true + eps).mean()

            total_loss += float(loss.item()) * labels.size(0)
            preds = p_fused.argmax(dim=1)
            total_correct += int((preds == labels).sum().item())
            total_samples += labels.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    acc = total_correct / max(total_samples, 1)
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser(description="Train branch-attentive fusion for dual-branch (GASF+GADF) classifiers")
    parser.add_argument("--run", type=int, default=DEFAULT_RUN, help=f"YOLO weights run index (default {DEFAULT_RUN})")
    parser.add_argument("--epochs", type=int, default=10, help="attention 模块训练轮数（默认 10）")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size（默认 64）")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率（默认 1e-3）")
    parser.add_argument(
        "--use-enhanced",
        action="store_true",
        help="使用结构增强版 SA-GAF 数据集（需先在 GAF 项目中生成并整理 yolo_datasets_enhanced）",
    )
    args = parser.parse_args()

    device = torch.device("cpu")  # 当前环境未检测到 GPU，统一使用 CPU

    # 选择数据根目录
    if args.use_enhanced and GASF_DATA_ROOT_ENH.is_dir() and GADF_DATA_ROOT_ENH.is_dir():
        gasf_root = GASF_DATA_ROOT_ENH
        gadf_root = GADF_DATA_ROOT_ENH
        print(f"[train_cwru_dual_attn] 使用增强版 SA-GAF 数据集：\n  GASF_ENH: {gasf_root}\n  GADF_ENH: {gadf_root}")
    else:
        if args.use_enhanced:
            print(
                "[train_cwru_dual_attn] 警告：指定了 --use-enhanced，但未检测到 "
                f"{GASF_DATA_ROOT_ENH} 或 {GADF_DATA_ROOT_ENH}，将退回使用原始数据集。"
            )
        gasf_root = GASF_DATA_ROOT
        gadf_root = GADF_DATA_ROOT
        print(f"[train_cwru_dual_attn] 使用原始 GAF 数据集：\n  GASF: {gasf_root}\n  GADF: {gadf_root}")

    # 构建数据集与 DataLoader
    train_ds = PairedGAFDataset(gasf_root, gadf_root, split="train")
    val_ds = PairedGAFDataset(gasf_root, gadf_root, split="val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 加载已训练好的 YOLOv8 分支模型（冻结）
    gasf_model, gadf_model = load_yolo_models(args.run)

    # 构建注意力模块
    attn_model = BranchAttention(num_classes=NUM_CLASSES, hidden_dim=16).to(device)
    optimizer = torch.optim.Adam(attn_model.parameters(), lr=args.lr)

    # 准备输出目录
    ATTN_RUN_DIR.mkdir(parents=True, exist_ok=True)
    attn_ckpt = ATTN_RUN_DIR / ("branch_attn_enh.pt" if args.use_enhanced else "branch_attn.pt")

    best_val_acc = 0.0
    print(
        f"[train_cwru_dual_attn] 开始训练分支注意力融合模块，epochs={args.epochs}, "
        f"batch_size={args.batch_size}, lr={args.lr}, run={args.run}"
    )

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(gasf_model, gadf_model, attn_model, train_loader, optimizer, device)
        val_loss, val_acc = eval_epoch(gasf_model, gadf_model, attn_model, val_loader, device)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            state = {
                "model_state": attn_model.state_dict(),
                "num_classes": NUM_CLASSES,
                "hidden_dim": 16,
                "run": args.run,
                "use_enhanced": args.use_enhanced,
            }
            torch.save(state, attn_ckpt)
            print(f"[train_cwru_dual_attn] 新的最优模型已保存到: {attn_ckpt} (val_acc={val_acc:.4f})")

    print(f"[train_cwru_dual_attn] 训练完成。最佳验证准确率: {best_val_acc:.4f}")
    print(f"  注意力权重文件: {attn_ckpt}")


if __name__ == "__main__":
    main()

