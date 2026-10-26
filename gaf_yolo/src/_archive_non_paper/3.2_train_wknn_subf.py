"""
使用 3.1 提取的 cAR 特征，在 SUBF 上做一个小体量的 wKNN 实验，
作为 AREEMD+cAR 思路的数值基线（简化版）。

注意：
- 这里使用 scikit-learn 的 KNeighborsClassifier，设置 weights='distance'
  近似论文中的 Weighted-KNN。
- 采用 5 折 StratifiedKFold 交叉验证，报告平均准确率。
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import utils.project_paths as pp  # noqa: E402

from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier


INPUT_NPZ = Path(r"D:/TEP_2_repro/gaf_yolo/data/processed/car_features_subf.npz")


def main():
    if not INPUT_NPZ.exists():
        raise FileNotFoundError(f"未找到特征文件: {INPUT_NPZ}")

    data = np.load(INPUT_NPZ)
    X = data["features"]
    y = data["labels"]
    ar_order = int(data["ar_order"])
    max_samples_per_class = int(data["max_samples_per_class"])

    print(f"[3.2] 载入 cAR 特征: X 形状={X.shape}, y 形状={y.shape}")
    print(f"      AR 阶数: {ar_order}, 每类最多样本: {max_samples_per_class}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []

    fold = 1
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",  # 近似 wKNN
            metric="euclidean",
        )
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        accs.append(acc)
        print(f"[3.2] Fold {fold}: accuracy = {acc:.4f}")
        fold += 1

    accs = np.array(accs)
    print(
        f"[3.2] 平均准确率 = {accs.mean():.4f} "
        f"(标准差 = {accs.std():.4f}, 各折: {accs})"
    )


if __name__ == "__main__":
    main()

