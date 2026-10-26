"""
Tri-axis polar mapping + GAF demo figure for paper.

Output:
  D:\\C论文\\优化材料\\tri_axis_polar_gaf_demo.png

This script generates a synthetic tri-axial vibration signal (X/Y/Z), then:
  1) normalizes each axis to [-1, 1]
  2) maps to polar angle: phi = arccos(x)
  3) builds GADF and shows how tri-axis GADF can be stacked as RGB

Font: Times New Roman (if available).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.font_manager as fm

import sys

# Ensure repo root is importable (for `gaf_yolo.utils.gaf_utils`)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaf_yolo.utils.gaf_utils import gaf_transform


def _set_tnr_style(font_size: float = 10.0) -> None:
    fonts = {f.name for f in fm.fontManager.ttflist}
    has_tnr = "Times New Roman" in fonts
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] if has_tnr else []
    plt.rcParams["font.sans-serif"] = plt.rcParams["font.serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = font_size


def _minmax_to_minus1_1(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (2 * x - hi - lo) / (hi - lo)


def main() -> None:
    _set_tnr_style(font_size=9.0)

    out_dir = Path(r"D:\C论文\优化材料")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tri_axis_polar_gaf_demo.png"

    # --- synthetic tri-axis vibration signal (example only) ---
    n = 224  # show 224×224 like many CNN inputs
    t = np.linspace(0, 1.0, n, endpoint=False)
    rng = np.random.default_rng(7)

    # X/Y/Z: same base freq with slight phase/amp differences + impulsive component + noise
    base = np.sin(2 * np.pi * 6 * t)
    impulse = (rng.random(n) < 0.03).astype(float) * rng.normal(2.5, 0.4, size=n)
    x_raw = 0.9 * base + 0.35 * np.sin(2 * np.pi * 13 * t + 0.4) + 0.25 * impulse + 0.08 * rng.normal(size=n)
    y_raw = 0.7 * np.sin(2 * np.pi * 6 * t + 0.8) + 0.25 * np.sin(2 * np.pi * 10 * t) + 0.18 * impulse + 0.08 * rng.normal(size=n)
    z_raw = 0.8 * np.sin(2 * np.pi * 6 * t + 1.4) + 0.20 * np.sin(2 * np.pi * 16 * t) + 0.30 * impulse + 0.08 * rng.normal(size=n)

    # normalize each axis to [-1, 1] (per-sample axis-wise, consistent with many GAF pipelines)
    x = _minmax_to_minus1_1(x_raw)
    y = _minmax_to_minus1_1(y_raw)
    z = _minmax_to_minus1_1(z_raw)

    # polar angles
    phi_x = np.arccos(np.clip(x, -1, 1))
    phi_y = np.arccos(np.clip(y, -1, 1))
    phi_z = np.arccos(np.clip(z, -1, 1))

    # GADF per-axis
    gadf_x = gaf_transform(x, method="d", normalize=False)
    gadf_y = gaf_transform(y, method="d", normalize=False)
    gadf_z = gaf_transform(z, method="d", normalize=False)

    # stack to RGB for a single image (tri-axis -> RGB)
    rgb = np.stack([gadf_x, gadf_y, gadf_z], axis=-1)
    # rescale for visualization to [0,1]
    rgb_vis = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-12)

    # --- figure layout ---
    fig = plt.figure(figsize=(10.2, 4.0), dpi=300)
    gs = fig.add_gridspec(2, 4, width_ratios=[1.15, 1.05, 1.05, 1.15], hspace=0.35, wspace=0.30)

    # (a) tri-axis signals
    ax_sig = fig.add_subplot(gs[:, 0])
    ax_sig.plot(t, x, lw=1.0, label="X-axis")
    ax_sig.plot(t, y, lw=1.0, label="Y-axis")
    ax_sig.plot(t, z, lw=1.0, label="Z-axis")
    ax_sig.set_title("(a) Tri-axis signal (normalized)", pad=6)
    ax_sig.set_xlabel("Time (normalized)")
    ax_sig.set_ylabel("Amplitude")
    ax_sig.grid(True, ls="--", alpha=0.25, lw=0.6)
    ax_sig.legend(frameon=True, fontsize=8, loc="upper right")

    # (b) polar mapping illustration: phi vs time for one axis + polar plot point cloud
    ax_phi = fig.add_subplot(gs[0, 1])
    ax_phi.plot(t, phi_x, color="#1f77b4", lw=1.0)
    ax_phi.set_title("(b) Polar angle ϕ = arccos(x) (X-axis)", pad=6)
    ax_phi.set_xlabel("Time (normalized)")
    ax_phi.set_ylabel("ϕ (rad)")
    ax_phi.grid(True, ls="--", alpha=0.25, lw=0.6)

    ax_pol = fig.add_subplot(gs[1, 1], projection="polar")
    r = np.linspace(0, 1.0, n, endpoint=False)
    ax_pol.scatter(phi_x, r, s=6, c=r, cmap="viridis", alpha=0.9, linewidths=0)
    ax_pol.set_title("Polar coordinates (X-axis)", pad=10)
    ax_pol.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax_pol.set_rlabel_position(135)

    # (c) per-axis GADF examples
    ax_gx = fig.add_subplot(gs[0, 2])
    ax_gy = fig.add_subplot(gs[1, 2])
    im1 = ax_gx.imshow(gadf_x, cmap="RdYlBu_r", vmin=-1, vmax=1)
    ax_gx.set_title("(c) GADF (X-axis)", pad=6)
    ax_gx.set_xticks([])
    ax_gx.set_yticks([])

    im2 = ax_gy.imshow(gadf_y, cmap="RdYlBu_r", vmin=-1, vmax=1)
    ax_gy.set_title("GADF (Y-axis)", pad=6)
    ax_gy.set_xticks([])
    ax_gy.set_yticks([])

    # (d) tri-axis stacked RGB
    ax_rgb = fig.add_subplot(gs[:, 3])
    ax_rgb.imshow(rgb_vis)
    ax_rgb.set_title("(d) Stack tri-axis GADF as RGB\nR=X, G=Y, B=Z", pad=6)
    ax_rgb.set_xticks([])
    ax_rgb.set_yticks([])

    # shared colorbar for GADF
    cax = fig.add_axes([0.57, 0.10, 0.012, 0.35])  # [left, bottom, width, height]
    cb = fig.colorbar(im1, cax=cax)
    cb.set_label("GADF value", rotation=90)

    # short note (for paper caption guidance)
    fig.text(
        0.02,
        0.01,
        "Note: For tri-axis vibration, GAF is computed per axis, then combined by channel stacking (RGB) for a single image input.",
        fontsize=8,
    )

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved -> {out_path}")


if __name__ == "__main__":
    main()

