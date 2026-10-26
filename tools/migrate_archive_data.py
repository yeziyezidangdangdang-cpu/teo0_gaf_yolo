"""
将 repro 内大体积中间数据迁移到 D:\\TEP_2，并删除 repro 副本（保留 raw 与权重）。

用法: python tools/migrate_archive_data.py [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

REPRO_GAF = Path(r"D:\TEP_2_repro\gaf_yolo")
ARCHIVE_GAF = Path(r"D:\TEP_2\gaf_yolo")

MOVE_DIRS = [
    "data/gaf_rgb_mcc5",
    "data/processed_mcc5",
    "data/images_v2",
    "data/images_enhanced_v2",
    "yolo_datasets_mcc5",
    "yolo_datasets_v2",
    "yolo_datasets_v2_enhanced",
]

DELETE_IF_ARCHIVE_OK = [
    REPRO_GAF / "data/gaf_rgb",
    REPRO_GAF / "data/processed",
]

DELETE_PLACEHOLDERS = [
    REPRO_GAF / "data/images",
    REPRO_GAF / "data/spectrograms",
    REPRO_GAF / "data/spectrograms_cwru",
    REPRO_GAF / "data/spec_x",
    REPRO_GAF / "data/spec_y",
    REPRO_GAF / "data/spec_z",
    REPRO_GAF / "data/gaf_rgb_env",
    REPRO_GAF / "data/images_enhanced",
    REPRO_GAF / "data/diagnostics",
    REPRO_GAF / "yolo_datasets_enhanced",
]


def dir_stats(p: Path) -> tuple[int, float]:
    if not p.exists():
        return 0, 0.0
    if p.is_symlink():
        return 1, 0.0
    n = sz = 0
    for f in p.rglob("*"):
        if f.is_file():
            n += 1
            sz += f.stat().st_size
    return n, sz / 1024 / 1024


def remove_path(p: Path, dry: bool) -> None:
    if not p.exists() and not p.is_symlink():
        return
    print(f"  remove {p}")
    if dry:
        return
    try:
        if p.is_symlink() or p.is_file():
            p.unlink()
        else:
            shutil.rmtree(p)
    except OSError:
        try:
            p.unlink()
        except OSError:
            subprocess.run(["cmd", "/c", "rmdir", "/S", "/Q", str(p)], check=False)


def robocopy_merge(src: Path, dst: Path, dry: bool) -> None:
    if not src.exists():
        print(f"  skip missing {src}")
        return
    n, mb = dir_stats(src)
    if n == 0:
        print(f"  skip empty {src}")
        return
    print(f"  merge {src} ({n} files, {mb:.0f} MB) -> {dst}")
    if dry:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["robocopy", str(src), str(dst), "/E", "/NFL", "/NDL", "/NJH", "/NJS", "/nc", "/ns", "/np"],
        check=False,
    )
    remove_path(src, dry=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for rel in MOVE_DIRS:
        src = REPRO_GAF.joinpath(*rel.split("/"))
        dst = ARCHIVE_GAF.joinpath(*rel.split("/"))
        if dst.exists() and dir_stats(dst)[0] >= dir_stats(src)[0] and dir_stats(src)[0] > 0:
            print(f"  archive already has {rel}, drop repro copy")
            remove_path(src, args.dry_run)
        else:
            robocopy_merge(src, dst, args.dry_run)

    repro_yolo = REPRO_GAF / "yolo_datasets"
    arch_yolo = ARCHIVE_GAF / "yolo_datasets"
    if repro_yolo.exists() and dir_stats(repro_yolo)[0] > 0:
        print("  merge yolo_datasets random splits -> archive")
        if not args.dry_run:
            subprocess.run(
                ["robocopy", str(repro_yolo), str(arch_yolo), "/E", "/NFL", "/NDL", "/NJH", "/NJS", "/nc", "/ns", "/np"],
                check=False,
            )
            remove_path(repro_yolo, False)
            repro_yolo.mkdir(parents=True, exist_ok=True)
            (repro_yolo / "README.txt").write_text(
                "YOLO datasets: D:\\TEP_2\\gaf_yolo\\yolo_datasets\nSee project_paths.py\n",
                encoding="utf-8",
            )

    for p in DELETE_IF_ARCHIVE_OK:
        if not p.exists() and not p.is_symlink():
            continue
        arch_name = "data/gaf_rgb" if p.name == "gaf_rgb" else "data/processed"
        arch = ARCHIVE_GAF.joinpath(*arch_name.split("/"))
        if arch.exists() and dir_stats(arch)[0] > 10:
            print(f"  drop repro duplicate {p.name} (archive ok)")
            remove_path(p, args.dry_run)
        else:
            print(f"  keep {p} (archive incomplete)")

    for p in DELETE_PLACEHOLDERS:
        if p.exists() or p.is_symlink():
            remove_path(p, args.dry_run)

    print("done")


if __name__ == "__main__":
    main()
