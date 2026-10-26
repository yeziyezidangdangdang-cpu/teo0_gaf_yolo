"""清理 TEP_2_repro 根目录杂项文件。"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(r"D:\TEP_2_repro")
DOCS = ROOT / "docs" / "notes"
DOCS.mkdir(parents=True, exist_ok=True)

MOVE_TXT = [
    "_bearing_ch34_outline.txt",
    "_bearing_ch3_body.txt",
    "_bearing_ch3_outline.txt",
    "_bearing_ch3_tail.txt",
    "_bearing_ch4_context.txt",
    "_bearing_full_extract.txt",
    "_log_cwru_search.txt",
    "_log_extract.txt",
    "_log_tail.txt",
]

DELETE = [
    "~$Log.docx",
    "read.py",
]

DELETE_DIRS = [".idea"]


def main() -> None:
    for name in MOVE_TXT:
        src = ROOT / name
        if src.is_file():
            dst = DOCS / name
            src.replace(dst)
            print("moved", name, "-> docs/notes/")
    for name in DELETE:
        p = ROOT / name
        if p.is_file():
            p.unlink()
            print("deleted", name)
    for name in DELETE_DIRS:
        p = ROOT / name
        if p.is_dir():
            shutil.rmtree(p)
            print("deleted dir", name)


if __name__ == "__main__":
    main()
