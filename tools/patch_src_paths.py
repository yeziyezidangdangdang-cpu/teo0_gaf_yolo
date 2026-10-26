"""
一次性：将 gaf_yolo/src 中硬编码的 TEP_2 / TEP_2_repro 数据路径改为 project_paths 变量。
"""
from __future__ import annotations

import re
from pathlib import Path

SRC = Path(r"D:\TEP_2_repro\gaf_yolo\src")

# (regex pattern, replacement expr using pp.*)
REPLACEMENTS: list[tuple[str, str]] = [
    (r'r?"D:/TEP_2_repro/gaf_yolo/data/processed/final_dataset_hust\.csv"', "str(pp.FINAL_DATASET_HUST)"),
    (r'r?"D:\\\\TEP_2_repro\\\\gaf_yolo\\\\data\\\\processed\\\\final_dataset_hust\.csv"', "str(pp.FINAL_DATASET_HUST)"),
    (r'r?"D:/TEP_2/gaf_yolo/data/processed/final_dataset_hust\.csv"', "str(pp.FINAL_DATASET_HUST)"),
    (r'r?"D:/TEP_2_repro/gaf_yolo/data/processed/stft_input_hust\.npy"', "str(pp.STFT_INPUT_HUST)"),
    (r'r?"D:/TEP_2_repro/gaf_yolo/data/processed/gaf_preprocessed_hust\.npy"', "str(pp.GAF_PREPROCESSED_HUST)"),
    (r'r?"D:/TEP_2_repro/gaf_yolo/data/processed/gaf_input_hust_axes\.npy"', "str(pp.GAF_INPUT_HUST_AXES)"),
    (r'r?"D:/TEP_2_repro/gaf_yolo/data/processed/gaf_input_hust_axes_env\.npy"', "str(pp.GAF_INPUT_HUST_AXES_ENV)"),
    (r'r?"D:/TEP_2_repro/gaf_yolo/data/gaf_rgb"', "str(pp.GAF_RGB_HUST)"),
    (r'r?"D:/TEP_2_repro/gaf_yolo/data/gaf_rgb_env"', "str(pp.GAF_RGB_ENV_HUST)"),
    (r'r?"D:/TEP_2_repro/gaf_yolo/data/spectrograms"', "str(pp.SPECTROGRAMS)"),
    (r'r?"D:/TEP_2_repro/gaf_yolo/data/spectrograms_mag_resid"', "str(pp.SPECTROGRAMS_MAG_RESID)"),
    (r'r?"D:/TEP_2_repro/gaf_yolo/data/diagnostics/hust_waveforms_raw"', "str(pp.DIAGNOSTICS / 'hust_waveforms_raw')"),
    (r'r?"D:/TEP_2_repro/gaf_yolo/yolo_datasets"', "str(pp.YOLO_DS)"),
    (r'r?"D:/TEP_2/gaf_yolo/yolo_datasets"', "str(pp.YOLO_DS)"),
    (r'r?"D:/TEP_2/gaf_yolo/data/processed/', 'str(pp.PROCESSED / "'),
]

HEADER = '''\
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import utils.project_paths as pp  # noqa: E402
'''

PP_MARKER = "import utils.project_paths as pp"


def needs_pp(text: str) -> bool:
    return any(
        s in text
        for s in (
            "TEP_2_repro/gaf_yolo/data",
            "TEP_2\\gaf_yolo\\data",
            "TEP_2/gaf_yolo/data",
            "TEP_2\\gaf_yolo\\yolo",
            "TEP_2/gaf_yolo/yolo",
        )
    )


def ensure_pp_import(text: str) -> str:
    if PP_MARKER in text:
        return text
    # insert after first block of docstring + existing sys.path if any
    lines = text.splitlines(keepends=True)
    insert_at = 0
    if lines and lines[0].startswith('"""'):
        for i, ln in enumerate(lines[1:], 1):
            if '"""' in ln:
                insert_at = i + 1
                break
    elif lines and lines[0].startswith("#"):
        insert_at = 1
    return "".join(lines[:insert_at]) + HEADER + "".join(lines[insert_at:])


def patch_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if not needs_pp(text):
        return False
    orig = text
    text = ensure_pp_import(text)
    for pat, repl in REPLACEMENTS:
        text = re.sub(pat, repl, text)
    if text != orig:
        path.write_text(text, encoding="utf-8")
        print("patched", path.name)
        return True
    return False


def main() -> None:
    n = 0
    for py in sorted(SRC.glob("*.py")):
        if patch_file(py):
            n += 1
    print("done,", n, "files")


if __name__ == "__main__":
    main()
