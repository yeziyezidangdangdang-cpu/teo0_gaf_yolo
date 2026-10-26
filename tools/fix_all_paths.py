"""批量修正 TEP_2_repro 内论文相关脚本路径 → project_paths / repro_paths。"""
from __future__ import annotations

import re
from pathlib import Path

REPRO = Path(r"D:\TEP_2_repro")

# 统一替换：旧绝对路径片段 → pp 表达式（用于 gaf_yolo/src）
GAF_SUBS: list[tuple[str, str]] = [
    (r'r?"D:[/\\]TEP_2_repro[/\\]gaf_yolo[/\\]data[/\\]raw"', "str(pp.RAW)"),
    (r'r?"D:[/\\]TEP_2_repro[/\\]gaf_yolo[/\\]data[/\\]raw[/\\]CWRU[/\\]cwrn"', "str(pp.CWRU_MAT_DIR)"),
    (r'r?"D:[/\\]TEP_2_repro[/\\]gaf_yolo[/\\]data[/\\]raw[/\\]CWRU[/\\]csv"', "str(pp.CWRU_CSV_DIR)"),
    (r'r?"D:[/\\]TEP_2[/\\]gaf_yolo[/\\]data[/\\]raw[/\\]CWRU[/\\]csv"', "str(pp.CWRU_CSV_DIR)"),
    (r'r?"D:[/\\]TEP_2_repro[/\\]gaf_yolo[/\\]data[/\\]processed[/\\]final_dataset_hust\.csv"', "str(pp.FINAL_DATASET_HUST)"),
    (r'r?"D:[/\\]TEP_2_repro[/\\]gaf_yolo[/\\]data[/\\]processed[/\\]file_ranges_hust\.csv"', "str(pp.FILE_RANGES_HUST)"),
    (r'r?"D:[/\\]TEP_2_repro[/\\]gaf_yolo[/\\]data[/\\]processed[/\\]final_dataset_cwru\.csv"', "str(pp.FINAL_DATASET_CWRU)"),
    (r'r?"D:[/\\]TEP_2_repro[/\\]gaf_yolo[/\\]data[/\\]processed[/\\]stft_input_hust\.npy"', "str(pp.STFT_INPUT_HUST)"),
    (r'r?"D:[/\\]TEP_2_repro[/\\]gaf_yolo[/\\]data[/\\]processed[/\\]stft_input_cwru\.npy"', "str(pp.PROCESSED / 'stft_input_cwru.npy')"),
    (r'r?"D:[/\\]TEP_2_repro[/\\]gaf_yolo[/\\]data[/\\]processed[/\\]gaf_preprocessed_hust\.npy"', "str(pp.GAF_PREPROCESSED_HUST)"),
    (r'r?"D:[/\\]TEP_2_repro[/\\]gaf_yolo[/\\]data[/\\]processed[/\\]gaf_preprocessed_cwru\.npy"', "str(pp.GAF_PREPROCESSED_CWRU)"),
    (r'r?"D:[/\\]TEP_2_repro[/\\]gaf_yolo[/\\]data[/\\]processed[/\\]gaf_input_hust_axes\.npy"', "str(pp.GAF_INPUT_HUST_AXES)"),
    (r'r?"D:[/\\]TEP_2_repro[/\\]gaf_yolo[/\\]data[/\\]processed[/\\]gaf_input_hust_axes_env\.npy"', "str(pp.GAF_INPUT_HUST_AXES_ENV)"),
    (r'r?"D:[/\\]TEP_2_repro[/\\]gaf_yolo[/\\]data[/\\]gaf_rgb"', "str(pp.GAF_RGB_HUST)"),
    (r'r?"D:[/\\]TEP_2_repro[/\\]gaf_yolo[/\\]data[/\\]gaf_rgb[/\\]gadf"', "str(pp.GADF_RGB_HUST)"),
    (r'r?"D:[/\\]TEP_2_repro[/\\]gaf_yolo[/\\]data[/\\]gaf_rgb_env"', "str(pp.GAF_RGB_ENV_HUST)"),
    (r'r?"D:[/\\]TEP_2_repro[/\\]gaf_yolo[/\\]data[/\\]spectrograms"', "str(pp.SPECTROGRAMS)"),
    (r'r?"D:[/\\]TEP_2_repro[/\\]gaf_yolo[/\\]data[/\\]spectrograms_cwru"', "str(pp.SPECTROGRAMS_CWRU)"),
    (r'r?"D:[/\\]TEP_2_repro[/\\]gaf_yolo[/\\]data[/\\]spectrograms_mag_resid"', "str(pp.SPECTROGRAMS_MAG_RESID)"),
    (r'r?"D:[/\\]TEP_2_repro[/\\]gaf_yolo[/\\]data[/\\]spectrograms_mv"', "str(pp.PROCESSED.parent / 'spectrograms_mv')"),
    (r'r?"D:[/\\]TEP_2_repro[/\\]gaf_yolo[/\\]data[/\\]spec_x"', "str(pp.SPEC_X)"),
    (r'r?"D:[/\\]TEP_2_repro[/\\]gaf_yolo[/\\]data[/\\]spec_y"', "str(pp.SPEC_Y)"),
    (r'r?"D:[/\\]TEP_2_repro[/\\]gaf_yolo[/\\]data[/\\]spec_z"', "str(pp.SPEC_Z)"),
    (r'r?"D:[/\\]TEP_2_repro[/\\]gaf_yolo[/\\]data[/\\]images_enhanced"', "str(pp.IMAGES_ENHANCED)"),
    (r'r?"D:[/\\]TEP_2_repro[/\\]gaf_yolo[/\\]yolo_datasets"', "str(pp.YOLO_DS)"),
    (r'r?"D:[/\\]TEP_2[/\\]gaf_yolo[/\\]yolo_datasets"', "str(pp.YOLO_DS)"),
    (r'r?"D:[/\\]TEP_2_repro[/\\]gaf_yolo[/\\]data[/\\]diagnostics[/\\]hust_waveforms_raw"', "str(pp.DIAGNOSTICS / 'hust_waveforms_raw')"),
    (r'r?"D:[/\\]TEP_2[/\\]wave_3_00792_xyz\.png"', "str(pp.PAPER_DRAW_ROOT / 'output' / 'wave_3_00792_xyz.png')"),
    (r'r?"D:[/\\]TEP_2[/\\]wave_3_00792_xyz_notext\.png"', "str(pp.PAPER_DRAW_ROOT / 'output' / 'wave_3_00792_xyz_notext.png')"),
    (r'r?"D:[/\\]TEP_2[/\\]wave_3_00792_single\.png"', "str(pp.PAPER_DRAW_ROOT / 'output' / 'wave_3_00792_single.png')"),
]

YOLO_SUBS: list[tuple[str, str]] = [
    (r'r?"D:[/\\]TEP_2_repro[/\\]gaf_yolo[/\\]yolo_datasets[/\\]spec_cls_strict"', "str(pp.YOLO_SPEC_STRICT)"),
    (r'r?"D:[/\\]TEP_2_repro[/\\]gaf_yolo[/\\]yolo_datasets[/\\]spec_cls"', "str(pp.YOLO_DS / 'spec_cls')"),
    (r'r?"D:[/\\]TEP_2_repro[/\\]gaf_yolo[/\\]yolo_datasets[/\\]hust_spec_mag_resid_cls"', "str(pp.YOLO_DS / 'hust_spec_mag_resid_cls')"),
    (r'r?"D:[/\\]TEP_2_repro[/\\]gaf_yolo[/\\]yolo_datasets[/\\]hust_spec_mv_cls"', "str(pp.YOLO_DS / 'hust_spec_mv_cls')"),
    (r'r?"D:[/\\]TEP_2_repro[/\\]gaf_yolo[/\\]yolo_datasets[/\\]hust_spec_x_cls"', "str(pp.YOLO_DS / 'hust_spec_x_cls')"),
    (r'r?"D:[/\\]TEP_2_repro[/\\]gaf_yolo[/\\]yolo_datasets[/\\]hust_spec_y_cls"', "str(pp.YOLO_DS / 'hust_spec_y_cls')"),
    (r'r?"D:[/\\]TEP_2_repro[/\\]gaf_yolo[/\\]yolo_datasets[/\\]hust_spec_z_cls"', "str(pp.YOLO_DS / 'hust_spec_z_cls')"),
    (r'r?"D:[/\\]TEP_2_repro[/\\]gaf_yolo[/\\]yolo_datasets[/\\]hust_gasf_rgb_cls"', "str(pp.YOLO_DS / 'hust_gasf_rgb_cls')"),
    (r'r?"D:[/\\]TEP_2_repro[/\\]gaf_yolo[/\\]yolo_datasets[/\\]hust_gadf_rgb_cls"', "str(pp.YOLO_DS / 'hust_gadf_rgb_cls')"),
    (r'r?"D:[/\\]TEP_2_repro[/\\]gaf_yolo[/\\]yolo_datasets_v2[/\\]gasf_cls_strict"', "str(pp.YOLO_GASF_STRICT_V2)"),
    (r'r?"D:[/\\]TEP_2_repro[/\\]gaf_yolo[/\\]yolo_datasets_v2[/\\]gadf_cls_strict"', "str(pp.YOLO_GADF_STRICT_V2)"),
    (r'r?"D:[/\\]TEP_2_repro[/\\]gaf_yolo[/\\]data[/\\]spectrograms_mag_resid"', "str(pp.SPECTROGRAMS_MAG_RESID)"),
    (r'r?"D:[/\\]TEP_2_repro[/\\]gaf_yolo[/\\]data[/\\]spectrograms_mv"', "str(pp.ARCHIVE_GAF / 'data' / 'spectrograms_mv')"),
    (r'r?"D:[/\\]TEP_2_repro[/\\]gaf_yolo[/\\]data[/\\]spectrograms"', "str(pp.SPECTROGRAMS)"),
    (r'Path\(r"D:[/\\]TEP_2[/\\]YOLOv8[/\\]runs[/\\]classify[/\\]runs[/\\]hust_spec[/\\]hust_spec_cls[/\\]weights[/\\]best\.pt"\),\n\s*', ""),
]

PP_HEADER_GAF = '''
import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
import utils.project_paths as pp  # noqa: E402
'''

PP_HEADER_YOLO = '''
import repro_paths as pp  # noqa: E402
'''


def _ensure_import(text: str, marker: str, header: str) -> str:
    if marker in text:
        return text
    lines = text.splitlines(keepends=True)
    insert = 0
    if lines and lines[0].startswith('"""'):
        for i, ln in enumerate(lines[1:], 1):
            if '"""' in ln:
                insert = i + 1
                break
    # after from __future__ if present
    for i, ln in enumerate(lines):
        if ln.strip().startswith("from __future__"):
            insert = max(insert, i + 1)
    return "".join(lines[:insert]) + header + "".join(lines[insert:])


def patch_file(path: Path, subs: list[tuple[str, str]], marker: str, header: str) -> bool:
    text = path.read_text(encoding="utf-8")
    orig = text
    if not any(re.search(p.replace("pp.", ""), text) for p, _ in subs) and "TEP_2" not in text:
        return False
    for pat, repl in subs:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    if "TEP_2_repro/gaf_yolo/data/processed" in text or "TEP_2_repro\\gaf_yolo\\data\\processed" in text:
        text = _ensure_import(text, marker, header)
    if "TEP_2_repro/gaf_yolo" in text or "TEP_2\\gaf_yolo" in text or "TEP_2_repro\\gaf_yolo" in text:
        text = _ensure_import(text, marker, header)
    if text != orig:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> None:
    ng = ny = 0
    archive_non_paper = {
        "0.0_prepare_vacuum_dataset.py",
        "0.0_prepare_vibration_machines_dataset.py",
        "0.0_prepare_vibration_machines_dataset_xyz.py",
        "0.0_prepare_subf_dataset.py",
        "1.0_areemd_preprocess_subf.py",
        "1.1_preprocess_vibration_machines.py",
        "1.1_preprocess_vibration_machines_xyz.py",
        "1.2_generate_gaf_vibration_machines_images.py",
        "1.2_generate_gaf_vibration_machines_xyz_rgb_images.py",
        "1.3_generate_gaf_vibration_machines_enhanced_images.py",
        "3.0_visualize_subf_preprocessed_signals.py",
        "3.1_visualize_preprocessed_spectra.py",
        "3.1_extract_car_features_subf.py",
        "3.2_train_wknn_subf.py",
    }
    arch_dir = REPRO / "gaf_yolo" / "src" / "_archive_non_paper"
    arch_dir.mkdir(parents=True, exist_ok=True)
    for name in archive_non_paper:
        src = REPRO / "gaf_yolo" / "src" / name
        if src.is_file():
            dst = arch_dir / name
            if not dst.exists():
                src.replace(dst)
                print("archived", name)

    for py in sorted((REPRO / "gaf_yolo" / "src").glob("*.py")):
        if patch_file(py, GAF_SUBS, "import utils.project_paths as pp", PP_HEADER_GAF):
            ng += 1
            print("gaf", py.name)

    for py in sorted((REPRO / "YOLOv8").glob("*.py")):
        if py.name in {"repro_paths.py"}:
            continue
        text = py.read_text(encoding="utf-8")
        if "TEP_2_repro/gaf_yolo" in text or "TEP_2_repro\\gaf_yolo" in text or 'r"D:\\TEP_2\\gaf_yolo' in text:
            orig = text
            for pat, repl in YOLO_SUBS:
                text = re.sub(pat, repl, text, flags=re.IGNORECASE)
            if "import repro_paths as pp" not in text and any(
                x in text for x in ("str(pp.", "pp.YOLO", "pp.SPECTROGRAMS", "pp.GASF")
            ):
                text = _ensure_import(text, "import repro_paths as pp", PP_HEADER_YOLO)
            if text != orig:
                py.write_text(text, encoding="utf-8")
                ny += 1
                print("yolo", py.name)

    print(f"patched gaf={ng} yolo={ny}")


if __name__ == "__main__":
    main()
