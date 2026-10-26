"""向 Log.docx 追加混合存储说明（保留原有内容）。"""
from __future__ import annotations

import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

LOG = Path(r"D:\TEP_2_repro\Log.docx")
APPEND = """
【2026-10 混合存储架构】
• 代码 / 权重 / 评估结果 / 原始数据 raw：D:\\TEP_2_repro
• 大体积中间数据（processed、GAF 图、yolo_datasets*）：D:\\TEP_2\\gaf_yolo
• 统一路径：gaf_yolo\\utils\\project_paths.py、YOLOv8\\repro_paths.py
• 验证：python tools\\verify_repro.py；表编译：paper_supplementary\\src\\compile_ch34_tables.py
• 非论文脚本已移至 gaf_yolo\\src\\_archive_non_paper\\
""".strip()


def append_paragraphs(docx_path: Path, lines: list[str]) -> None:
    with zipfile.ZipFile(docx_path, "r") as zin:
        xml = zin.read("word/document.xml")
        other = {n: zin.read(n) for n in zin.namelist() if n != "word/document.xml"}

    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    ET.register_namespace("w", ns["w"])
    root = ET.fromstring(xml)
    body = root.find("w:body", ns)
    if body is None:
        raise RuntimeError("invalid docx")

    for line in lines:
        p = ET.SubElement(body, f"{{{ns['w']}}}p")
        r = ET.SubElement(p, f"{{{ns['w']}}}r")
        t = ET.SubElement(r, f"{{{ns['w']}}}t")
        t.text = line
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")

    new_xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    with zipfile.ZipFile(docx_path, "w", zipfile.ZIP_DEFLATED) as zout:
        for n, data in other.items():
            zout.writestr(n, data)
        zout.writestr("word/document.xml", new_xml)


def main() -> None:
    if not LOG.is_file():
        print("Log.docx missing")
        return
    append_paragraphs(LOG, [""] + APPEND.split("\n"))
    print("Appended hybrid storage section to", LOG)


if __name__ == "__main__":
    main()
