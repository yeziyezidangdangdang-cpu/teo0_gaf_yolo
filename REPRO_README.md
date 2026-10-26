# TEP_2_repro — 论文复现（轻量代码仓）

## 存储分工

| 位置 | 内容 |
|------|------|
| **`D:\TEP_2_repro`** | 代码、`gaf_yolo/data/raw` 原始数据、YOLO 权重与 `runs/` 评估结果、`paper_draw`、`paper_supplementary`、`Log.docx` |
| **`D:\TEP_2`** | 大体积中间数据：`processed/` NPY/CSV、`gaf_rgb/`、`*spectrograms*`、`yolo_datasets*` 等 |

所有路径由 `gaf_yolo/utils/project_paths.py` 与 `YOLOv8/repro_paths.py` 统一管理。

## 快速验证

```powershell
cd D:\TEP_2_repro
python tools\verify_repro.py
python paper_supplementary\src\compile_ch34_tables.py
```

## 论文主线命令（在 repro 根目录相对路径）

详见 `Log.docx`。数据集自动从 `D:\TEP_2` 读取。

```powershell
cd D:\TEP_2_repro\YOLOv8
python eval_mcc5_dual.py --exp-point b
python eval_hust_gaf_dual.py --exp-point main
```

## 维护工具

- `tools/migrate_archive_data.py` — 将 repro 内误存的大文件迁到 TEP_2
- `tools/fix_all_paths.py` — 批量修正硬编码路径
- `tools/cleanup_repro_root.py` — 清理根目录杂项

## GitHub 上传

勿提交：`gaf_yolo/data/raw` 若过大可改用 Git LFS；`YOLOv8/runs/**/weights/*.pt` 按需 LFS。中间数据始终在本地 `D:\TEP_2`。
