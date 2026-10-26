"""YOLOv8 侧导入 gaf_yolo.utils.project_paths。"""
from __future__ import annotations

import sys
from pathlib import Path

_GAF_ROOT = Path(__file__).resolve().parent.parent / "gaf_yolo"
if str(_GAF_ROOT) not in sys.path:
    sys.path.insert(0, str(_GAF_ROOT))

from utils.project_paths import *  # noqa: F403, E402
