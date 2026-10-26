"""Stratified per-file index selection for CWRU GAF image generation."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np


def load_file_ranges(csv_path: Path) -> List[Tuple[int, int, int]]:
    ranges: List[Tuple[int, int, int]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ranges.append((int(row["file_id"]), int(row["start_idx"]), int(row["end_idx"])))
    ranges.sort(key=lambda x: x[1])
    return ranges


def build_idx_to_file(ranges: Iterable[Tuple[int, int, int]]) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for fid, start, end in ranges:
        for idx in range(start, end):
            mapping[idx] = fid
    return mapping


def files_per_label(ranges: List[Tuple[int, int, int]], labels: np.ndarray) -> Dict[int, Set[int]]:
    out: Dict[int, Set[int]] = defaultdict(set)
    for fid, start, _end in ranges:
        out[int(labels[start])].add(fid)
    return dict(out)


def select_indices_stratified(
    labels: np.ndarray,
    idx_to_file: Dict[int, int],
    files_by_label: Dict[int, Set[int]],
    file_ranges: List[Tuple[int, int, int]],
    max_per_label: int,
    keep_mod: int,
    keep_pattern: Set[int],
) -> List[int]:
    """
    Select global row indices for GAF image generation.

    Fixes two historical bugs:
    1) Global per-label cap exhausted quota on the first source file only.
    2) Taking the earliest passing indices inside each file only covered the
       train temporal zone, leaving strict val/test splits empty.
    """
    cap_per_file: Dict[int, int] = {}
    for lab, file_ids in files_by_label.items():
        n_files = max(len(file_ids), 1)
        cap_per_file[lab] = max(1, max_per_label // n_files)

    selected: List[int] = []
    label_totals: Dict[int, int] = defaultdict(int)

    for lab, file_ids in sorted(files_by_label.items()):
        for fid in sorted(file_ids):
            if label_totals[lab] >= max_per_label:
                break
            start = end = None
            for range_fid, range_start, range_end in file_ranges:
                if range_fid == fid:
                    start, end = range_start, range_end
                    break
            if start is None or end is None:
                continue

            candidates = [
                idx
                for idx in range(start, end)
                if int(labels[idx]) == lab and (idx % keep_mod) in keep_pattern
            ]
            if not candidates:
                continue

            quota = min(cap_per_file[lab], max_per_label - label_totals[lab])
            if len(candidates) <= quota:
                picked = candidates
            else:
                # Evenly spaced picks span train/val/test zones inside the file.
                positions = np.linspace(0, len(candidates) - 1, quota, dtype=int)
                picked = [candidates[int(pos)] for pos in positions]

            selected.extend(picked)
            label_totals[lab] += len(picked)

    selected.sort()
    return selected


def summarize_selection(
    selected: Iterable[int],
    labels: np.ndarray,
    idx_to_file: Dict[int, int],
) -> Dict[Tuple[int, int], int]:
    counts: Dict[Tuple[int, int], int] = defaultdict(int)
    for idx in selected:
        counts[(int(labels[idx]), idx_to_file[idx])] += 1
    return dict(counts)
