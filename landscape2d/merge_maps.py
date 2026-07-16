from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import TextIO

import numpy as np

from .io import load_matrix


@dataclass(frozen=True)
class MergeResult:
    group_id: str
    output_path: Path
    source_paths: list[Path]
    shape: tuple[int, int]


def merge_patient_maps(
    input_dir: str | Path,
    prefix_length: int,
    blank_cols: int = 1,
    output_suffix: str = "_merged",
    delete_originals: bool = True,
    dry_run: bool = False,
    progress: bool = True,
    progress_stream: TextIO | None = None,
) -> list[MergeResult]:
    if prefix_length <= 0:
        raise ValueError("prefix_length must be positive")
    if blank_cols < 0:
        raise ValueError("blank_cols must be non-negative")

    stream = progress_stream or sys.stderr
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not input_path.is_dir():
        raise NotADirectoryError(input_path)

    files = sorted(
        path
        for path in input_path.iterdir()
        if path.suffix == ".npz" and not path.stem.endswith(output_suffix)
    )
    groups: dict[str, list[Path]] = {}
    for path in files:
        groups.setdefault(path.stem[:prefix_length], []).append(path)

    results: list[MergeResult] = []
    merge_groups = [(group_id, paths) for group_id, paths in groups.items() if len(paths) > 1]
    for index, (group_id, paths) in enumerate(merge_groups, start=1):
        _show_progress(
            progress,
            stream,
            f"Merging group {index}/{len(merge_groups)} {group_id}: {len(paths)} files",
        )
        matrices = [load_matrix(path) for path in paths]
        merged = merge_matrices_horizontally(matrices, blank_cols=blank_cols)
        output_path = input_path / f"{group_id}{output_suffix}.npz"
        result = MergeResult(
            group_id=group_id,
            output_path=output_path,
            source_paths=paths,
            shape=tuple(int(value) for value in merged.shape),
        )
        results.append(result)
        if dry_run:
            continue
        np.savez(
            output_path,
            matrix=merged,
            labels=merged,
            source_files=np.asarray([str(path) for path in paths]),
            group_id=np.asarray([group_id]),
            blank_cols=np.asarray([blank_cols], dtype=np.int64),
        )
        if delete_originals:
            for path in paths:
                path.unlink()
    _show_progress(progress, stream, f"Merged groups: {len(results)}")
    return results


def merge_matrices_horizontally(
    matrices: list[np.ndarray],
    blank_cols: int = 1,
) -> np.ndarray:
    if not matrices:
        raise ValueError("at least one matrix is required")
    validated = [np.asarray(matrix, dtype=np.int64) for matrix in matrices]
    if any(matrix.ndim != 2 for matrix in validated):
        raise ValueError("all matrices must be 2D")
    max_rows = max(matrix.shape[0] for matrix in validated)
    total_cols = sum(matrix.shape[1] for matrix in validated)
    total_cols += blank_cols * (len(validated) - 1)
    merged = np.zeros((max_rows, total_cols), dtype=np.int64)
    col = 0
    for matrix in validated:
        rows, cols = matrix.shape
        merged[:rows, col : col + cols] = matrix
        col += cols + blank_cols
    return merged


def _show_progress(progress: bool, stream: TextIO, message: str) -> None:
    if progress:
        print(f"[landscape2d-merge] {message}", file=stream, flush=True)
