from __future__ import annotations

from pathlib import Path
import sys
from typing import TextIO

import pandas as pd

from .io import (
    iter_matrix_files,
    load_matrix,
    parse_classes,
    parse_spacing,
    sample_id_from_path,
)
from .metrics import compute_metrics
from .visualization import save_label_png


def compute_batch(
    input_dir: str | Path,
    output_csv: str | Path,
    classes: list[int] | str | None = None,
    spacing: tuple[float, float] | str = (1, 1),
    connectivity: int = 4,
    search_radius: float = float("inf"),
    contrast_weights: dict[tuple[int, int], float] | None = None,
    max_classes: int | None = None,
    min_patch_cells: int = 0,
    plot_dir: str | Path | None = None,
    progress: bool = False,
    progress_stream: TextIO | None = None,
) -> pd.DataFrame:
    stream = progress_stream or sys.stderr
    files = iter_matrix_files(input_dir)
    if not files:
        raise ValueError(f"no .npy, .npz, .csv, or .txt files found in {input_dir}")

    class_values = parse_classes(classes)
    spacing_tuple = parse_spacing(spacing)
    matrices = []
    for index, path in enumerate(files, start=1):
        _show_progress(
            progress,
            stream,
            f"Loading matrices: {index}/{len(files)} {path.name}",
        )
        matrices.append((path, load_matrix(path)))
    if class_values is None:
        max_class = max(int(matrix.max(initial=0)) for _, matrix in matrices)
        class_values = list(range(1, max_class + 1))

    rows = []
    plot_path = Path(plot_dir) if plot_dir is not None else None
    if plot_path is not None:
        plot_path.mkdir(parents=True, exist_ok=True)
    for index, (path, matrix) in enumerate(matrices, start=1):
        _show_progress(
            progress,
            stream,
            f"Computing metrics: {index}/{len(matrices)} {path.name}",
        )
        row = {"sample_id": sample_id_from_path(path), "source_file": str(path)}
        row.update(
            compute_metrics(
                matrix,
                classes=class_values,
                spacing=spacing_tuple,
                connectivity=connectivity,
                search_radius=search_radius,
                contrast_weights=contrast_weights,
                max_classes=max_classes,
                min_patch_cells=min_patch_cells,
            )
        )
        rows.append(row)
        if plot_path is not None:
            image_path = plot_path / f"{sample_id_from_path(path)}.png"
            save_label_png(
                matrix,
                image_path,
                classes=class_values,
                min_patch_cells=min_patch_cells,
                connectivity=connectivity,
            )
            _show_progress(progress, stream, f"Saved plot: {image_path}")

    dataframe = pd.DataFrame(rows)
    ordered_columns = ["sample_id", "source_file"] + sorted(
        column for column in dataframe.columns if column not in {"sample_id", "source_file"}
    )
    dataframe = dataframe[ordered_columns]
    dataframe.to_csv(output_csv, index=False)
    _show_progress(progress, stream, f"Saved metrics: {output_csv}")
    return dataframe


def _show_progress(progress: bool, stream: TextIO, message: str) -> None:
    if progress:
        print(f"[landscape2d] {message}", file=stream, flush=True)
