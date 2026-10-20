from __future__ import annotations

from pathlib import Path
import sys
from typing import TextIO

import pandas as pd
import numpy as np
from numbers import Integral

from .io import (
    iter_volume_files,
    load_volume,
    parse_classes,
    parse_spacing,
    sample_id_from_path,
)
from .metrics import compute_metrics


def compute_batch(
    input_dir: str | Path,
    output_csv: str | Path,
    classes: list[int] | str | None = None,
    spacing: tuple[float, float, float] | str = (1, 1, 1),
    connectivity: int = 6,
    min_patch_voxels: int = 0,
    min_patch_volume: float = 0.0,
    min_non_background_voxels: int = 0,
    min_non_background_volume: float = 0.0,
    progress: bool = False,
    progress_stream: TextIO | None = None,
) -> pd.DataFrame:
    stream = progress_stream or sys.stderr
    if not isinstance(min_patch_voxels, Integral) or min_patch_voxels < 0:
        raise ValueError("min_patch_voxels must be non-negative")
    if not np.isfinite(min_patch_volume) or min_patch_volume < 0:
        raise ValueError("min_patch_volume must be non-negative")
    if not isinstance(min_non_background_voxels, Integral) or min_non_background_voxels < 0:
        raise ValueError("min_non_background_voxels must be non-negative")
    if not np.isfinite(min_non_background_volume) or min_non_background_volume < 0:
        raise ValueError("min_non_background_volume must be non-negative")

    files = [path for path in iter_volume_files(input_dir) if path.resolve() != Path(output_csv).resolve()]
    if not files:
        raise ValueError(f"no .npy, .npz, .nii, or .nii.gz files found in {input_dir}")

    class_values = parse_classes(classes)
    spacing_tuple = parse_spacing(spacing)
    if class_values is None:
        observed: set[int] = set()
        for path in files:
            observed.update(int(value) for value in np.unique(load_volume(path)) if value > 0)
        class_values = sorted(observed)

    rows = []
    skipped_count = 0
    for index, path in enumerate(files, start=1):
        volume = load_volume(path)
        sample_voxels = int((volume > 0).sum())
        sample_volume = sample_voxels * float(spacing_tuple[0] * spacing_tuple[1] * spacing_tuple[2])
        if (
            sample_voxels < min_non_background_voxels
            or sample_volume < min_non_background_volume
        ):
            skipped_count += 1
            _show_progress(
                progress,
                stream,
                (
                    f"Skipping sample: {index}/{len(files)} {path.name} "
                    f"(non_background_voxels={sample_voxels}, "
                    f"non_background_volume={sample_volume:g})"
                ),
            )
            continue

        _show_progress(
            progress,
            stream,
            f"Computing metrics: {index}/{len(files)} {path.name}",
        )
        row = {"sample_id": sample_id_from_path(path), "source_file": str(path)}
        row.update(
            compute_metrics(
                volume,
                classes=class_values,
                spacing=spacing_tuple,
                connectivity=connectivity,
                min_patch_voxels=min_patch_voxels,
                min_patch_volume=min_patch_volume,
            )
        )
        rows.append(row)

    if not rows:
        raise ValueError(
            "all samples were filtered out; lower the minimum non-background "
            "voxel/volume thresholds"
        )

    dataframe = pd.DataFrame(rows)
    ordered_columns = ["sample_id", "source_file"] + sorted(
        column for column in dataframe.columns if column not in {"sample_id", "source_file"}
    )
    dataframe = dataframe[ordered_columns]
    dataframe.to_csv(output_csv, index=False)
    _show_progress(
        progress,
        stream,
        f"Saved metrics: {output_csv} ({len(rows)} kept, {skipped_count} skipped)",
    )
    return dataframe


def _show_progress(progress: bool, stream: TextIO, message: str) -> None:
    if progress:
        print(f"[landscape3d] {message}", file=stream, flush=True)
