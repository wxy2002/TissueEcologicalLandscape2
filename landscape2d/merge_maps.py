from __future__ import annotations

import csv
import json
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import tempfile
from typing import TextIO
import warnings

import numpy as np

from .io import iter_matrix_files, load_label_map, validate_matrix


@dataclass(frozen=True)
class MergeResult:
    group_id: str
    output_path: Path
    source_paths: list[Path]
    shape: tuple[int, int]


def merge_patient_maps(
    input_dir: str | Path,
    prefix_length: int | None = None,
    blank_cols: int = 1,
    output_suffix: str = "_merged",
    delete_originals: bool = False,
    dry_run: bool = False,
    progress: bool = True,
    progress_stream: TextIO | None = None,
    patient_manifest: str | Path | None = None,
) -> list[MergeResult]:
    """Create disjoint-slide maps; existing outputs are never overwritten.

    A CSV with ``source_file,patient_id`` is the preferred grouping contract.
    Prefix grouping is an explicit, dataset-specific fallback. Maps lacking a
    common model_id and physical spacing are rejected before any writes.
    """
    if (prefix_length is None) == (patient_manifest is None):
        raise ValueError("provide exactly one of patient_manifest or prefix_length")
    if prefix_length is not None and (not isinstance(prefix_length, int) or prefix_length <= 0):
        raise ValueError("prefix_length must be a positive integer")
    if not isinstance(blank_cols, int) or blank_cols < 0:
        raise ValueError("blank_cols must be a non-negative integer")
    if not output_suffix or Path(output_suffix).name != output_suffix or "/" in output_suffix or "\\" in output_suffix:
        raise ValueError("output_suffix must be a non-empty filename suffix")

    stream = progress_stream or sys.stderr
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not input_path.is_dir():
        raise NotADirectoryError(input_path)
    matrix_files = [path for path in iter_matrix_files(input_path)
                    if patient_manifest is None or path.resolve() != Path(patient_manifest).resolve()]
    map_manifest = input_path / "label_maps.json"
    previous_manifest = map_manifest.read_bytes() if map_manifest.exists() else None

    if patient_manifest is not None:
        groups = _manifest_groups(input_path, Path(patient_manifest))
    else:
        warnings.warn(
            "Prefix grouping must be verified for this dataset; prefer a patient manifest.",
            UserWarning,
            stacklevel=2,
        )
        files = sorted(
            path for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() == ".npz"
            and not path.stem.endswith(output_suffix) and path.name != "all_clusters.npz"
        )
        groups: dict[str, list[Path]] = {}
        for path in files:
            groups.setdefault(path.stem[:prefix_length], []).append(path)

    merge_groups = [(group_id, paths) for group_id, paths in sorted(groups.items()) if len(paths) > 1]
    # Preflight every output before writing or deleting any input.
    for group_id, _ in merge_groups:
        _validate_group_id(group_id)
        output_path = input_path / f"{group_id}{output_suffix}.npz"
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite {output_path}; use a new output suffix")

    results: list[MergeResult] = []
    originals_to_delete = []
    for index, (group_id, paths) in enumerate(merge_groups, start=1):
        _show_progress(progress, stream, f"Merging group {index}/{len(merge_groups)} {group_id}: {len(paths)} files")
        source_stats = [path.stat() for path in paths]
        maps = [load_label_map(path) for path in paths]
        if any(label_map.region_ids is not None for label_map in maps):
            raise ValueError("merge original single-slide maps, not previously merged maps")
        model_ids = [str(label_map.metadata.get("model_id", np.asarray("")).item()) for label_map in maps]
        if any(not model_id for model_id in model_ids) or len(set(model_ids)) != 1:
            raise ValueError("all slides must declare the same non-empty model_id")
        spacings = [label_map.spacing for label_map in maps]
        if any(spacing is None for spacing in spacings) or not all(
            np.allclose(spacing, spacings[0], rtol=1e-9, atol=0) for spacing in spacings[1:]
        ):
            raise ValueError("all slides must declare the same physical spacing")
        units = [str(label_map.metadata.get("spacing_unit", np.asarray("")).item()) for label_map in maps]
        if len(set(units)) != 1 or not units[0] or units[0] == "coordinate_units":
            raise ValueError("all slides must declare a common physical spacing_unit (for example um)")
        matrices = [label_map.matrix for label_map in maps]
        merged = merge_matrices_horizontally(matrices, blank_cols)
        regions = merge_matrices_horizontally(
            [np.full(matrix.shape, index, dtype=np.int64) for index, matrix in enumerate(matrices, 1)],
            blank_cols,
        )
        output_path = input_path / f"{group_id}{output_suffix}.npz"
        result = MergeResult(group_id, output_path, paths, tuple(merged.shape))
        results.append(result)
        if dry_run:
            continue
        ranges = []
        col = 0
        for matrix in matrices:
            rows, cols = matrix.shape
            ranges.append((0, rows, col, col + cols))
            col += cols + blank_cols
        payload = dict(
            matrix=merged,
            labels=merged,
            region_ids=regions,
            source_files=np.asarray([str(path) for path in paths]),
            slide_ids=np.asarray([path.stem for path in paths]),
            slide_bounds=np.asarray(ranges, dtype=np.int64),
            group_id=np.asarray(group_id),
            grouping_method=np.asarray("manifest" if patient_manifest else "prefix"),
            model_id=np.asarray(model_ids[0]),
            spacing=np.asarray(spacings[0]),
            spacing_unit=np.asarray(units[0]),
            blank_cols=np.asarray(blank_cols, dtype=np.int64),
            metrics_compatible=np.asarray(True),
            source_metadata=np.asarray(json.dumps([
                {key: np.asarray(value).tolist() for key, value in label_map.metadata.items()}
                for label_map in maps
            ])),
        )
        _save_new_npz(output_path, payload)
        if delete_originals:
            for path, before in zip(paths, source_stats):
                after = path.stat()
                if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                    raise RuntimeError(f"source changed during merge; retained {path}")
            originals_to_delete.extend(zip(paths, source_stats))
    if results and not dry_run:
        for path, before in originals_to_delete:
            after = path.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"source changed during merge; retained {path}")
        merged_sources = {path.resolve() for result in results for path in result.source_paths}
        replaced_groups = {result.group_id for result in results}
        names = []
        for path in matrix_files:
            if path.resolve() in merged_sources:
                continue
            if path.suffix.lower() == ".npz":
                with np.load(path, allow_pickle=False) as data:
                    if "group_id" in data and str(data["group_id"].item()) in replaced_groups:
                        continue
            names.append(path.name)
        names.extend(result.output_path.name for result in results)
        if (map_manifest.read_bytes() if map_manifest.exists() else None) != previous_manifest:
            raise RuntimeError("label_maps.json changed during merge; original files retained")
        temporary_manifest = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=input_path, suffix=".tmp", delete=False) as handle:
                temporary_manifest = Path(handle.name)
                json.dump({"label_maps": sorted(names)}, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_manifest, map_manifest)
        finally:
            if temporary_manifest is not None:
                temporary_manifest.unlink(missing_ok=True)
        for path, _ in originals_to_delete:
            path.unlink()
    _show_progress(progress, stream, f"Merged groups: {len(results)}")
    return results


def _validate_group_id(group_id: str) -> None:
    if not group_id or group_id in {".", ".."} or Path(group_id).name != group_id or "\\" in group_id:
        raise ValueError(f"invalid patient_id: {group_id!r}")


def _manifest_groups(input_dir: Path, manifest: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    seen: set[Path] = set()
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"source_file", "patient_id"}.issubset(reader.fieldnames or []):
            raise ValueError("patient manifest requires source_file,patient_id columns")
        for row in reader:
            group_id = row["patient_id"].strip()
            _validate_group_id(group_id)
            source = Path(row["source_file"])
            path = source if source.is_absolute() else input_dir / source
            if path.resolve().parent != input_dir.resolve() or path.suffix.lower() != ".npz" or not path.is_file():
                raise ValueError(f"invalid source_file in patient manifest: {path}")
            if path.resolve() in seen:
                raise ValueError(f"duplicate source_file in patient manifest: {path}")
            seen.add(path.resolve())
            groups.setdefault(group_id, []).append(path)
    return groups


def _save_new_npz(output_path: Path, payload: dict[str, np.ndarray]) -> None:
    """Validate a completed temporary archive, then atomically publish without replacing."""
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=output_path.parent, suffix=".tmp", delete=False) as handle:
            temp_path = Path(handle.name)
            np.savez(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        with np.load(temp_path, allow_pickle=False) as saved:
            for key, value in payload.items():
                if not np.array_equal(saved[key], value):
                    raise OSError(f"merged archive verification failed: {key}")
        # Unlike replace(), link() also rejects a concurrently created output.
        os.link(temp_path, output_path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def merge_matrices_horizontally(matrices: list[np.ndarray], blank_cols: int = 1) -> np.ndarray:
    """Lay out matrices for display. Use region_ids for patient-level metrics."""
    if not matrices:
        raise ValueError("at least one matrix is required")
    if not isinstance(blank_cols, int) or blank_cols < 0:
        raise ValueError("blank_cols must be a non-negative integer")
    validated = [validate_matrix(matrix) for matrix in matrices]
    max_rows = max(matrix.shape[0] for matrix in validated)
    total_cols = sum(matrix.shape[1] for matrix in validated) + blank_cols * (len(validated) - 1)
    merged = np.zeros((max_rows, total_cols), dtype=np.int64)
    col = 0
    for matrix in validated:
        rows, cols = matrix.shape
        merged[:rows, col:col + cols] = matrix
        col += cols + blank_cols
    return merged


def _show_progress(progress: bool, stream: TextIO, message: str) -> None:
    if progress:
        print(f"[landscape2d-merge] {message}", file=stream, flush=True)
