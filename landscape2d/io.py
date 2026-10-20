from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

SUPPORTED_SUFFIXES = (".npy", ".npz", ".csv", ".txt")


@dataclass(frozen=True)
class LabelMap:
    matrix: np.ndarray
    region_ids: np.ndarray | None = None
    spacing: tuple[float, float] | None = None
    metadata: dict[str, np.ndarray] = field(default_factory=dict)


def validate_matrix(matrix: np.ndarray) -> np.ndarray:
    """Return a validated integer 2D label matrix."""
    array = np.asarray(matrix)
    if array.ndim != 2:
        raise ValueError(f"matrix must be 2D, got shape {array.shape!r}")
    if 0 in array.shape:
        raise ValueError("matrix dimensions must be non-empty")
    if np.issubdtype(array.dtype, np.floating):
        if not np.all(np.isfinite(array)):
            raise ValueError("matrix labels must be finite")
        if not np.all(np.equal(array, np.rint(array))):
            raise ValueError("matrix must contain integer labels")
        array = np.rint(array).astype(np.int64)
    elif not np.issubdtype(array.dtype, np.integer):
        raise ValueError("matrix must contain integer labels")
    if np.any(array < 0):
        raise ValueError("matrix labels must be non-negative integers")
    return array


def parse_classes(classes: str | Iterable[int] | None) -> list[int] | None:
    if classes is None:
        return None
    if isinstance(classes, str):
        if not classes.strip():
            return None
        values = [int(item.strip()) for item in classes.split(",")]
    else:
        values = []
        for item in classes:
            value = int(item)
            if item != value:
                raise ValueError("classes must be integer labels")
            values.append(value)
    if any(value <= 0 for value in values):
        raise ValueError("classes must be positive integers; 0 is background")
    return sorted(dict.fromkeys(values))


def parse_spacing(spacing: str | Iterable[float]) -> tuple[float, float]:
    if isinstance(spacing, str):
        values = [float(item.strip()) for item in spacing.split(",")]
    else:
        values = [float(item) for item in spacing]
    if len(values) != 2:
        raise ValueError("spacing must have exactly two values: sx,sy")
    if any(not np.isfinite(value) or value <= 0 for value in values):
        raise ValueError("spacing values must be finite and positive")
    return (values[0], values[1])


def load_matrix(path: str | Path) -> np.ndarray:
    label_map = load_label_map(path)
    if label_map.region_ids is not None:
        raise ValueError("merged maps require load_label_map() and compute_metrics(..., region_ids=...)")
    return label_map.matrix


def validate_regions(matrix: np.ndarray, region_ids: np.ndarray) -> np.ndarray:
    regions = validate_matrix(region_ids)
    if regions.shape != matrix.shape:
        raise ValueError("region_ids must have the same shape as matrix")
    if np.any((matrix > 0) & (regions == 0)):
        raise ValueError("every tissue cell must belong to a positive slide region")
    return regions


def iter_regions(matrix: np.ndarray, region_ids: np.ndarray | None = None):
    """Yield independent slide views, their canvas slices, and original region IDs."""
    if region_ids is None:
        yield 0, matrix, (slice(0, matrix.shape[0]), slice(0, matrix.shape[1]))
        return
    regions = validate_regions(matrix, region_ids)
    for region_id in np.unique(regions):
        if region_id == 0:
            continue
        rows, cols = np.nonzero(regions == region_id)
        bounds = (slice(int(rows.min()), int(rows.max()) + 1), slice(int(cols.min()), int(cols.max()) + 1))
        yield int(region_id), np.where(regions[bounds] == region_id, matrix[bounds], 0), bounds


def load_label_map(path: str | Path) -> LabelMap:
    """Load labels together with the slide boundaries needed for spatial metrics."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return LabelMap(validate_matrix(np.load(path, allow_pickle=False)))
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            key = next((key for key in ("matrix", "array", "labels") if key in data), None)
            if key is None:
                keys = list(data.keys())
                if len(keys) != 1:
                    raise ValueError(f"{path} has no spatial label matrix")
                key = keys[0]
            matrix = validate_matrix(data[key])
            metadata = {name: data[name] for name in ("model_id", "spacing_unit", "grid_stride", "grid_origin", "coordinate_level", "coord_downsample", "mpp", "source_files", "slide_ids", "slide_bounds", "group_id") if name in data}
            if "metrics_compatible" in data and not bool(data["metrics_compatible"].item()):
                raise ValueError(f"{path} is a visualization-only merge; verify model and physical spacing first")
            if "source_files" in data and data["source_files"].size > 1 and "region_ids" not in data:
                raise ValueError(f"{path} is a legacy merged map without slide boundaries; rebuild from original slides")
            regions = validate_regions(matrix, data["region_ids"]) if "region_ids" in data else None
            spacing = parse_spacing(data["spacing"]) if "spacing" in data else None
            return LabelMap(matrix, regions, spacing, metadata)
    if suffix in {".csv", ".txt"}:
        delimiter = "," if suffix == ".csv" else None
        return LabelMap(validate_matrix(np.loadtxt(path, delimiter=delimiter, ndmin=2)))
    raise ValueError(f"unsupported input file: {path}")


def sample_id_from_path(path: str | Path) -> str:
    path = Path(path)
    name = path.name
    for suffix in SUPPORTED_SUFFIXES:
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def iter_matrix_files(input_dir: str | Path) -> list[Path]:
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not input_path.is_dir():
        raise NotADirectoryError(input_path)
    manifest = input_path / "label_maps.json"
    if manifest.is_file():
        import json

        payload = json.loads(manifest.read_text(encoding="utf-8"))
        paths = [input_path / name for name in payload["label_maps"]]
        if len(set(paths)) != len(paths):
            raise ValueError("duplicate paths in label_maps.json")
        for path in paths:
            if path.resolve().parent != input_path.resolve() or not path.is_file():
                raise ValueError(f"invalid label-map manifest entry: {path}")
        return sorted(paths)
    return sorted(path for path in input_path.iterdir() if path.is_file() and is_matrix_file(path))


def is_matrix_file(path: str | Path) -> bool:
    path = Path(path)
    name = path.name.lower()
    if name in {"all_clusters.npz", "cluster_metrics.csv"}:
        return False
    if path.suffix.lower() == ".csv" and path.is_file():
        with path.open(encoding="utf-8-sig") as handle:
            if handle.readline().startswith("sample_id,source_file,"):
                return False
    return any(name.endswith(suffix) for suffix in SUPPORTED_SUFFIXES)
