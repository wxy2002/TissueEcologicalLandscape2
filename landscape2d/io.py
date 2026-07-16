from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

SUPPORTED_SUFFIXES = (".npy", ".npz", ".csv", ".txt")


def validate_matrix(matrix: np.ndarray) -> np.ndarray:
    """Return a validated integer 2D label matrix."""
    array = np.asarray(matrix)
    if array.ndim != 2:
        raise ValueError(f"matrix must be 2D, got shape {array.shape!r}")
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
        values = [int(item) for item in classes]
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
    if any(value <= 0 for value in values):
        raise ValueError("spacing values must be positive")
    return (values[0], values[1])


def load_matrix(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return validate_matrix(np.load(path))
    if path.suffix == ".npz":
        with np.load(path) as data:
            if "matrix" in data:
                return validate_matrix(data["matrix"])
            if "array" in data:
                return validate_matrix(data["array"])
            keys = list(data.keys())
            if len(keys) != 1:
                raise ValueError(
                    f"{path} must contain one array or an array named 'matrix'"
                )
            return validate_matrix(data[keys[0]])
    if path.suffix in {".csv", ".txt"}:
        delimiter = "," if path.suffix == ".csv" else None
        return validate_matrix(np.loadtxt(path, delimiter=delimiter))
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
    return sorted(path for path in input_path.iterdir() if is_matrix_file(path))


def is_matrix_file(path: str | Path) -> bool:
    name = Path(path).name.lower()
    return any(name.endswith(suffix) for suffix in SUPPORTED_SUFFIXES)
