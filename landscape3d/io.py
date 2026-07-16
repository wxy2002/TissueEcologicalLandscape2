from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

SUPPORTED_SUFFIXES = (".npy", ".npz", ".nii", ".nii.gz")


def validate_volume(volume: np.ndarray) -> np.ndarray:
    """Return a validated integer 3D label volume."""
    array = np.asarray(volume)
    if array.ndim != 3:
        raise ValueError(f"volume must be 3D, got shape {array.shape!r}")
    if np.issubdtype(array.dtype, np.floating):
        if not np.all(np.isfinite(array)):
            raise ValueError("volume labels must be finite")
        if not np.all(np.equal(array, np.rint(array))):
            raise ValueError("volume must contain integer labels")
        array = np.rint(array).astype(np.int64)
    elif not np.issubdtype(array.dtype, np.integer):
        raise ValueError("volume must contain integer labels")
    if np.any(array < 0):
        raise ValueError("volume labels must be non-negative integers")
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


def parse_spacing(spacing: str | Iterable[float]) -> tuple[float, float, float]:
    if isinstance(spacing, str):
        values = [float(item.strip()) for item in spacing.split(",")]
    else:
        values = [float(item) for item in spacing]
    if len(values) != 3:
        raise ValueError("spacing must have exactly three values: sx,sy,sz")
    if any(value <= 0 for value in values):
        raise ValueError("spacing values must be positive")
    return (values[0], values[1], values[2])


def load_volume(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return validate_volume(np.load(path))
    if path.suffix == ".npz":
        with np.load(path) as data:
            if "volume" in data:
                return validate_volume(data["volume"])
            keys = list(data.keys())
            if len(keys) != 1:
                raise ValueError(
                    f"{path} must contain one array or an array named 'volume'"
                )
            return validate_volume(data[keys[0]])
    if _has_suffix(path, (".nii", ".nii.gz")):
        return _load_nifti(path)
    raise ValueError(f"unsupported input file: {path}")


def sample_id_from_path(path: str | Path) -> str:
    path = Path(path)
    name = path.name
    for suffix in SUPPORTED_SUFFIXES:
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def iter_volume_files(input_dir: str | Path) -> list[Path]:
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not input_path.is_dir():
        raise NotADirectoryError(input_path)
    return sorted(path for path in input_path.iterdir() if is_volume_file(path))


def is_volume_file(path: str | Path) -> bool:
    return _has_suffix(Path(path), SUPPORTED_SUFFIXES)


def _has_suffix(path: Path, suffixes: tuple[str, ...]) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in suffixes)


def _load_nifti(path: Path) -> np.ndarray:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError(
            "Reading .nii/.nii.gz files requires nibabel. Install with: "
            "pip install nibabel"
        ) from exc

    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    return validate_volume(data)
