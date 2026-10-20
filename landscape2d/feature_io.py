"""Feature files, spatial metadata, and row-preserving sampled reads."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import hashlib

import numpy as np

from .io import parse_spacing


@dataclass(frozen=True)
class FeatureBlock:
    sample_id: str
    source_path: Path
    features: np.ndarray
    coords: np.ndarray | None = None
    grid_stride: tuple[float, float] | None = None
    grid_origin: tuple[float, float] | None = None
    coordinate_level: int = 0
    coord_downsample: float = 1.0
    mpp: tuple[float, float] | None = None
    row_indices: np.ndarray | None = None

    def __post_init__(self):
        if self.coords is not None and self.coords.shape[0] != self.features.shape[0]:
            raise ValueError("coords and features length mismatch")


def _grid_metadata(values) -> dict[str, object]:
    metadata: dict[str, object] = {}
    stride = values.get("grid_stride", values.get("stride"))
    if stride is None and "stride_x" in values and "stride_y" in values:
        stride = (values["stride_x"], values["stride_y"])
    if stride is not None:
        metadata["grid_stride"] = parse_spacing(np.asarray(stride).ravel())
    if "grid_origin" in values:
        origin = np.asarray(values["grid_origin"], dtype=float).ravel()
        if origin.shape != (2,) or not np.all(np.isfinite(origin)):
            raise ValueError("grid_origin must contain two finite coordinates")
        metadata["grid_origin"] = tuple(float(value) for value in origin)
    level = np.asarray(values.get("coordinate_level", 0)).item()
    downsample = float(np.asarray(values.get("coord_downsample", 1.0)).item())
    if not isinstance(level, (int, float)) or not np.isfinite(level) or int(level) != level or level < 0:
        raise ValueError("coordinate_level must be a non-negative integer")
    if not np.isfinite(downsample) or downsample <= 0 or (level > 0 and downsample <= 1):
        raise ValueError("coord_downsample must be finite and positive, and greater than 1 for nonzero levels")
    metadata.update(coordinate_level=int(level), coord_downsample=downsample)
    if "mpp" in values:
        metadata["mpp"] = parse_spacing(np.asarray(values["mpp"]).ravel())
    return metadata


def _with_grid_overrides(block, grid_stride, grid_origin, coordinate_level, coord_downsample, mpp):
    values = {key: getattr(block, key) for key in ("grid_stride", "grid_origin", "coordinate_level", "coord_downsample", "mpp") if getattr(block, key) is not None}
    values.update({key: value for key, value in dict(grid_stride=grid_stride, grid_origin=grid_origin,
                  coordinate_level=coordinate_level, coord_downsample=coord_downsample, mpp=mpp).items() if value is not None})
    return replace(block, **_grid_metadata(values))


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_features(path: str | Path) -> np.ndarray:
    return load_feature_block(path).features


def load_feature_block(path: str | Path) -> FeatureBlock:
    path = Path(path)
    if path.suffix == ".npy":
        return FeatureBlock(_sample_id(path), path, validate_features(np.load(path)))
    if path.suffix == ".npz":
        with np.load(path) as data:
            for key in ("features", "x", "array"):
                if key in data:
                    coords = validate_coords(data["coords"]) if "coords" in data else None
                    return FeatureBlock(
                        _sample_id(path),
                        path,
                        validate_features(data[key]),
                        coords=coords,
                        **_grid_metadata(data),
                    )
            keys = list(data.keys())
            if len(keys) != 1:
                raise ValueError(
                    f"{path} must contain one array or an array named 'features'"
                )
            return FeatureBlock(_sample_id(path), path, validate_features(data[keys[0]]))
    if path.suffix in {".csv", ".txt"}:
        delimiter = "," if path.suffix == ".csv" else None
        return FeatureBlock(
            _sample_id(path),
            path,
            validate_features(np.loadtxt(path, delimiter=delimiter, ndmin=2)),
        )
    if path.suffix in {".h5", ".hdf5"}:
        return load_h5_feature_block(path)
    raise ValueError(f"unsupported feature file: {path}")


def load_feature_sample(
    path: str | Path,
    indices: np.ndarray | None = None,
) -> FeatureBlock:
    path = Path(path)
    if indices is None:
        return load_feature_block(path)
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        raise ValueError("sample indices cannot be empty")
    if path.suffix in {".h5", ".hdf5"}:
        return load_h5_feature_block(path, indices=indices)
    block = load_feature_block(path)
    return FeatureBlock(
        block.sample_id,
        block.source_path,
        validate_features(block.features[indices]),
        coords=validate_coords(block.coords[indices]) if block.coords is not None else None,
        grid_stride=block.grid_stride,
        grid_origin=block.grid_origin,
        coordinate_level=block.coordinate_level,
        coord_downsample=block.coord_downsample,
        mpp=block.mpp,
        row_indices=indices,
    )


def load_h5_feature_block(
    path: str | Path,
    feature_key: str = "feats",
    coords_key: str = "coords",
    indices: np.ndarray | None = None,
) -> FeatureBlock:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("Reading .h5 feature files requires h5py.") from exc
    path = Path(path)
    with h5py.File(path, "r") as handle:
        if feature_key not in handle:
            raise ValueError(f"{path} does not contain dataset {feature_key!r}")
        if indices is None:
            features = validate_features(handle[feature_key][()])
            coords = validate_coords(handle[coords_key][()]) if coords_key in handle else None
        else:
            order = np.argsort(indices)
            sorted_indices = np.asarray(indices[order], dtype=np.int64)
            inverse = np.argsort(order)
            features = validate_features(handle[feature_key][sorted_indices][inverse])
            coords = (
                validate_coords(handle[coords_key][sorted_indices][inverse])
                if coords_key in handle
                else None
            )
        metadata = dict(handle.attrs)
        if coords_key in handle:
            metadata.update(dict(handle[coords_key].attrs))
        grid_metadata = _grid_metadata(metadata)
    if coords is not None and coords.shape[0] != features.shape[0]:
        raise ValueError("coords and features length mismatch")
    return FeatureBlock(_sample_id(path), path, features, coords=coords, row_indices=indices, **grid_metadata)


def iter_feature_files(input_path: str | Path) -> list[Path]:
    path = Path(input_path)
    suffixes = (".npy", ".npz", ".csv", ".txt", ".h5", ".hdf5")
    if path.is_file():
        if path.name.lower().endswith(suffixes):
            return [path]
        raise ValueError(f"unsupported feature file: {path}")
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_dir():
        raise NotADirectoryError(path)
    files = sorted(item for item in path.iterdir() if item.is_file() and item.name.lower().endswith(suffixes))
    if not files:
        raise ValueError(f"no feature files found in {path}")
    return files


def feature_count(path: str | Path) -> int:
    path = Path(path)
    if path.suffix in {".h5", ".hdf5"}:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("Reading .h5 feature files requires h5py.") from exc
        with h5py.File(path, "r") as handle:
            if "feats" not in handle:
                raise ValueError(f"{path} does not contain dataset 'feats'")
            return int(handle["feats"].shape[0])
    return int(load_feature_block(path).features.shape[0])


def validate_features(features: np.ndarray) -> np.ndarray:
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {array.shape!r}")
    if array.shape[0] < 1:
        raise ValueError("features must contain at least one sample")
    if array.shape[1] < 1:
        raise ValueError("features must contain at least one feature column")
    if not np.all(np.isfinite(array)):
        raise ValueError("features must be finite")
    return array


def validate_coords(coords: np.ndarray) -> np.ndarray:
    array = np.asarray(coords)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"coords must have shape n x 2, got {array.shape!r}")
    if not np.all(np.isfinite(array)):
        raise ValueError("coords must be finite")
    return array


def _sample_id(path: Path) -> str:
    name = path.name
    for suffix in (".hdf5", ".h5", ".npy", ".npz", ".csv", ".txt"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


