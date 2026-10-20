"""Rasterize tiles only when their spatial grid is known."""
from __future__ import annotations

import numpy as np

from .io import parse_spacing


def add_grid_arguments(parser) -> None:
    parser.add_argument("--grid-stride", type=parse_spacing, help="Coordinate stride sx,sy; required for spatial maps unless stored in the feature file.")
    parser.add_argument("--grid-origin", type=lambda value: tuple(float(v) for v in value.split(",")), help="Grid coordinate origin x,y; default: minimum observed coordinates.")
    parser.add_argument("--coordinate-level", type=int, default=None, help="Coordinate pyramid level; default: metadata or level 0.")
    parser.add_argument("--coord-downsample", type=float, default=None, help="Coordinate-to-level-0 downsample; required for nonzero levels.")
    parser.add_argument("--mpp", type=parse_spacing, help="Level-0 micrometers per pixel sx,sy; enables physical map spacing.")


def grid_options(args) -> dict:
    return {key: getattr(args, key) for key in ("grid_stride", "grid_origin", "coordinate_level", "coord_downsample", "mpp")}


def labels_to_matrix(
    labels: np.ndarray,
    coords: np.ndarray | None,
    stride: tuple[float, float] | None = None,
    origin: tuple[float, float] | None = None,
    grid_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    values = np.asarray(labels)
    if values.size == 0 or not np.issubdtype(values.dtype, np.integer) or np.any(values < 0):
        raise ValueError("tile labels must be non-empty non-negative integers")
    values = values.ravel().astype(np.int64)
    if coords is None or stride is None:
        raise ValueError("spatial maps require coords and an explicit grid stride; file order is not spatial adjacency")
    points = np.asarray(coords, dtype=float)
    if points.shape != (values.size, 2) or not np.all(np.isfinite(points)):
        raise ValueError("coords must be finite with shape (number of labels, 2)")
    step = np.asarray(parse_spacing(stride))
    start = points.min(axis=0) if origin is None else np.asarray(origin, dtype=float)
    if start.shape != (2,) or not np.all(np.isfinite(start)):
        raise ValueError("grid origin must contain two finite coordinates")
    positions = (points - start) / step
    if not np.all(np.isfinite(positions)) or not np.allclose(positions, np.rint(positions), atol=1e-6, rtol=0):
        raise ValueError("tile coordinates are not aligned to grid origin/stride")
    if np.any(positions < 0) or np.any(positions > np.iinfo(np.int32).max):
        raise ValueError("tile grid indices are out of range")
    positions = np.rint(positions).astype(np.int64)
    if np.unique(positions, axis=0).shape[0] != values.size:
        raise ValueError("duplicate tile coordinates are not allowed")
    shape = (int(positions[:, 1].max()) + 1, int(positions[:, 0].max()) + 1)
    if grid_shape is not None:
        if len(grid_shape) != 2 or any(int(v) != v or v < 1 for v in grid_shape) or any(a > b for a, b in zip(shape, grid_shape)):
            raise ValueError("grid_shape must be a positive (rows, cols) shape containing every tile")
        shape = tuple(int(v) for v in grid_shape)
    matrix = np.zeros(shape, dtype=np.int64)
    matrix[positions[:, 1], positions[:, 0]] = values
    return matrix
