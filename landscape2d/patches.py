from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class PatchStats:
    label: int
    cell_count: int
    area: float
    perimeter: float
    centroid: tuple[float, float]
    radius_gyration: float
    shape_index_2d: float
    compactness: float
    fractal_dimension_2d: float
    related_circumscribing_circle: float
    contiguity_index: float


def connectivity_structure(connectivity: int) -> np.ndarray:
    if connectivity == 4:
        rank = 1
    elif connectivity == 8:
        rank = 2
    else:
        raise ValueError("connectivity must be one of 4 or 8")
    return ndimage.generate_binary_structure(2, rank)


def label_patches(mask: np.ndarray, connectivity: int) -> tuple[np.ndarray, int]:
    return ndimage.label(mask, structure=connectivity_structure(connectivity))


def patch_perimeter(mask: np.ndarray, spacing: tuple[float, float]) -> float:
    sx, sy = spacing
    edge_lengths = (sx, sy)
    perimeter = 0.0
    for axis, edge_length in enumerate(edge_lengths):
        padded = np.pad(mask, [(1, 1) if i == axis else (0, 0) for i in range(2)])
        forward = np.take(padded, range(1, padded.shape[axis]), axis=axis)
        backward = np.take(padded, range(0, padded.shape[axis] - 1), axis=axis)
        perimeter += float(np.count_nonzero(forward != backward) * edge_length)
    return perimeter


def summarize_patches(
    labels: np.ndarray,
    patch_count: int,
    spacing: tuple[float, float],
) -> list[PatchStats]:
    cell_area = float(np.prod(spacing))
    stats: list[PatchStats] = []
    for patch_label in range(1, patch_count + 1):
        mask = labels == patch_label
        coords = np.argwhere(mask)
        cell_count = int(coords.shape[0])
        area = cell_count * cell_area
        perimeter = patch_perimeter(mask, spacing)
        physical_coords = coords.astype(float) * np.asarray((spacing[1], spacing[0]), dtype=float)
        centroid_array = physical_coords.mean(axis=0)
        squared_distances = np.sum((physical_coords - centroid_array) ** 2, axis=1)
        radius_gyration = float(np.sqrt(np.mean(squared_distances)))
        stats.append(
            PatchStats(
                label=patch_label,
                cell_count=cell_count,
                area=float(area),
                perimeter=float(perimeter),
                centroid=tuple(float(value) for value in centroid_array),
                radius_gyration=radius_gyration,
                shape_index_2d=_shape_index_2d(perimeter, area),
                compactness=_compactness(perimeter, area),
                fractal_dimension_2d=_fractal_dimension_2d(perimeter, area),
                related_circumscribing_circle=_related_circumscribing_circle(
                    physical_coords, area, spacing
                ),
                contiguity_index=_contiguity_index(mask),
            )
        )
    return stats


def _shape_index_2d(perimeter: float, area: float) -> float:
    if perimeter <= 0 or area <= 0:
        return float("nan")
    return float(perimeter / (4.0 * np.sqrt(area)))


def _compactness(perimeter: float, area: float) -> float:
    if perimeter <= 0 or area <= 0:
        return float("nan")
    return float((4.0 * np.pi * area) / (perimeter * perimeter))


def _fractal_dimension_2d(perimeter: float, area: float) -> float:
    if perimeter <= 4 or area <= 1:
        return float("nan")
    return float(2.0 * np.log(0.25 * perimeter) / np.log(area))


def _related_circumscribing_circle(
    physical_coords: np.ndarray,
    area: float,
    spacing: tuple[float, float],
) -> float:
    if area <= 0 or physical_coords.size == 0:
        return float("nan")
    centroid = physical_coords.mean(axis=0)
    half_diagonal = 0.5 * float(np.hypot(spacing[0], spacing[1]))
    radius = float(np.max(np.linalg.norm(physical_coords - centroid, axis=1))) + half_diagonal
    circle_area = float(np.pi * radius * radius)
    return 1.0 - float(area / circle_area) if circle_area > 0 else float("nan")


def _contiguity_index(mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    weights = np.asarray([[1, 2, 1], [2, 1, 2], [1, 2, 1]], dtype=float)
    scores = ndimage.convolve(mask.astype(float), weights, mode="constant", cval=0.0)
    observed = float(scores[mask].sum())
    maximum = float(weights.sum() * np.count_nonzero(mask))
    minimum = float(np.count_nonzero(mask))
    if maximum == minimum:
        return float("nan")
    return (observed - minimum) / (maximum - minimum)
