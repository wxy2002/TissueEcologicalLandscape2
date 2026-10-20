from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class PatchStats:
    label: int
    voxel_count: int
    volume: float
    surface_area: float
    centroid: tuple[float, float, float]
    radius_gyration: float
    shape_index: float
    sphericity: float
    boundary_points: np.ndarray = field(compare=False, repr=False)


def connectivity_structure(connectivity: int) -> np.ndarray:
    if connectivity == 6:
        rank = 1
    elif connectivity == 18:
        rank = 2
    elif connectivity == 26:
        rank = 3
    else:
        raise ValueError("connectivity must be one of 6, 18, or 26")
    return ndimage.generate_binary_structure(3, rank)


def label_patches(mask: np.ndarray, connectivity: int) -> tuple[np.ndarray, int]:
    return ndimage.label(mask, structure=connectivity_structure(connectivity))


def patch_surface_area(mask: np.ndarray, spacing: tuple[float, float, float]) -> float:
    sx, sy, sz = spacing
    face_areas = (sy * sz, sx * sz, sx * sy)
    area = 0.0
    for axis, face_area in enumerate(face_areas):
        padded = np.pad(mask, [(1, 1) if i == axis else (0, 0) for i in range(3)])
        forward = np.take(padded, range(1, padded.shape[axis]), axis=axis)
        backward = np.take(padded, range(0, padded.shape[axis] - 1), axis=axis)
        area += float(np.count_nonzero(forward != backward) * face_area)
    return area


def summarize_patches(
    labels: np.ndarray,
    patch_count: int,
    spacing: tuple[float, float, float],
) -> list[PatchStats]:
    voxel_volume = float(np.prod(spacing))
    stats: list[PatchStats] = []
    objects = ndimage.find_objects(labels, max_label=patch_count)
    for patch_label, bounds in enumerate(objects, 1):
        if bounds is None:
            continue
        mask = labels[bounds] == patch_label
        start = np.asarray([part.start for part in bounds])
        coords = np.argwhere(mask) + start
        voxel_count = int(coords.shape[0])
        volume = voxel_count * voxel_volume
        surface_area = patch_surface_area(mask, spacing)
        physical_coords = coords.astype(float) * np.asarray(spacing, dtype=float)
        centroid_array = physical_coords.mean(axis=0)
        squared_distances = np.sum((physical_coords - centroid_array) ** 2, axis=1)
        radius_gyration = float(np.sqrt(np.mean(squared_distances)))
        shape_index = _shape_index_3d(surface_area, volume)
        sphericity = _sphericity(surface_area, volume)
        stats.append(
            PatchStats(
                label=patch_label,
                voxel_count=voxel_count,
                volume=float(volume),
                surface_area=float(surface_area),
                centroid=tuple(float(value) for value in centroid_array),
                radius_gyration=radius_gyration,
                shape_index=shape_index,
                sphericity=sphericity,
                boundary_points=(np.argwhere(mask & ~ndimage.binary_erosion(mask)) + start) * np.asarray(spacing),
            )
        )
    return stats


def _shape_index_3d(surface_area: float, volume: float) -> float:
    if surface_area <= 0 or volume <= 0:
        return float("nan")
    cube_surface_for_volume = 6.0 * (volume ** (2.0 / 3.0))
    return float(surface_area / cube_surface_for_volume)


def _sphericity(surface_area: float, volume: float) -> float:
    if surface_area <= 0 or volume <= 0:
        return float("nan")
    return float((np.pi ** (1.0 / 3.0)) * ((6.0 * volume) ** (2.0 / 3.0)) / surface_area)
