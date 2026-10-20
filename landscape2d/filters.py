from __future__ import annotations

import numpy as np
from numbers import Integral

from .patches import connectivity_structure, label_patches
from .io import iter_regions, validate_regions


def filter_small_patches(
    matrix: np.ndarray,
    min_patch_cells: int = 0,
    connectivity: int = 4,
    region_ids: np.ndarray | None = None,
) -> np.ndarray:
    if not isinstance(min_patch_cells, Integral) or min_patch_cells < 0:
        raise ValueError("min_patch_cells must be a non-negative integer")
    connectivity_structure(connectivity)
    if region_ids is not None:
        regions = validate_regions(matrix, region_ids)
        filtered = np.asarray(matrix).copy()
        for region_id, local, bounds in iter_regions(matrix, regions):
            local_filtered = filter_small_patches(local, min_patch_cells, connectivity)
            inside = regions[bounds] == region_id
            filtered[bounds][inside] = local_filtered[inside]
        return filtered
    if min_patch_cells <= 1:
        return np.asarray(matrix).copy()
    labels = np.asarray(matrix).copy()
    for cls in sorted(int(value) for value in np.unique(labels) if value > 0):
        patch_labels, patch_count = label_patches(labels == cls, connectivity)
        if patch_count == 0:
            continue
        sizes = np.bincount(patch_labels.ravel())
        remove_patch_ids = np.flatnonzero(sizes < min_patch_cells)
        remove_patch_ids = remove_patch_ids[remove_patch_ids != 0]
        if remove_patch_ids.size:
            labels[np.isin(patch_labels, remove_patch_ids)] = 0
    return labels
