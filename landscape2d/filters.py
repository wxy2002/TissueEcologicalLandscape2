from __future__ import annotations

import numpy as np

from .patches import label_patches


def filter_small_patches(
    matrix: np.ndarray,
    min_patch_cells: int = 0,
    connectivity: int = 4,
) -> np.ndarray:
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
