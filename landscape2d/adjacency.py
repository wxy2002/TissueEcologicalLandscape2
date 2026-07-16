from __future__ import annotations

from collections import defaultdict

import numpy as np


def edge_and_contact_lengths(
    matrix: np.ndarray,
    classes: list[int],
    spacing: tuple[float, float],
) -> tuple[dict[int, dict[str, float]], dict[tuple[int, int], float]]:
    sx, sy = spacing
    edge_lengths = (sx, sy)
    class_set = set(classes)
    per_class = {
        cls: {
            "boundary_to_background_length": 0.0,
            "boundary_to_other_classes_length": 0.0,
            "unlike_boundary_length": 0.0,
            "total_edge_length": 0.0,
            "like_internal_edge_length": 0.0,
            "total_internal_edge_length": 0.0,
        }
        for cls in classes
    }
    pairwise: dict[tuple[int, int], float] = defaultdict(float)

    for axis, edge_length in enumerate(edge_lengths):
        left_slices = [slice(None)] * 2
        right_slices = [slice(None)] * 2
        left_slices[axis] = slice(0, -1)
        right_slices[axis] = slice(1, None)
        left = matrix[tuple(left_slices)]
        right = matrix[tuple(right_slices)]

        non_background_pair = (left > 0) & (right > 0)
        same = non_background_pair & (left == right)
        diff = non_background_pair & (left != right)

        for cls in classes:
            cls_left = left == cls
            cls_right = right == cls
            per_class[cls]["like_internal_edge_length"] += (
                float(np.count_nonzero(same & cls_left)) * edge_length
            )
            per_class[cls]["total_internal_edge_length"] += (
                float(np.count_nonzero(non_background_pair & (cls_left | cls_right)))
                * edge_length
            )
            other_count = int(np.count_nonzero(diff & (cls_left | cls_right)))
            per_class[cls]["boundary_to_other_classes_length"] += other_count * edge_length
            per_class[cls]["unlike_boundary_length"] += other_count * edge_length

        if np.any(diff):
            for a, b in zip(left[diff].tolist(), right[diff].tolist()):
                if a in class_set and b in class_set:
                    pairwise[tuple(sorted((int(a), int(b))))] += edge_length

    for cls in classes:
        class_mask = matrix == cls
        per_class[cls]["boundary_to_background_length"] = _boundary_to_value_length(
            matrix, class_mask, 0, spacing
        )
        per_class[cls]["total_edge_length"] = (
            per_class[cls]["boundary_to_background_length"]
            + per_class[cls]["boundary_to_other_classes_length"]
            + _boundary_to_outside_length(class_mask, spacing)
        )
    return per_class, dict(pairwise)


def _boundary_to_value_length(
    matrix: np.ndarray,
    mask: np.ndarray,
    value: int,
    spacing: tuple[float, float],
) -> float:
    sx, sy = spacing
    edge_lengths = (sx, sy)
    length = 0.0
    for axis, edge_length in enumerate(edge_lengths):
        left_slices = [slice(None)] * 2
        right_slices = [slice(None)] * 2
        left_slices[axis] = slice(0, -1)
        right_slices[axis] = slice(1, None)
        left_mask = mask[tuple(left_slices)]
        right_mask = mask[tuple(right_slices)]
        left_values = matrix[tuple(left_slices)]
        right_values = matrix[tuple(right_slices)]
        length += float(np.count_nonzero(left_mask & (right_values == value)) * edge_length)
        length += float(np.count_nonzero(right_mask & (left_values == value)) * edge_length)
    return length


def _boundary_to_outside_length(mask: np.ndarray, spacing: tuple[float, float]) -> float:
    sx, sy = spacing
    edge_lengths = (sx, sy)
    length = 0.0
    for axis, edge_length in enumerate(edge_lengths):
        first = np.take(mask, 0, axis=axis)
        last = np.take(mask, mask.shape[axis] - 1, axis=axis)
        length += float((np.count_nonzero(first) + np.count_nonzero(last)) * edge_length)
    return length
