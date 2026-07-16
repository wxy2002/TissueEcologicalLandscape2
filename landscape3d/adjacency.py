from __future__ import annotations

from collections import defaultdict

import numpy as np


def surface_and_contact_areas(
    volume: np.ndarray,
    classes: list[int],
    spacing: tuple[float, float, float],
) -> tuple[dict[int, dict[str, float]], dict[tuple[int, int], float]]:
    sx, sy, sz = spacing
    face_areas = (sy * sz, sx * sz, sx * sy)
    class_set = set(classes)
    per_class = {
        cls: {
            "boundary_to_background_area": 0.0,
            "boundary_to_other_classes_area": 0.0,
            "unlike_boundary_area": 0.0,
            "total_surface_area": 0.0,
            "like_internal_face_area": 0.0,
            "total_internal_face_area": 0.0,
        }
        for cls in classes
    }
    pairwise: dict[tuple[int, int], float] = defaultdict(float)

    for axis, face_area in enumerate(face_areas):
        left_slices = [slice(None)] * 3
        right_slices = [slice(None)] * 3
        left_slices[axis] = slice(0, -1)
        right_slices[axis] = slice(1, None)
        left = volume[tuple(left_slices)]
        right = volume[tuple(right_slices)]

        non_background_pair = (left > 0) & (right > 0)
        same = non_background_pair & (left == right)
        diff = non_background_pair & (left != right)

        for cls in classes:
            cls_left = left == cls
            cls_right = right == cls
            per_class[cls]["like_internal_face_area"] += (
                float(np.count_nonzero(same & cls_left)) * face_area
            )
            per_class[cls]["total_internal_face_area"] += (
                float(np.count_nonzero(non_background_pair & (cls_left | cls_right)))
                * face_area
            )
            other_count = int(np.count_nonzero(diff & (cls_left | cls_right)))
            per_class[cls]["boundary_to_other_classes_area"] += other_count * face_area
            per_class[cls]["unlike_boundary_area"] += other_count * face_area

        if np.any(diff):
            left_diff = left[diff]
            right_diff = right[diff]
            for a, b in zip(left_diff.tolist(), right_diff.tolist()):
                if a in class_set and b in class_set:
                    pairwise[tuple(sorted((int(a), int(b))))] += face_area

    for cls in classes:
        class_mask = volume == cls
        per_class[cls]["boundary_to_background_area"] = _boundary_to_value_area(
            volume, class_mask, 0, spacing
        )
        per_class[cls]["total_surface_area"] = (
            per_class[cls]["boundary_to_background_area"]
            + per_class[cls]["boundary_to_other_classes_area"]
            + _boundary_to_outside_area(class_mask, spacing)
        )
    return per_class, dict(pairwise)


def _boundary_to_value_area(
    volume: np.ndarray,
    mask: np.ndarray,
    value: int,
    spacing: tuple[float, float, float],
) -> float:
    sx, sy, sz = spacing
    face_areas = (sy * sz, sx * sz, sx * sy)
    area = 0.0
    for axis, face_area in enumerate(face_areas):
        left_slices = [slice(None)] * 3
        right_slices = [slice(None)] * 3
        left_slices[axis] = slice(0, -1)
        right_slices[axis] = slice(1, None)
        left_mask = mask[tuple(left_slices)]
        right_mask = mask[tuple(right_slices)]
        left_values = volume[tuple(left_slices)]
        right_values = volume[tuple(right_slices)]
        area += float(np.count_nonzero(left_mask & (right_values == value)) * face_area)
        area += float(np.count_nonzero(right_mask & (left_values == value)) * face_area)
    return area


def _boundary_to_outside_area(
    mask: np.ndarray, spacing: tuple[float, float, float]
) -> float:
    sx, sy, sz = spacing
    face_areas = (sy * sz, sx * sz, sx * sy)
    area = 0.0
    for axis, face_area in enumerate(face_areas):
        first = np.take(mask, 0, axis=axis)
        last = np.take(mask, mask.shape[axis] - 1, axis=axis)
        area += float((np.count_nonzero(first) + np.count_nonzero(last)) * face_area)
    return area
